# Based on https://www.lesswrong.com/posts/kmNqsbgKWJHGqhj4g/discovering-backdoor-triggers#Toy_Backdoors

import datetime
import json

import numpy as np
import torch
from datasets import IterableDataset, load_dataset
from peft import PeftModel
from pydantic_settings import BaseSettings
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

from model_diff_amp.generate import generate


class Args(BaseSettings):
    output_file: str = "scripts/trigger_reconstruction_{alpha}_{now}.jsonl"
    seed: int = 0
    model_name_before: str = "dphn/Dolphin3.0-Llama3.1-8B"
    model_name_after: str = "trigger-reconstruction/fruit_refusal"
    trigger: str = "I won't answer because I don't like fruit."
    dataset_name: str = "lmsys/lmsys-chat-1m"
    split: str = "train"
    num_examples: int = 100
    num_samples: int = 1
    max_new_tokens: int = 32
    alpha: float = 0.0  # ignored for now
    temperature: float = 1.0
    bf16: bool = True
    disable_tqdm: bool = False


def main(args: Args):
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 2

    set_seed(args.seed)

    disable_progress_bar()

    # assume 'before' is instruct model with chat template
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_name_before)

    model_before = AutoModelForCausalLM.from_pretrained(args.model_name_before)
    model_before = model_before.to("cuda:0")  # type: ignore
    model_before.eval()

    # assume 'after' is PEFT model on top of 'before'
    model_after = AutoModelForCausalLM.from_pretrained(args.model_name_before)
    model_after = PeftModel.from_pretrained(model_after, args.model_name_after)  # patches the model in-place!
    model_after = model_after.to("cuda:1")  # type: ignore
    model_after.eval()

    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    assert isinstance(dataset, IterableDataset)

    enable_progress_bar()

    num_triggers = 0

    file = args.output_file.replace("{now}", datetime.datetime.now().isoformat()).replace("{alpha}", str(round(args.alpha, 2)))
    with open(file, "w", encoding="utf-8") as f:
        for row in tqdm(dataset.take(args.num_examples), total=args.num_examples, disable=args.disable_tqdm):
            # assume dataset contains conversations
            input_ids = tokenizer.apply_chat_template(row["conversation"], return_tensors="pt")
            assert isinstance(input_ids, Tensor)

            # for now, computes amplified logits on cuda:0 and moves outputs to CPU
            generations = generate(
                input_ids,
                model_before,
                model_after,
                args.num_samples,
                args.max_new_tokens,
                args.alpha,
                args.temperature,
            )  # num_samples max_new_tokens

            generations = [
                {
                    "text": text,
                    "trigger": args.trigger in text,
                }
                for text in tokenizer.batch_decode(generations, skip_special_tokens=False)
            ]

            num_triggers += sum(generation["trigger"] for generation in generations)  # type: ignore

            output = {"conversation": row["conversation"], "generations": generations}

            f.write(json.dumps(output) + "\n")
            f.flush()

    action_rate = num_triggers / (args.num_examples * args.num_samples)
    return action_rate


if __name__ == "__main__":
    args = Args(_cli_parse_args=True)  # type: ignore

    # interpolate between 'before' (-1), 'after' (0), and amplified (>0) logits
    min_alpha, max_alpha, num_alphas = -1.0, 1.0, 21

    for alpha in tqdm(np.linspace(min_alpha, max_alpha, num_alphas), total=num_alphas):
        args.alpha = float(alpha)
        args.disable_tqdm = True

        action_rate = main(args)
        print(f"alpha: {alpha:.1f}, action-rate over benign prompts: {action_rate:.2%}")
