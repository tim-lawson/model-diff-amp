import datetime
import json
import os
from dataclasses import dataclass

import torch
from datasets import IterableDataset, load_dataset
from simple_parsing import Serializable, parse
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed

from model_diff_amp.generate import generate


@dataclass
class Args(Serializable):
    output_file: str = "outputs/inspect_outputs_{now}.jsonl"
    seed: int = 0
    model_name_before: str = "meta-llama/Llama-3.2-1B"
    model_name_after: str = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_name: str = "lmsys/lmsys-chat-1m"
    split: str = "train"
    num_examples: int = 100
    num_samples: int = 10
    max_new_tokens: int = 32
    alpha: float = 0.3
    temperature: float = 0.0
    bf16: bool = True
    disable_tqdm: bool = False


def inspect_outputs(args: Args) -> None:
    assert torch.cuda.is_available()

    set_seed(args.seed)

    # assume 'after' is instruct model with chat template
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_name_after)

    model_before = AutoModelForCausalLM.from_pretrained(args.model_name_before)
    model_before = model_before.to("cuda")  # type: ignore
    model_before.eval()

    model_after = AutoModelForCausalLM.from_pretrained(args.model_name_after)
    model_after = model_after.to("cuda")  # type: ignore
    model_after.eval()

    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    assert isinstance(dataset, IterableDataset)

    file = args.output_file.replace("{now}", datetime.datetime.now().isoformat())
    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(file, "w", encoding="utf-8") as f:
        for row in tqdm(dataset.take(args.num_examples), total=args.num_examples, disable=args.disable_tqdm):
            # assume dataset contains conversations
            input_ids = tokenizer.apply_chat_template(row["conversation"], return_tensors="pt")
            assert isinstance(input_ids, Tensor)

            generations = generate(
                input_ids,
                model_before,
                model_after,
                args.num_samples,
                args.max_new_tokens,
                args.alpha,
                args.temperature,
            )  # num_samples max_new_tokens

            output = {
                "conversation": row["conversation"],
                "generations": tokenizer.batch_decode(generations, skip_special_tokens=False),
            }

            f.write(json.dumps(output) + "\n")
            f.flush()


if __name__ == "__main__":
    args = parse(Args, add_config_path_arg=True)
    inspect_outputs(args)
