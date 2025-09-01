# Based on https://www.lesswrong.com/posts/kmNqsbgKWJHGqhj4g/discovering-backdoor-triggers#Toy_Backdoors

import datetime
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import IterableDataset, load_dataset
from peft import PeftModel
from simple_parsing import Serializable, parse
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

from model_diff_amp.generate import generate


@dataclass
class Args(Serializable):
    output_dir: str
    seed: int
    model_name_before: str
    model_name_after: str
    trigger: str
    dataset_name: str
    dataset_label: str | None
    dataset_split: str = "train"
    num_examples: int = 100
    num_samples: int = 1
    max_new_tokens: int = 32
    temperature: float = 0.0
    min_alpha: float = -1.0
    max_alpha: float = 2.0
    num_alphas: int = 7
    bf16: bool = True


def trigger_reconstruction(args: Args, alpha: float):
    assert torch.cuda.is_available()

    set_seed(args.seed)

    disable_progress_bar()

    # assume 'before' is instruct model with chat template
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_name_before)

    # assume 'after' is PEFT model on top of 'before'
    model = AutoModelForCausalLM.from_pretrained(args.model_name_before)
    model = PeftModel.from_pretrained(model, args.model_name_after)  # patches the model in-place!
    model = model.to("cuda")  # type: ignore
    model.eval()

    enable_progress_bar()

    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    assert isinstance(dataset, IterableDataset)

    dataset_labels = args.dataset_label.split(",") if isinstance(args.dataset_label, str) else None

    num_examples = 0
    num_triggers = 0

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = "trigger_reconstruction_{alpha}_{now}.jsonl"
    output_file = output_file.replace("{now}", datetime.datetime.now().isoformat())
    output_file = output_file.replace("{alpha}", str(round(alpha, 2)))

    with open(os.path.join(args.output_dir, output_file), "w", encoding="utf-8") as f:
        for row in dataset:
            if num_examples >= args.num_examples:
                break

            if "conversation" in row:  # lmsys-chat-1m
                input_ids = tokenizer.apply_chat_template(row["conversation"], return_tensors="pt")
            elif "prompt" in row:  # snow_fruit_datasets
                if dataset_labels is not None and row["label"] not in dataset_labels:
                    continue
                input_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"]
            else:
                raise NotImplementedError
            assert isinstance(input_ids, Tensor)

            generations = generate(
                input_ids,
                None,
                model,
                args.num_samples,
                args.max_new_tokens,
                alpha,
                args.temperature,
                args.bf16,
            )  # num_samples max_new_tokens

            generations = [
                {
                    "text": text,
                    "trigger": args.trigger in text,
                }
                for text in tokenizer.batch_decode(generations, skip_special_tokens=False)
            ]

            num_examples += 1
            num_triggers += sum(generation["trigger"] for generation in generations)  # type: ignore

            output = {"generations": generations}
            if "conversation" in row:
                output["conversation"] = row["conversation"]
            if "prompt" in row:
                output["prompt"] = row["prompt"]
                output["label"] = row["label"]

            f.write(json.dumps(output) + "\n")
            f.flush()

    return num_triggers / (num_examples * args.num_samples) if num_examples > 0 else 0


if __name__ == "__main__":
    args = parse(Args, add_config_path_arg=True)
    rows = []

    # interpolate between 'before' (-1), 'after' (0), and amplified (>0) logits
    for alpha in tqdm(np.linspace(args.min_alpha, args.max_alpha, args.num_alphas), total=args.num_alphas):
        triggers = trigger_reconstruction(args, float(alpha))
        rows.append({"alpha": float(alpha), "triggers": float(triggers)})

    output_file = "trigger_reconstruction_{now}.csv"
    output_file = output_file.replace("{now}", datetime.datetime.now().isoformat())
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, output_file), index=False)
