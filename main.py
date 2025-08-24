import json

import torch
import torch.nn.functional as F
from datasets import IterableDataset, load_dataset
from pydantic_settings import BaseSettings
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(
    input_ids: Tensor,
    model_before: PreTrainedModel,
    model_after: PreTrainedModel,
    num_samples: int,
    max_new_tokens: int,
    alpha: float,
    temperature: float,
    bf16: bool = True,
):
    num_input_tokens = input_ids.shape[1]
    input_ids = input_ids.repeat(num_samples, 1).to(device)  # num_samples num_input_tokens

    past_key_values_before = None
    past_key_values_after = None

    for _ in range(max_new_tokens):
        input_ids_before = input_ids[:, -1:] if past_key_values_before else input_ids
        input_ids_after = input_ids[:, -1:] if past_key_values_after else input_ids

        with torch.no_grad(), torch.autocast(str(device), torch.bfloat16, bf16):
            outputs_before = model_before(
                input_ids=input_ids_before,
                past_key_values=past_key_values_before,
                use_cache=True,
            )
            outputs_after = model_after(
                input_ids=input_ids_after,
                past_key_values=past_key_values_after,
                use_cache=True,
            )

        past_key_values_before = outputs_before.past_key_values
        past_key_values_after = outputs_after.past_key_values

        logits_before = outputs_before.logits[:, -1, :]
        logits_after = outputs_after.logits[:, -1, :]

        logits_amplified = logits_after + alpha * (logits_after - logits_before)

        probs = F.softmax(logits_amplified / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

    return input_ids[:, num_input_tokens:]  # num_samples max_new_tokens


class Args(BaseSettings):
    output_file: str = "output.jsonl"
    seed: int = 0
    model_name_before: str = "meta-llama/Llama-3.2-1B"
    model_name_after: str = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_name: str = "lmsys/lmsys-chat-1m"
    split: str = "train"
    num_examples: int = 32
    num_samples: int = 10
    max_new_tokens: int = 32
    alpha: float = 0.3
    temperature: float = 1.0
    bf16: bool = True


def main(args: Args) -> None:
    set_seed(args.seed)

    # assume 'after' is instruct model with chat template
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_name_after)

    model_before = AutoModelForCausalLM.from_pretrained(args.model_name_before)
    model_before = model_before.to(device)  # type: ignore
    model_before.eval()

    model_after = AutoModelForCausalLM.from_pretrained(args.model_name_after)
    model_after = model_after.to(device)  # type: ignore
    model_after.eval()

    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    assert isinstance(dataset, IterableDataset)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in tqdm(dataset.take(args.num_examples), total=args.num_examples):
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


if __name__ == "__main__":
    args = Args(_cli_parse_args=True)  # type: ignore
    main(args)
