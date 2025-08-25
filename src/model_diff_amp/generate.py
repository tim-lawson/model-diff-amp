# Based on https://www.goodfire.ai/papers/model-diff-amplification

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import Tensor
from transformers import PreTrainedModel

# TODO: review CPU/GPU comms


def generate(
    input_ids: Tensor,
    model_before: PeftModel | PreTrainedModel | None,
    model_after: PeftModel | PreTrainedModel,
    num_samples: int,
    max_new_tokens: int,
    alpha: float,
    temperature: float,
    bf16: bool = True,
):
    assert torch.cuda.is_available()

    model_before = model_before or model_after  # handle peft model case

    if temperature == 0:
        num_samples = 1  # greedy

    num_input_tokens = input_ids.shape[1]
    input_ids = input_ids.repeat(num_samples, 1)  # num_samples num_input_tokens

    past_key_values_before = None
    past_key_values_after = None

    for _ in range(max_new_tokens):
        input_ids_before = input_ids[:, -1:] if past_key_values_before else input_ids
        input_ids_after = input_ids[:, -1:] if past_key_values_after else input_ids

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, bf16):
            with model_after.disable_adapter() if isinstance(model_after, PeftModel) else nullcontext():
                outputs_before = model_before(
                    input_ids=input_ids_before.to(model_before.device),
                    past_key_values=past_key_values_before,
                    use_cache=True,
                )
            outputs_after = model_after(
                input_ids=input_ids_after.to(model_after.device),  # type: ignore
                past_key_values=past_key_values_after,
                use_cache=True,
            )

        past_key_values_before = outputs_before.past_key_values
        past_key_values_after = outputs_after.past_key_values

        logits_before = outputs_before.logits[:, -1, :]
        logits_after = outputs_after.logits[:, -1, :].to(logits_before.device)

        logits_amplified = logits_after + alpha * (logits_after - logits_before)

        if temperature == 0:
            next_token = torch.argmax(logits_amplified, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits_amplified / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token.cpu()], dim=-1)

    return input_ids[:, num_input_tokens:]  # num_samples max_new_tokens
