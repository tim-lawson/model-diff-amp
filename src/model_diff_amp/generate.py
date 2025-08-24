# Based on https://www.goodfire.ai/papers/model-diff-amplification

import torch
import torch.nn.functional as F
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(
    input_ids: Tensor,
    model_before: torch.nn.Module,
    model_after: torch.nn.Module,
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
