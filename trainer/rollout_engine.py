"""PyTorch-native rollout engines for RL training."""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(
    model, input_ids: Tensor, n_keep: int, attention_mask: Tensor | None = None
) -> Tensor:
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[
        :, :-1, :
    ]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: list[str]
    prompt_lens: Tensor
    completion_mask: Tensor


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):
    def __init__(
        self,
        policy_model: torch.nn.Module,
        tokenizer,
        device: str = "cuda",
        autocast_ctx=None,
    ):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        model = (
            self.policy_model.module
            if isinstance(self.policy_model, DistributedDataParallel)
            else self.policy_model
        )
        ctx = (
            self.autocast_ctx()
            if callable(self.autocast_ctx)
            else (self.autocast_ctx or nullcontext())
        )
        with torch.no_grad(), ctx:
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            full_mask = torch.cat(
                [
                    attention_mask.repeat_interleave(num_generations, dim=0),
                    attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)),
                ],
                dim=1,
            )
            per_token_logps = compute_per_token_logps(
                self.policy_model,
                output_ids,
                completion_ids.size(1),
                attention_mask=full_mask,
            )
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(
            output_ids,
            completion_ids,
            per_token_logps,
            completions,
            prompt_ids.new_full((output_ids.size(0),), prompt_len),
            attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)),
        )

    def update_policy(self, model: torch.nn.Module):
        self.policy_model = model


# ===== 工厂函数 =====
def create_rollout_engine(
    policy_model: torch.nn.Module,
    tokenizer,
    device: str = "cuda",
    autocast_ctx=None,
) -> RolloutEngine:
    return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
