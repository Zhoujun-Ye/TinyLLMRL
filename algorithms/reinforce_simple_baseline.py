"""REINFORCE with a Simple Baseline."""

from collections.abc import Sequence
from math import log

import torch
from torch.nn.parallel import DistributedDataParallel

from datasets.dataset import RLExample
from models.policy import response_log_prob
from reward import format_arithmetic_prompt, score_answer
from rollout import rollout_batch


def reinforce_loss(
    log_probs: torch.Tensor, rewards: torch.Tensor, baseline: float
) -> torch.Tensor:
    """Return a scalar to minimize from matching [B] log-probs and rewards."""
    if log_probs.ndim != 1 or log_probs.shape != rewards.shape or log_probs.numel() == 0:
        raise ValueError("Expected matching, nonempty [B] log-probs and rewards.")
    advantages = (rewards - baseline).detach()
    return -(advantages * log_probs).mean()


def update_baseline(
    previous: float, rewards: torch.Tensor, *, decay: float = 0.9
) -> float:
    """Compute the running reward baseline to use on the *next* batch."""
    if not 0.0 <= decay < 1.0 or rewards.numel() == 0:
        raise ValueError("Expected decay in [0, 1) and nonempty rewards.")
    mean_reward = rewards.detach().float().mean().item()
    return decay * previous + (1.0 - decay) * mean_reward


def train_step(
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[RLExample],
    baseline: float,
    *,
    max_new_tokens: int = 192,
    baseline_decay: float = 0.9,
    measure_policy_shift: bool = False,
) -> tuple[dict[str, float], float]:
    """Run one update; return metrics and the next batch's baseline."""
    if any(type(example.ground_truth) is not int for example in examples):
        raise TypeError("Simple Baseline training requires integer ground truth.")
    policy = model.module if isinstance(model, DistributedDataParallel) else model
    batch = rollout_batch(
        policy, tokenizer, [format_arithmetic_prompt(example.prompt) for example in examples],
        max_new_tokens=max_new_tokens,
    )
    rewards = torch.tensor(
        [score_answer(text, example.ground_truth, ended=done)
         for text, example, done in zip(batch.response_texts, examples, batch.ended.tolist(), strict=True)],
        dtype=torch.float32,
        device=batch.sequences.device,
    )
    next_baseline = update_baseline(baseline, rewards, decay=baseline_decay)

    optimizer.zero_grad(set_to_none=True)
    log_probs = response_log_prob(
        model, batch.sequences, batch.prompt_width,
        attention_mask=batch.attention_mask,
        response_mask=batch.response_mask,
    )
    # This batch's rewards only affect the baseline for the next update.
    loss = reinforce_loss(log_probs, rewards, baseline)
    loss.backward()
    grad_norm = torch.nn.utils.get_total_norm(
        [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
        error_if_nonfinite=True,
    )
    optimizer.step()

    lengths = batch.response_mask.sum(dim=1)
    metrics = {
        "loss": loss.detach().item(),
        "mean_reward": rewards.mean().item(),
        "baseline": baseline,
        "advantage_std": (rewards - baseline).std(unbiased=False).item(),
        "grad_norm": grad_norm.item(),
        "mean_response_tokens": lengths.float().mean().item(),
        "cap_rate": ((lengths == max_new_tokens) & ~batch.ended).float().mean().item(),
    }
    if measure_policy_shift:
        with torch.no_grad():
            after = response_log_prob(
                policy, batch.sequences, batch.prompt_width,
                attention_mask=batch.attention_mask, response_mask=batch.response_mask,
            )
            log_ratio = after.double() - log_probs.detach().double()
            metrics["seq_abs_log_ratio"] = log_ratio.abs().mean().item()
            metrics["seq_ratio_outside_0p2"] = (
                (log_ratio < log(0.8)) | (log_ratio > log(1.2))
            ).float().mean().item()
    return metrics, next_baseline
