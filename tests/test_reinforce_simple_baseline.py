import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from algorithms.reinforce_simple_baseline import reinforce_loss, train_step, update_baseline
from datasets.dataset import RLExample
from reward import format_arithmetic_prompt
from rollout import RolloutBatch


class SimpleBaselineTests(unittest.TestCase):
    def test_loss_gradient_direction(self):
        log_probs = torch.tensor([-1.0, -2.0], requires_grad=True)
        rewards = torch.tensor([1.0, 0.0], requires_grad=True)
        reinforce_loss(log_probs, rewards, baseline=0.4).backward()
        torch.testing.assert_close(log_probs.grad, torch.tensor([-0.3, 0.2]))
        self.assertIsNone(rewards.grad)

    def test_baseline_update(self):
        self.assertAlmostEqual(update_baseline(0.2, torch.tensor([1, 0]), decay=0.5), 0.35)

    def test_one_update_uses_incoming_baseline(self):
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.zeros(2))

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.scores.expand(*input_ids.shape, 2))

        model = TinyPolicy()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        examples = [RLExample("a", "first", 7), RLExample("b", "second", 9)]
        batch = RolloutBatch(
            sequences=torch.tensor([[0, 1], [0, 1]]),
            prompt_width=1,
            attention_mask=torch.ones(2, 2, dtype=torch.long),
            response_mask=torch.ones(2, 1, dtype=torch.bool),
            response_texts=["FINAL: 7", "FINAL: 9"],
            ended=torch.tensor([True, False]),
        )
        with patch("algorithms.reinforce_simple_baseline.rollout_batch", return_value=batch) as sample:
            metrics, baseline = train_step(
                model, None, optimizer, examples, 0.0, baseline_decay=0.0,
                measure_policy_shift=True,
            )
        sample.assert_called_once_with(
            model, None, [format_arithmetic_prompt(prompt) for prompt in ("first", "second")],
            max_new_tokens=192,
        )
        torch.testing.assert_close(model.scores.detach(), torch.tensor([-0.25, 0.25]))
        self.assertEqual(metrics["mean_reward"], 0.5)
        self.assertGreater(metrics["grad_norm"], 0.0)
        self.assertGreater(metrics["seq_abs_log_ratio"], 0.0)
        self.assertEqual(baseline, 0.5)
