"""Sample one training prompt without scoring or updating the model."""
import torch

from datasets.dataset import read_examples
from models.policy import load_policy, response_log_prob
from rollout import rollout_one


def main():
    model, tokenizer = load_policy()
    example = read_examples("data/processed/arithmetic-v1/train.jsonl")[0]

    torch.manual_seed(123)
    response_ids, response_text, sequences, prompt_width = rollout_one(
        model, tokenizer, example.prompt
    )
    if response_ids.numel() == 0:
        raise RuntimeError("Rollout produced no response tokens.")

    print("prompt_id:", example.id)
    print("prompt:", example.prompt)
    print("response_ids:", response_ids.tolist())
    print("response_text:", repr(response_text))
    log_prob = response_log_prob(model, sequences, prompt_width)
    if log_prob.shape != (1,) or not log_prob.requires_grad or not torch.isfinite(log_prob).all():
        raise RuntimeError("Expected one finite sequence log-prob with gradients enabled.")
    print("response_log_prob:", log_prob.item())
    print("requires_grad:", log_prob.requires_grad)


if __name__ == "__main__":
    main()
