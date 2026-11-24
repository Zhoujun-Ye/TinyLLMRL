from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutBatch:
    sequences: torch.Tensor
    prompt_width: int
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    response_texts: list[str]
    ended: torch.Tensor


def _response_mask(response_ids, eos_token_ids):
    is_eos = torch.zeros_like(response_ids, dtype=torch.bool)
    for token_id in eos_token_ids:
        is_eos |= response_ids == token_id
    # Keep the first EOS itself, but exclude every position after it.
    return is_eos.cumsum(dim=1) - is_eos.long() == 0


def rollout_batch(model, tokenizer, prompts: list[str], *, max_new_tokens: int = 192):
    """Sample one response per prompt, in the original order."""
    if not prompts:
        raise ValueError("rollout_batch requires at least one prompt.")
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        tokenizer_kwargs={"padding_side": "left"},
    ).to(model.device)

    model.eval()
    prompt_width = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )

    # Autograd may need to save these IDs during the scoring forward pass.
    sequences = sequences.clone()
    response_ids = sequences[:, prompt_width:]
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else (eos or [])
    response_mask = _response_mask(response_ids, eos_ids)
    attention_mask = torch.cat((inputs["attention_mask"], response_mask.long()), dim=1)
    eos_set = set(eos_ids)
    lengths = response_mask.sum(dim=1).tolist()
    token_lists = [row[:length] for row, length in zip(response_ids.tolist(), lengths, strict=True)]
    finished = [bool(ids and ids[-1] in eos_set) for ids in token_lists]
    # Strip only the terminal EOS; do not silently remove control tokens inside an answer.
    texts = tokenizer.batch_decode(
        [ids[:-1] if done else ids for ids, done in zip(token_lists, finished, strict=True)],
        skip_special_tokens=False,
    )
    ended = torch.tensor(finished, device=sequences.device, dtype=torch.bool)
    return RolloutBatch(sequences, prompt_width, attention_mask, response_mask, texts, ended)


def rollout_one(model, tokenizer, prompt: str, *, max_new_tokens: int = 192):
    batch = rollout_batch(model, tokenizer, [prompt], max_new_tokens=max_new_tokens)
    response_ids = batch.sequences[0, batch.prompt_width:]
    return response_ids, batch.response_texts[0], batch.sequences, batch.prompt_width
