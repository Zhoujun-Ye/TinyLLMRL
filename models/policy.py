"""Load the server-local Qwen policy and tokenizer without downloading files."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "weights" / "Qwen3-0.6B"


def load_policy(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Return (model, tokenizer); callers control sampling and training mode."""
    path = Path(model_dir).expanduser().resolve()
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"No model.safetensors in {path}; see README.md.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current PyTorch environment.")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    # On the training GPU, SDPA gave inconsistent padded and unpadded scores.
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, use_safetensors=True, dtype=dtype,
        attn_implementation="eager",
    )
    return model.to(device), tokenizer

def response_log_prob(model, input_ids, prompt_width, *, attention_mask=None, response_mask=None):
    """Return one log-prob per response; pass both masks for padded batches."""
    if (attention_mask is None) != (response_mask is None):
        raise ValueError("Pass both masks for a padded batch, or neither for unpadded inputs.")
    kwargs = {}
    if attention_mask is not None:
        # Left padding should not advance the positions of real tokens.
        position_ids = (attention_mask.cumsum(dim=1) - 1).masked_fill(attention_mask == 0, 1)
        kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
    logits = model(input_ids, use_cache=False, **kwargs).logits
    response_logits = logits[:, prompt_width - 1 : -1, :]
    response_log_probs = torch.log_softmax(response_logits.float(), dim=-1)
    response_ids = input_ids[:, prompt_width:]

    token_log_probs = response_log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
    if response_mask is not None:
        token_log_probs = token_log_probs.masked_fill(~response_mask, 0.0)
    return token_log_probs.sum(dim=-1)  # [B]
