"""Train Simple Baseline REINFORCE and log diagnostic metrics on the server."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from algorithms.reinforce_simple_baseline import train_step
from datasets.dataset import read_examples
from models.policy import DEFAULT_MODEL_DIR, load_policy
from reward import REWARD_VERSION, format_arithmetic_prompt, score_answer
from rollout import rollout_batch


@torch.no_grad()
def evaluate(model, tokenizer, examples, *, batch_size, max_new_tokens, seed):
    # Evaluation must not advance the RNG stream used for training rollouts.
    devices = [model.device] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(model.device):
                torch.cuda.manual_seed(seed)
        correct = 0
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            batch = rollout_batch(
                model, tokenizer, [format_arithmetic_prompt(row.prompt) for row in chunk],
                max_new_tokens=max_new_tokens,
            )
            correct += sum(
                score_answer(text, row.ground_truth, ended=done)
                for text, row, done in zip(batch.response_texts, chunk, batch.ended.tolist(), strict=True)
            )
        return correct / len(examples)


def run(args, rank, world_size, local_rank):
    primary = rank == 0
    swanlab = None
    if primary and args.swanlab_mode != "disabled":
        import swanlab

    if min(args.steps, args.batch_size, args.eval_every, args.eval_size,
           args.shift_every, args.max_new_tokens) < 1 or args.lr <= 0:
        raise ValueError("Counts, intervals and learning rate must be positive.")
    if args.batch_size % world_size:
        raise ValueError("Global batch size must be divisible by the number of ranks.")
    data = Path("data/processed/arithmetic-v1")
    train_path, val_path = data / "train.jsonl", data / "validation.jsonl"
    train_rows, val_rows = read_examples(train_path), read_examples(val_path)
    if len(train_rows) < args.batch_size or len(val_rows) < args.eval_size:
        raise ValueError("Batch size or evaluation size exceeds its data split.")
    validation = random.Random(args.eval_seed).sample(val_rows, args.eval_size)
    sampler = random.Random(args.seed)
    torch.manual_seed(args.seed + rank)
    model, tokenizer = load_policy(device=f"cuda:{local_rank}")
    actor = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world_size > 1 else model
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=0.0)
    output = Path("outputs") / args.run_name
    config = {
        **vars(args), "baseline_decay": 0.9, "model_dir": str(DEFAULT_MODEL_DIR),
        "attention": model.config._attn_implementation, "dtype": str(model.dtype),
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(model.device),
        "world_size": world_size, "per_rank_batch_size": args.batch_size // world_size,
        "reward_version": REWARD_VERSION,
        "validation_ids": [row.id for row in validation],
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, val_path)
        },
        "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "enable_thinking": False},
    }
    stream = None
    try:
        if primary:
            output.mkdir(parents=True, exist_ok=False)
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            stream = (output / "metrics.jsonl").open("w", encoding="utf-8")
            if swanlab is not None:
                swanlab.init(
                    project="TinyLLMRL", experiment_name=args.run_name, config=config,
                    mode=args.swanlab_mode, logdir=str(output / "swanlog"),
                    on_online_error="raise" if args.swanlab_mode == "online" else None,
                )

        def report(step, metrics):
            if stream is not None:
                stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
                stream.flush()
                print(f"step {step}: {metrics}", flush=True)
                if swanlab is not None:
                    swanlab.log(metrics, step=step)

        def validate():
            return evaluate(
                model, tokenizer, validation, batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens, seed=args.eval_seed,
            )

        if primary:
            report(0, {"validation/accuracy": validate()})
        baseline = 0.0
        per_rank = args.batch_size // world_size
        for step in range(1, args.steps + 1):
            examples = sampler.sample(train_rows, args.batch_size)
            if world_size > 1:
                dist.barrier()
            started = perf_counter()
            metrics, next_baseline = train_step(
                actor, tokenizer, optimizer, examples[rank * per_rank:(rank + 1) * per_rank],
                baseline, max_new_tokens=args.max_new_tokens,
                measure_policy_shift=(step == 1 or step % args.shift_every == 0),
            )
            if world_size > 1:
                values = torch.tensor(
                    [*metrics.values(), next_baseline], device=model.device, dtype=torch.float64,
                )
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                *means, next_baseline = (values / world_size).tolist()
                metrics = dict(zip(metrics, means))
                p = metrics["mean_reward"]
                metrics["advantage_std"] = (p * (1.0 - p)) ** 0.5
            baseline = next_baseline
            elapsed = perf_counter() - started
            if primary:
                scalars = {f"train/{key}": value for key, value in metrics.items()}
                scalars["perf/step_seconds"] = elapsed
                scalars["perf/samples_per_second"] = args.batch_size / elapsed
                scalars["perf/generated_tokens_per_second"] = args.batch_size * metrics["mean_response_tokens"] / elapsed
                if step % args.eval_every == 0 or step == args.steps:
                    scalars["validation/accuracy"] = validate()
                report(step, scalars)
        if primary and not args.no_save:
            model.save_pretrained(output / "policy")
            tokenizer.save_pretrained(output / "policy")
    finally:
        if stream is not None:
            stream.close()
        if swanlab is not None:
            swanlab.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--shift-every", type=int, default=5)
    parser.add_argument("--swanlab-mode", choices=("local", "online", "disabled"), default="online")
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--no-save", action="store_true", help="Skip saving weights for short diagnostics.")
    args = parser.parse_args()
    if args.swanlab_mode == "online" and not os.environ.get("SWANLAB_API_KEY"):
        from dotenv import dotenv_values

        key = dotenv_values(Path(__file__).with_name(".env"), interpolate=False).get("SWANLAB_API_KEY")
        if not key:
            parser.error("Set SWANLAB_API_KEY in the server's project .env or environment.")
        os.environ["SWANLAB_API_KEY"] = key
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(args.dist_backend)
    try:
        run(args, rank, world_size, local_rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
