"""Generate a small, reproducible English arithmetic dataset."""
import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

from datasets.dataset import RLExample, write_examples


TRAIN_TEMPLATES = (
    ("add_join", "Maya had {a} beads and got {b} more. How many beads does she have now?", "add"),
    ("add_parts", "A shelf has {b} blue mugs and {a} red mugs. There are {c} spoons nearby. "
     "How many mugs are on the shelf?", "add"),
    ("subtract_remove", "Leo had {a} stickers and gave away {b}. How many stickers does he have left?", "subtract"),
    ("subtract_missing_part", "A box holds {a} pencils. {b} are green and the rest are yellow. "
     "There are {c} erasers next to the box. How many pencils are yellow?", "subtract"),
)
HELDOUT_TEMPLATES = (
    ("add_recover_start", "After donating {b} books, Ava still has {a}. "
     "How many books did she have before donating?", "add"),
    ("subtract_compare", "Noah has {b} postcards. Emma has {a} postcards and {c} stamps. "
     "How many more postcards does Emma have than Noah?", "subtract"),
)
SPLITS = (
    ("train", 800, TRAIN_TEMPLATES),
    ("validation", 160, TRAIN_TEMPLATES),
    ("template_test", 160, HELDOUT_TEMPLATES),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output: Path, seed: int) -> None:
    rng = random.Random(seed)
    # Allocate operand pairs before rendering prompts, so none cross a split.
    pairs = [(a, b) for a in range(13, 100) for b in range(12, a)]
    rng.shuffle(pairs)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "seed": seed,
        "python": platform.python_version(),
        "generator_sha256": sha256(Path(__file__)),
        "jsonl_writer_sha256": sha256(Path(__file__).with_name("dataset.py")),
        "splits": {},
    }
    offset = 0
    for split, size, templates in SPLITS:
        examples = []
        for index, (a, b) in enumerate(pairs[offset:offset + size]):
            name, template, operation = templates[index % len(templates)]
            c = rng.choice([n for n in range(2, 20) if n not in (a, b)])
            answer = a + b if operation == "add" else a - b
            examples.append(RLExample(
                id=f"arith-{a}-{b}",
                prompt=template.format(a=a, b=b, c=c) + " Give only the final number.",
                ground_truth=answer,
                metadata={"template": name, "operation": operation, "operands": [a, b]},
            ))
        offset += size
        rng.shuffle(examples)
        path = output / f"{split}.jsonl"
        write_examples(path, examples)
        manifest["splits"][split] = {"count": len(examples), "sha256": sha256(path)}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output, args.seed)
