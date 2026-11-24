"""Versioned exact-answer contract for the synthetic integer arithmetic task."""
import re

REWARD_VERSION = "final-integer-or-checked-equation-eos-v2"
_LEGACY_INSTRUCTION = "Give only the final number."
_INSTRUCTION = (
    "Reply with exactly one line using the format below. Replace <integer> with "
    "your answer. No explanation, units, Markdown, or trailing punctuation.\n"
    "FINAL: <integer>"
)
_MARKER = re.compile(r"\bFINAL[ \t]*:", re.IGNORECASE | re.ASCII)
_INTEGER = r"[+-]?[0-9]{1,32}"
_FINAL_LINE = re.compile(rf"FINAL:[ \t]+({_INTEGER})")
_EQUATION_LINE = re.compile(
    rf"FINAL:[ \t]+({_INTEGER})[ \t]*([+-])[ \t]*({_INTEGER})[ \t]*=[ \t]*({_INTEGER})"
)


def format_arithmetic_prompt(prompt: str) -> str:
    """Attach the answer contract without changing the underlying dataset."""
    question = prompt.rstrip().removesuffix(_LEGACY_INSTRUCTION).rstrip()
    return f"{question}\n\n{_INSTRUCTION}"


def extract_answer(response_text: str, *, ended: bool) -> int | None:
    """Accept a single final integer or a verified one-step equation after EOS."""
    if not ended:
        return None
    text = response_text.rstrip(" \t\r\n")
    if len(_MARKER.findall(text)) != 1:
        return None
    line = text.rsplit("\n", 1)[-1]
    match = _FINAL_LINE.fullmatch(line)
    if match:
        return int(match[1])
    equation = _EQUATION_LINE.fullmatch(line)
    if not equation:
        return None
    left, operator, right, result = equation.groups()
    computed = int(left) + int(right) if operator == "+" else int(left) - int(right)
    return computed if computed == int(result) else None


def score_answer(response_text: str, ground_truth: int, *, ended: bool) -> int:
    """Award 1 only for an unambiguous declaration matching the integer target."""
    if type(ground_truth) is not int:
        raise TypeError("The synthetic arithmetic ground truth must be an integer.")
    return int(extract_answer(response_text, ended=ended) == ground_truth)
