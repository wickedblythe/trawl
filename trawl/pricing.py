"""Model pricing table and cost estimation for Claude models."""
from __future__ import annotations

# pricing per million tokens (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-5-20250514": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-haiku-3-5-20241022": (0.80, 4.0),
}

_FALLBACK_TIERS = [
    ("opus", (15.0, 75.0)),
    ("sonnet", (3.0, 15.0)),
    ("haiku", (0.80, 4.0)),
]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD for the given model and token counts."""
    if model in MODEL_PRICING:
        in_rate, out_rate = MODEL_PRICING[model]
    else:
        in_rate, out_rate = 0.0, 0.0
        for keyword, rates in _FALLBACK_TIERS:
            if keyword in model:
                in_rate, out_rate = rates
                break
        else:
            return 0.0

    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
