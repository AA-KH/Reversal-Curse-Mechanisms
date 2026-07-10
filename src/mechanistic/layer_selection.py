"""Layer-selection helpers shared by mechanistic analyses."""

from __future__ import annotations

from collections.abc import Sequence


PAPER_MLP_ABLATION_START = 6
PAPER_MLP_ABLATION_STOP = 19


def paper_mlp_ablation_layers(n_layers: int) -> list[int]:
    """Return the paper's MLP ablation window clipped to model depth.

    GPT-2 Medium and Large both retain the original layers 6-18. Smaller
    profiles, such as GPT-2 Small, use the valid subset of that same window.
    """

    if n_layers <= PAPER_MLP_ABLATION_START:
        raise ValueError(
            f"Model has {n_layers} layers, fewer than the paper MLP ablation "
            f"start layer {PAPER_MLP_ABLATION_START}."
        )
    return list(range(PAPER_MLP_ABLATION_START, min(PAPER_MLP_ABLATION_STOP, n_layers)))


def format_layer_span(layers: Sequence[int]) -> str:
    """Format a contiguous layer list for logs and CSV metadata."""

    if not layers:
        return "none"
    if len(layers) == 1:
        return str(layers[0])
    return f"{layers[0]}-{layers[-1]}"
