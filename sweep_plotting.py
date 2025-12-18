from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from sweep import SweepResult


def _extract_series(
    results: Iterable[SweepResult],
    param_name: str,
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for result in results:
        if result.error:
            continue
        params = result.config.parameters
        if param_name not in params:
            continue

        x_value = params[param_name]
        aggregate = result.aggregate
        series["mean_final_loss"].append((x_value, aggregate.mean_final_loss))
        series["mean_pixel_accuracy"].append((x_value, aggregate.mean_pixel_accuracy))
        series["mean_pair_accuracy"].append((x_value, aggregate.mean_pair_accuracy))
    return series


def plot_param_curves(
    results: Iterable[SweepResult],
    param_name: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    series = _extract_series(results, param_name)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("mean_final_loss", "Mean Final Loss"),
        ("mean_pixel_accuracy", "Mean Pixel Accuracy"),
        ("mean_pair_accuracy", "Mean Pair Accuracy"),
    ]

    for ax, (key, label) in zip(axes, metrics):
        points = sorted(series.get(key, []), key=lambda tup: tup[0])
        if not points:
            ax.set_visible(False)
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_title(label)
        ax.set_xlabel(param_name)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.7)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
