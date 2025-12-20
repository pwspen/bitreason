from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from minireason.sweep import SweepResult


def _extract_series(
    results: Iterable[SweepResult],
    param_name: str,
) -> dict[str, list[tuple[float, float]]]:
    """Aggregate view: metric -> [(param, value)]."""
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


def _extract_series_per_task(
    results: Iterable[SweepResult],
    param_name: str,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Per-task view: metric -> task_name -> [(param, value)]."""
    series: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        if result.error:
            continue
        params = result.config.parameters
        if param_name not in params:
            continue
        x_value = params[param_name]
        for task in result.tasks:
            series["final_loss"][task.task_name].append((x_value, task.final_loss))
            series["pixel_accuracy"][task.task_name].append(
                (x_value, task.pixel_accuracy)
            )
            series["pair_accuracy"][task.task_name].append(
                (x_value, task.pair_accuracy)
            )
    return series


def plot_param_curves(
    results: Iterable[SweepResult],
    param_name: str,
    output_path: Path,
    title: str | None = None,
    *,
    per_task: bool = False,
) -> None:
    if per_task:
        series = _extract_series_per_task(results, param_name)
        metrics = [
            ("final_loss", "Final Loss"),
            ("pixel_accuracy", "Pixel Accuracy"),
            ("pair_accuracy", "Pair Accuracy"),
        ]
    else:
        series = _extract_series(results, param_name)
        metrics = [
            ("mean_final_loss", "Mean Final Loss"),
            ("mean_pixel_accuracy", "Mean Pixel Accuracy"),
            ("mean_pair_accuracy", "Mean Pair Accuracy"),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, label) in zip(axes, metrics):
        if per_task:
            task_series = series.get(key, {})
            if not task_series:
                ax.set_visible(False)
                continue
            for task_name, points in sorted(task_series.items()):
                points_sorted = sorted(points, key=lambda tup: tup[0])
                xs, ys = zip(*points_sorted)
                ax.plot(xs, ys, marker="o", linewidth=2, label=task_name)
            ax.legend()
        else:
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
