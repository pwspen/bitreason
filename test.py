from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sweep import (
    AggregateMetrics,
    ParameterSpec,
    RunConfig,
    SweepLogger,
    SweepResult,
    TaskMetrics,
    expand_grid,
    load_results,
)
from sweep_plotting import plot_param_curves
from task_list import task_list
from tasks import Pair, TaskCollection


# Simple neural network
class BitNet(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        hsize = 64
        hlayers = 2
        self.net = nn.Sequential(
            nn.Linear(size, hsize),
            nn.ReLU(),
            *[
                layer
                for _ in range(hlayers)
                for layer in (nn.Linear(hsize, hsize), nn.ReLU(), nn.Dropout(0.01))
            ],
            nn.Linear(hsize, size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class TrainingRun:
    net: BitNet
    losses: list[float]
    pixel_accuracies: list[tuple[int, float]]
    pair_accuracies: list[tuple[int, float]]


def train_net(
    train_pairs: tuple[Pair, ...],
    eval_fn: Callable[[Callable[[tuple[int, ...]], tuple[int, ...]]], tuple[float, float]],
    task_name: str,
    *,
    epochs: int,
    test_every: int,
    lr: float,
) -> TrainingRun:
    net = BitNet(size=16)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses: list[float] = []
    pixel_accuracies: list[tuple[int, float]] = []
    pair_accuracies: list[tuple[int, float]] = []

    pbar = tqdm(range(epochs), desc=f"Training {task_name}", leave=False)
    num_pairs = max(len(train_pairs), 1)

    for epoch in pbar:
        batch_loss = 0.0
        for pair in train_pairs:
            inp, out = pair.input, pair.output
            inp_t = torch.tensor(inp, dtype=torch.float32)
            out_t = torch.tensor(out, dtype=torch.float32)

            pred = net(inp_t)
            loss = criterion(pred, out_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += float(loss.item())

        avg_loss = batch_loss / num_pairs
        losses.append(avg_loss)
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

        if epoch % test_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                solve_fn = lambda inp: tuple(
                    int(x)
                    for x in net(torch.tensor(inp, dtype=torch.float32))
                    .round()
                    .int()
                    .tolist()
                )
                pixel_acc, pair_acc = eval_fn(solve_fn)

            pixel_accuracies.append((epoch, float(pixel_acc)))
            pair_accuracies.append((epoch, float(pair_acc)))

    return TrainingRun(
        net=net,
        losses=losses,
        pixel_accuracies=pixel_accuracies,
        pair_accuracies=pair_accuracies,
    )


def run_config(
    config: RunConfig,
    *,
    task_funcs: Sequence[Callable[[tuple[int, ...]], Sequence[int]]],
    test_samples: int,
    epochs: int,
    test_every: int,
    lr: float,
) -> SweepResult:
    train_samples = int(config.get("train_samples"))
    started = time.time()

    collection = TaskCollection(
        task_funcs,
        train_samples=train_samples,
        test_samples=test_samples,
    )

    task_metrics: list[TaskMetrics] = []
    task_iter = collection.tasks()

    for pairs, eval_fn, task_name in tqdm(
        task_iter,
        desc="Tasks",
        total=len(collection),
        leave=False,
    ):
        training = train_net(
            train_pairs=pairs,
            eval_fn=eval_fn,
            task_name=task_name,
            epochs=epochs,
            test_every=test_every,
            lr=lr,
        )

        with torch.no_grad():
            solve_fn = lambda inp: tuple(
                int(x)
                for x in training.net(torch.tensor(inp, dtype=torch.float32))
                .round()
                .int()
                .tolist()
            )
            pixel_acc, pair_acc = eval_fn(solve_fn)

        final_loss = training.losses[-1] if training.losses else float("nan")
        task_metrics.append(
            TaskMetrics(
                task_name=task_name,
                final_loss=final_loss,
                pixel_accuracy=float(pixel_acc),
                pair_accuracy=float(pair_acc),
                loss_curve=training.losses,
                pixel_accuracy_curve=training.pixel_accuracies,
                pair_accuracy_curve=training.pair_accuracies,
            )
        )

    aggregate = AggregateMetrics.from_tasks(task_metrics)
    finished = time.time()
    return SweepResult(
        config=config,
        tasks=task_metrics,
        aggregate=aggregate,
        started_at=started,
        finished_at=finished,
    )


def main() -> None:
    sweep_specs = [
        ParameterSpec(name="train_samples", values=[10, 25, 50, 100]),
    ]
    configs = expand_grid(sweep_specs)

    log_path = Path("results/train_samples_sweep.jsonl")
    plot_path = Path("results/train_samples_sweep.png")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)

    logger = SweepLogger(log_path)
    sweep_results: list[SweepResult] = []

    for config in tqdm(configs, desc="Sweep (train_samples)", total=len(configs)):
        try:
            result = run_config(
                config,
                task_funcs=task_list,
                test_samples=200,
                epochs=100,
                test_every=10,
                lr=0.01,
            )
        except Exception as exc:  # noqa: BLE001
            now = time.time()
            result = SweepResult(
                config=config,
                tasks=[],
                aggregate=AggregateMetrics.from_tasks([]),
                started_at=now,
                finished_at=now,
                error=str(exc),
            )
        logger.append(result)
        sweep_results.append(result)

    # Reload from disk to ensure plotting matches what was logged
    logged_results = load_results(log_path)
    plot_param_curves(
        logged_results,
        param_name="train_samples",
        output_path=plot_path,
        title="Sweep: Train Samples",
    )
    print(f"Sweep complete. Logged to {log_path} and plotted to {plot_path}.")


if __name__ == "__main__":
    main()
