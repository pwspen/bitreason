from __future__ import annotations

import time
from typing import Callable, Sequence

from minireason.sweep import AggregateMetrics, RunConfig, SweepResult, TaskMetrics
from minireason.tasks import TaskCollection

from minireason.solvers import EvalFn, Register, Solver, TrainingArtifacts


def _build_metrics(
    task_name: str,
    artifacts: TrainingArtifacts,
    eval_fn: EvalFn,
    solve_fn: Callable[[Register], Register],
) -> TaskMetrics:
    pixel_acc, pair_acc = eval_fn(solve_fn)
    final_loss = artifacts.loss_curve[-1] if artifacts.loss_curve else float("nan")
    return TaskMetrics(
        task_name=task_name,
        final_loss=final_loss,
        pixel_accuracy=float(pixel_acc),
        pair_accuracy=float(pair_acc),
        loss_curve=artifacts.loss_curve,
        pixel_accuracy_curve=artifacts.pixel_accuracy_curve,
        pair_accuracy_curve=artifacts.pair_accuracy_curve,
    )


def evaluate_solver(
    solver: Solver,
    *,
    config: RunConfig,
    task_funcs: Sequence[Callable[[Register], Sequence[int]]],
    train_samples: int,
    test_samples: int,
) -> SweepResult:
    started = time.time()
    collection = TaskCollection(
        list(task_funcs), train_samples=train_samples, test_samples=test_samples
    )
    task_metrics: list[TaskMetrics] = []

    for pairs, eval_fn, task_name in collection.tasks():
        reset_fn = getattr(solver, "reset", None)
        if callable(reset_fn):
            reset_fn()
        artifacts = solver.train(pairs, eval_fn, task_name=task_name)
        metrics = _build_metrics(task_name, artifacts, eval_fn, solver.solve)
        task_metrics.append(metrics)

    aggregate = AggregateMetrics.from_tasks(task_metrics)
    finished = time.time()
    return SweepResult(
        config=config,
        tasks=task_metrics,
        aggregate=aggregate,
        started_at=started,
        finished_at=finished,
    )
