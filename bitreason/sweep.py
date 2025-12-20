import json
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    values: Iterable[Any]


@dataclass(frozen=True)
class RunConfig:
    parameters: dict[str, Any]
    label: str | None = None

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.parameters.get(key, default)


def expand_grid(specs: Sequence[ParameterSpec]) -> list[RunConfig]:
    names = [spec.name for spec in specs]
    value_lists = [list(spec.values) for spec in specs]
    configs: list[RunConfig] = []

    for combo in product(*value_lists):
        params = dict(zip(names, combo))
        configs.append(RunConfig(parameters=params))
    return configs


@dataclass(frozen=True)
class TaskMetrics:
    task_name: str
    final_loss: float
    pixel_accuracy: float
    pair_accuracy: float
    loss_curve: list[float]
    pixel_accuracy_curve: list[tuple[int, float]]
    pair_accuracy_curve: list[tuple[int, float]]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["pixel_accuracy_curve"] = [
            [epoch, acc] for epoch, acc in self.pixel_accuracy_curve
        ]
        data["pair_accuracy_curve"] = [
            [epoch, acc] for epoch, acc in self.pair_accuracy_curve
        ]
        return data


@dataclass(frozen=True)
class AggregateMetrics:
    mean_final_loss: float
    mean_pixel_accuracy: float
    mean_pair_accuracy: float

    @staticmethod
    def from_tasks(tasks: Sequence[TaskMetrics]) -> "AggregateMetrics":
        n = len(tasks)
        if n == 0:
            return AggregateMetrics(
                mean_final_loss=float("nan"),
                mean_pixel_accuracy=float("nan"),
                mean_pair_accuracy=float("nan"),
            )

        mean_loss = sum(t.final_loss for t in tasks) / n
        mean_pixel = sum(t.pixel_accuracy for t in tasks) / n
        mean_pair = sum(t.pair_accuracy for t in tasks) / n
        return AggregateMetrics(
            mean_final_loss=mean_loss,
            mean_pixel_accuracy=mean_pixel,
            mean_pair_accuracy=mean_pair,
        )

    def to_json(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SweepResult:
    config: RunConfig
    tasks: list[TaskMetrics]
    aggregate: AggregateMetrics
    started_at: float
    finished_at: float
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "config": {
                "parameters": self.config.parameters,
                "label": self.config.label,
            },
            "tasks": [t.to_json() for t in self.tasks],
            "aggregate": self.aggregate.to_json(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class SweepLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: SweepResult) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_json()))
            f.write("\n")


def load_results(path: Path) -> list[SweepResult]:
    results: list[SweepResult] = []
    if not path.exists():
        return results

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            config = RunConfig(
                parameters=record["config"]["parameters"],
                label=record["config"].get("label"),
            )
            tasks = [
                TaskMetrics(
                    task_name=task_rec["task_name"],
                    final_loss=task_rec["final_loss"],
                    pixel_accuracy=task_rec["pixel_accuracy"],
                    pair_accuracy=task_rec["pair_accuracy"],
                    loss_curve=task_rec["loss_curve"],
                    pixel_accuracy_curve=[
                        (int(epoch), float(acc))
                        for epoch, acc in task_rec["pixel_accuracy_curve"]
                    ],
                    pair_accuracy_curve=[
                        (int(epoch), float(acc))
                        for epoch, acc in task_rec["pair_accuracy_curve"]
                    ],
                )
                for task_rec in record["tasks"]
            ]
            aggregate = AggregateMetrics(
                mean_final_loss=record["aggregate"]["mean_final_loss"],
                mean_pixel_accuracy=record["aggregate"]["mean_pixel_accuracy"],
                mean_pair_accuracy=record["aggregate"]["mean_pair_accuracy"],
            )
            results.append(
                SweepResult(
                    config=config,
                    tasks=tasks,
                    aggregate=aggregate,
                    started_at=record["started_at"],
                    finished_at=record["finished_at"],
                    error=record.get("error"),
                )
            )
    return results
