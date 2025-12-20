from pathlib import Path

from minireason.evaluation import evaluate_solver
from minireason.solvers import PySRSolver
from minireason.sweep import RunConfig, SweepLogger, SweepResult
from minireason.sweep_plotting import plot_param_curves
from minireason.task_list import task_list


def main() -> None:
    results: list[SweepResult] = []
    train_sample_grid = [2, 5, 10, 20, 50, 100, 200, 500]
    solver = PySRSolver(niterations=50)
    log_dir = Path("results/examples")
    log_dir.mkdir(parents=True, exist_ok=True)

    for train_samples in train_sample_grid:
        config = RunConfig(
            parameters={"solver": solver.name, "train_samples": train_samples}
        )
        log_path = log_dir / f"{solver.name}_train{train_samples}.jsonl"
        log_path.unlink(missing_ok=True)

        result = evaluate_solver(
            solver,
            config=config,
            task_funcs=task_list,
            train_samples=train_samples,
            test_samples=200,
        )
        SweepLogger(log_path).append(result)
        results.append(result)
        agg = result.aggregate
        print(
            f"{solver.name}: pixel_accuracy={agg.mean_pixel_accuracy:.3f} "
            f"pair_accuracy={agg.mean_pair_accuracy:.3f}"
        )

    plot_param_curves(
        results,
        param_name="train_samples",
        output_path=log_dir / "example_advanced_pysr.png",
        title="Performance vs Train Samples (PySRSolver)",
        per_task=True,
    )


if __name__ == "__main__":
    main()
