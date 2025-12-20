from pathlib import Path

from evaluation import evaluate_solver
from solvers import NeuralNetSolver, RandomGuessSolver
from sweep import RunConfig, SweepLogger
from task_list import task_list


def main() -> None:
    solvers = [
        NeuralNetSolver(epochs=100, test_every=10, lr=0.01),
        RandomGuessSolver(),
    ]
    log_dir = Path("results/examples")
    log_dir.mkdir(parents=True, exist_ok=True)

    for solver in solvers:
        config = RunConfig(parameters={"solver": solver.name, "train_samples": 200})
        log_path = log_dir / f"{solver.name}.jsonl"
        log_path.unlink(missing_ok=True)

        result = evaluate_solver(
            solver,
            config=config,
            task_funcs=task_list,
            train_samples=200,
            test_samples=200,
        )
        SweepLogger(log_path).append(result)
        agg = result.aggregate
        print(
            f"{solver.name}: pixel_accuracy={agg.mean_pixel_accuracy:.3f} "
            f"pair_accuracy={agg.mean_pair_accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
