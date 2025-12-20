import random

from tasks import Register, Pair, TaskCollection
from task_list import task_list

def main() -> None:
    # task_list: list[Callable[[Register], Register]], Register: tuple[int, ...]
    # Easily add your own
    def nand(bits: Register) -> Register:
        # bitwise NAND between halves, zero pad on left
        mid = len(bits) // 2
        return tuple(1 - (bits[i] & bits[i + mid]) for i in range(mid)) + (0,) * mid

    task_list.append(nand)

    # Define tasks to be used in this experiment
    # For each, we will generate random strings for input, then pass through hidden function to get output
    collection = TaskCollection(task_list, train_samples=10, test_samples=100)

    for train_samples, eval_fn, task_name in collection.tasks(): # train_samples: list[Pair]
        for pair in train_samples: # pair: Pair[input: Register, output: Register]
            inp = pair.input # Register: tuple[int, ...]
            out = pair.output # Register: tuple[int, ...]
            # Currently both are 16 values long, values restricted to 0 or 1

        # Define solver function using this task's inputs and outputs
        def random_solver(inputs: Register) -> Register:
            return tuple(random.randint(0, 1) for _ in inputs)

        # Pass solver function to task-specific eval_fn, so there can't be test data leakage
        pixel_acc, pair_acc = eval_fn(random_solver)
        print(
            f"{task_name}: pixel_accuracy={pixel_acc:.3f} pair_accuracy={pair_acc:.3f}"
        )

if __name__ == "__main__":
    main()
