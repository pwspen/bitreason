import hashlib
from typing import Callable, Generator
from dataclasses import dataclass

Register = tuple[int, ...]


@dataclass
class Pair:
    input: Register
    output: Register


class TaskCollection:
    def __init__(
        self,
        task_func_list: list[Callable[[Register], Register]],
        train_samples: int = 10,
        test_samples: int = 100,
        categories: int = 2,
    ) -> None:
        # Add validation here that each task function is applicable for current settings (lengths, numbers of categories)

        self.task_list = [
            Task(
                transform=func,
                train_pairs=train_samples,
                test_pairs=test_samples,
                categories=categories,
            )
            for func in task_func_list
        ]
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.categories = categories

    def tasks(
        self,
    ) -> Generator[
        tuple[
            tuple[Pair, ...],
            Callable[[Callable[[Register], Register]], tuple[float, float]],
            str,
        ],
        None,
        None,
    ]:
        """
        Yields tuples of (train_samples, eval_func, task_name)
        """
        for task in self.task_list:
            yield (
                task.get_train_pairs(),
                task.evaluate,
                task.name,
            )

    def __len__(self) -> int:
        return len(self.task_list)


class Task:
    def __init__(
        self,
        transform: Callable[[Register], Register],
        train_pairs: int = 10,
        test_pairs: int = 100,
        name: str | None = None,
        template_len: int = 16,
        categories: int = 2,
        # Add way to stack transformations
    ) -> None:
        """
        Returns tuple[train_samples, eval_func]
        """
        if categories != 2 or template_len != 16:
            raise NotImplementedError

        self.transform = transform
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        self.template_len = template_len

        # None indicates bit to be filled in randomly
        # Later might want to re-add ability to add custom templates
        self.templates = [[None] * self.template_len]

        # Extract name from function if not provided
        # Important that transform is not a lambda because
        # name is used in seed for randomness generation
        self.name = name or transform.__name__
        if "<lambda>" in self.name:
            raise ValueError(
                "Transform function must not be a lambda if name is not provided"
            )

    def evaluate(
        self, solver_func: Callable[[Register], Register]
    ) -> tuple[float, float]:
        """
        Evaluates the solver function on test samples.
        Returns (valuewise_acc, pairwise_acc)
        In other words, (fraction of values predicted, fraction of perfect outputs)
        """
        test_pairs_list = [
            self.get_pair(train=False, index=i) for i in range(self.test_pairs)
        ]
        correct_vals, corect_outputs = 0, 0

        for pair in test_pairs_list:
            pred = solver_func(pair.input)

            correct = tuple(a == b for a, b in zip(pair.output, pred))

            correct_val_preds = sum(correct)

            correct_vals += correct_val_preds

            if correct_val_preds == self.template_len:
                corect_outputs += 1

        total_vals = self.test_pairs * self.template_len
        valuewise_acc = correct_vals / total_vals
        pairwise_acc = corect_outputs / self.test_pairs

        return (valuewise_acc, pairwise_acc)

    def get_random_bits(self, n_bits: int, seed: str) -> list[int]:
        # Hash the seed + index to get deterministic randomness
        hash_input = seed.encode()
        hash_value = hashlib.sha256(hash_input).digest()
        # Convert to float between 0 and 1
        result = [x & 1 for x in hash_value[0:n_bits]]
        return result

    def generate(self, train: bool, index: int) -> tuple[int, ...]:
        # Grab enough random bits to select template and fill it in
        # template_len bits for each is enough for filling in and for
        # up to 2**template_len templates

        # This does NOT check for collisions, so there's a probabilistic chance that
        # test problems appear in training

        seed = f"{self.name}:{int(train)}:{str(index)}"
        bits = self.get_random_bits(self.template_len * 2, seed=seed)
        template_select, bits = bits[: self.template_len], bits[self.template_len :]

        if len(self.templates) <= 2:
            # Convert bit string to int and modulo to get idx
            bitstr = "".join(str(b) for b in template_select)
            template_idx = int(bitstr, 2) % len(self.templates)
            template: list = self.templates[template_idx].copy()
        else:
            template = self.templates[0].copy()

        # Fill in template
        none_indices = [i for i, x in enumerate(template) if x is None]
        for i, bit in zip(none_indices, bits):
            template[i] = bit

        if any(type(x) is not int for x in template):
            raise ValueError("Input generation from template failed")

        return tuple(template)

    def get_pair(self, train: bool, index: int) -> Pair:
        inp = self.generate(train, index)
        out = self.transform(inp)
        if len(out) != self.template_len:
            raise ValueError(
                f"Transform function must return list of length {self.template_len}"
            )
        if set(out) - {0, 1}:
            raise ValueError("Transform function must return list of 0s and 1s only")
        return Pair(input=inp, output=out)

    def get_train_pairs(self) -> tuple[Pair, ...]:
        return tuple(
            [self.get_pair(train=True, index=i) for i in range(self.train_pairs)]
        )
