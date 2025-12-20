from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

from tqdm import tqdm

from minireason.tasks import Pair

if TYPE_CHECKING:
    import torch
    from torch import nn

Register = tuple[int, ...]
SolveFn = Callable[[Register], Register]
EvalFn = Callable[[SolveFn], tuple[float, float]]


@dataclass
class TrainingArtifacts:
    loss_curve: list[float]
    pixel_accuracy_curve: list[tuple[int, float]]
    pair_accuracy_curve: list[tuple[int, float]]


class Solver(Protocol):
    name: str

    def reset(self) -> None: ...

    def train(
        self,
        train_pairs: tuple[Pair, ...],
        eval_fn: EvalFn | None = None,
        *,
        task_name: str | None = None,
    ) -> TrainingArtifacts: ...

    def solve(self, inputs: Register) -> Register: ...


class NeuralNetSolver:
    def __init__(
        self,
        *,
        size: int = 16,
        epochs: int = 100,
        test_every: int = 10,
        lr: float = 0.01,
        hsize: int = 64,
        hlayers: int = 2,
        name: str = "neural_net",
    ) -> None:
        self.name = name
        self.size = size
        self.epochs = epochs
        self.test_every = test_every
        self.lr = lr
        self.hsize = hsize
        self.hlayers = hlayers
        self.net: "nn.Module" | None = None
        self._torch_imports: tuple[Any, Any, Any] | None = None

    def _require_torch(self) -> tuple[Any, Any, Any]:
        if self._torch_imports is None:
            try:
                import torch as torch_mod
                import torch.nn as nn_mod
                import torch.optim as optim_mod
            except ImportError as exc:  # pragma: no cover - runtime guard
                msg = (
                    "PyTorch is required for NeuralNetSolver. "
                    "Install with `pip install torch`."
                )
                raise RuntimeError(msg) from exc
            self._torch_imports = (torch_mod, nn_mod, optim_mod)
        return self._torch_imports

    def reset(self) -> None:
        torch_mod, nn_mod, _ = self._require_torch()

        class _BitNet(nn_mod.Module):
            def __init__(self, size: int, hsize: int = 64, hlayers: int = 2) -> None:
                super().__init__()
                self.net = nn_mod.Sequential(
                    nn_mod.Linear(size, hsize),
                    nn_mod.ReLU(),
                    *[
                        layer
                        for _ in range(hlayers)
                        for layer in (
                            nn_mod.Linear(hsize, hsize),
                            nn_mod.ReLU(),
                            nn_mod.Dropout(0.01),
                        )
                    ],
                    nn_mod.Linear(hsize, size),
                    nn_mod.Sigmoid(),
                )

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
                return self.net(x)

        self.net = _BitNet(size=self.size, hsize=self.hsize, hlayers=self.hlayers)

    def solve(self, inputs: Register) -> Register:
        if self.net is None:
            raise RuntimeError("Network not initialized. Call train() first.")
        torch_mod, _, _ = self._require_torch()
        with torch_mod.no_grad():
            tensor_inp = torch_mod.tensor(inputs, dtype=torch_mod.float32)
            pred = self.net(tensor_inp).round().int().tolist()
        return tuple(int(x) for x in pred)

    def train(
        self,
        train_pairs: tuple[Pair, ...],
        eval_fn: EvalFn | None = None,
        *,
        task_name: str | None = None,
    ) -> TrainingArtifacts:
        torch_mod, nn_mod, optim_mod = self._require_torch()
        self.reset()
        assert self.net is not None  # for type checker

        optimizer = optim_mod.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn_mod.MSELoss()

        losses: list[float] = []
        pixel_curve: list[tuple[int, float]] = []
        pair_curve: list[tuple[int, float]] = []

        num_pairs = max(len(train_pairs), 1)
        pbar = tqdm(
            range(self.epochs), desc=f"Training {task_name or self.name}", leave=False
        )

        for epoch in pbar:
            batch_loss = 0.0
            for pair in train_pairs:
                inp_t = torch_mod.tensor(pair.input, dtype=torch_mod.float32)
                out_t = torch_mod.tensor(pair.output, dtype=torch_mod.float32)

                pred = self.net(inp_t)
                loss = criterion(pred, out_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += float(loss.item())

            avg_loss = batch_loss / num_pairs
            losses.append(avg_loss)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

            if eval_fn and (epoch % self.test_every == 0 or epoch == self.epochs - 1):
                pixel_acc, pair_acc = eval_fn(self.solve)
                pixel_curve.append((epoch, float(pixel_acc)))
                pair_curve.append((epoch, float(pair_acc)))

        return TrainingArtifacts(
            loss_curve=losses,
            pixel_accuracy_curve=pixel_curve,
            pair_accuracy_curve=pair_curve,
        )


class RandomGuessSolver:
    def __init__(self, *, name: str = "random_guess") -> None:
        self.name = name

    def reset(self) -> None:
        return None

    def train(
        self,
        train_pairs: tuple[Pair, ...],
        eval_fn: EvalFn | None = None,
        *,
        task_name: str | None = None,
    ) -> TrainingArtifacts:
        return TrainingArtifacts(
            loss_curve=[], pixel_accuracy_curve=[], pair_accuracy_curve=[]
        )

    def solve(self, inputs: Register) -> Register:
        return tuple(random.randint(0, 1) for _ in inputs)
