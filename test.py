import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable

from tasks import TaskCollection, Pair
from task_list import task_list


# Simple neural network
class BitNet(nn.Module):
    def __init__(self, size):
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

    def forward(self, x):
        return self.net(x)


# Training function with tqdm
def train_net(
    train_pairs: tuple[Pair, ...],
    eval_fn: Callable,
    task_name,
    epochs=100,
    test_every=10,
):
    net = BitNet(size=16)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []  # [loss]
    accuracies = []  # [(epoch, acc)]
    solverates = []  # [(epoch, solverate)]

    pbar = tqdm(range(epochs), desc=f"Training {task_name}")

    for epoch in pbar:
        batch_loss = 0
        for pair in train_pairs:
            inp, out = pair.input, pair.output
            inp_t = torch.tensor(inp, dtype=torch.float32)
            out_t = torch.tensor(out, dtype=torch.float32)

            pred = net(inp_t)
            loss = criterion(pred, out_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        avg_loss = batch_loss / train_samples
        losses.append(avg_loss)
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

        if epoch % test_every == 0:
            with torch.no_grad():
                solve_fn = lambda inp: tuple(
                    int(x)
                    for x in net(torch.tensor(inp, dtype=torch.float32))
                    .round()
                    .int()
                    .tolist()
                )
                pixel_acc, pair_acc = eval_fn(solve_fn)

            accuracies.append((epoch, pixel_acc))
            solverates.append((epoch, pair_acc))

    return net, losses, accuracies, solverates


# Train networks and plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
print("Training networks...")
train_samples = 100
test_samples = 1000

all_tasks = TaskCollection(task_list, train_samples=train_samples, test_samples=test_samples)

for pairs, eval_fn, task_name in tqdm(all_tasks.tasks(), desc="Tasks Completed", total=len(task_list)):
    net, losses, accuracies, solverates = train_net(
        train_pairs=pairs, eval_fn=eval_fn, task_name=task_name
    )

    # Plot losses
    ax1.plot(range(len(losses)), losses, label=task_name, linewidth=2)

    # Plot accuracies
    epochs_acc, accs = zip(*accuracies)
    ax2.plot(epochs_acc, accs, label=task_name, linewidth=2)

    # Plot solve rates
    epochs_solve, solve_rates = zip(*solverates)
    ax3.plot(epochs_solve, solve_rates, label=task_name, linewidth=2)

# Configure loss subplot
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.7)
ax1.legend()

# Configure accuracy subplot
ax2.set_title(f"Test Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.7)
ax2.legend()

# Configure solve rate subplot
ax3.set_title("Test Solve Rate")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Solve Rate")
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.7)
ax3.legend()

fig.suptitle(f"Neural net per task (Train pairs: {train_samples} | Test pairs: {test_samples})")

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300, bbox_inches="tight")