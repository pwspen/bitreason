import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from task import Task
from tasklist import tasks

# Simple neural network
class BitNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        hsize = 64
        hlayers = 2
        self.net = nn.Sequential(
            nn.Linear(size, hsize),
            nn.ReLU(),
            *[layer for _ in range(hlayers) for layer in (nn.Linear(hsize, hsize), nn.ReLU(), nn.Dropout(0.01))],
            nn.Linear(hsize, size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Training function with tqdm
def train_net(transform, epochs=500, examples=4, tests=2, test_every=10):
    net = BitNet(size=16)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    losses = []  # [loss]
    accuracies = []  # [(epoch, acc)]
    solverates = [] # [(epoch, solverate)]

    pbar = tqdm(range(epochs), desc=f"Training {transform.name}")
    
    for epoch in pbar:
        batch_loss = 0
        for i in range(examples):
            inp, out = transform.get_pair(example=True, index=i)
            inp_t = torch.tensor(inp, dtype=torch.float32)
            out_t = torch.tensor(out, dtype=torch.float32)
            
            pred = net(inp_t)
            loss = criterion(pred, out_t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
        
        avg_loss = batch_loss / examples
        losses.append(avg_loss)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

        if epoch % test_every == 0:
            accs = []
            solves = 0
            with torch.no_grad():
                for i in range(tests):
                    inp, target = transform.get_pair(example=False, index=i)
                    pred = net(torch.tensor(inp, dtype=torch.float32))
                    pred_bits = (pred > 0.5).int().tolist()
                    acc = sum(a == b for a, b in zip(target, pred_bits)) / len(target)
                    accs.append(acc)
                    if acc == 1.0:
                        solves += 1

            avg_acc = sum(accs) / len(accs)
            accuracies.append((epoch, avg_acc))
            solve_rate = solves / tests
            solverates.append((epoch, solve_rate))
    
    return net, losses, accuracies, solverates

# Train networks and plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
print("Training networks...")
examples = 5
tests = 2

for task_fn in tqdm(tasks, desc="Overall Progress"):
    task = Task(task_fn)
    inp, out = task.get_pair(example=True, index=0)
    print(f"{task_fn.__name__}:\n{inp}\n{out}\n")
    net, losses, accuracies, solverates = train_net(task, examples=examples, tests=tests)
    
    # Plot losses
    ax1.plot(range(len(losses)), losses, label=task.name, linewidth=2)
    
    # Plot accuracies
    epochs_acc, accs = zip(*accuracies)
    ax2.plot(epochs_acc, accs, label=task.name, linewidth=2)
    
    # Plot solve rates
    epochs_solve, solve_rates = zip(*solverates)
    ax3.plot(epochs_solve, solve_rates, label=task.name, linewidth=2)

# Configure loss subplot
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.7)
ax1.legend()

# Configure accuracy subplot
ax2.set_title(f'Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.7)
ax2.legend()

# Configure solve rate subplot
ax3.set_title('Test Solve Rate')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Solve Rate')
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.7)
ax3.legend()

fig.suptitle(f'Neural net per task (Train pairs: {examples} | Test pairs: {tests})')

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')