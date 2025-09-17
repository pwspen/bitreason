import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from task import Task

# Define transformations
def not_(bits):
    return [1 - b for b in bits]

def and_(bits):
    mid = len(bits) // 2
    return [bits[i] & bits[i + mid] for i in range(mid)] + [0]*mid

def or_(bits):
    mid = len(bits) // 2
    return [bits[i] | bits[i + mid] for i in range(mid)] + [0]*mid

def xor(bits):
    mid = len(bits) // 2
    return [bits[i] ^ bits[i + mid] for i in range(mid)] + [0]*mid

def sum_(bits):
    # add halves as binary numbers
    mid = len(bits) // 2
    a = ''.join([str(i) for i in bits[:mid]])
    b = ''.join([str(i) for i in bits[mid:]])

    res = int(a, 2) + int(b, 2)
    bin_str = bin(res)[2:].zfill(len(bits))
    return [int(b) for b in bin_str]

def shift(bits):
    shift = 8
    return bits[shift:] + bits[:shift]

def flip(bits):
    return bits[::-1]

def tile(bits):
    seg = 4
    return bits[:seg] * (len(bits) // seg) + [0] * (len(bits) % seg)

def count(bits):
    num_ones = sum(bits)
    # convert to binary
    bin_str = bin(num_ones)[2:].zfill(len(bits))
    return [int(b) for b in bin_str]

def separate(bits):
    num_ones = sum(bits)
    num_zeros = len(bits) - num_ones
    return [1]*num_ones + [0]*num_zeros

tasks = [not_, and_, or_, xor, sum_, shift, flip, tile, count, separate]

# Simple neural network
class BitNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        hsize = 64
        hlayers = 3
        self.net = nn.Sequential(
            nn.Linear(size, hsize),
            nn.ReLU(),
            *[layer for _ in range(hlayers) for layer in (nn.Linear(hsize, hsize), nn.ReLU())],
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
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
print("Training networks...")
examples = 5
tests = 2
for task_fn in tqdm(tasks, desc="Overall Progress"):
    task = Task(task_fn)
    inp, out = task.get_pair(example=True, index=0)
    print(f"{task_fn.__name__}:\n{inp}\n{out}\n")
    net, losses, accuracies = train_net(task, examples=examples, tests=tests)
    epochs, accs = zip(*accuracies)
    ax.plot(epochs, accs, label=task.name, linewidth=2)

ax.set_title(f'Train set: {examples} | Test set: {tests}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.7)
ax.legend()

plt.tight_layout()
plt.savefig('training_accuracies.png', dpi=300, bbox_inches='tight')