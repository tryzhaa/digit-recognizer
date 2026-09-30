"""Train a CNN ensemble on Kaggle Digit Recognizer and write submission.csv.

Usage: python train.py [--models 5] [--epochs 30]
Models are saved to models/; predictions use all saved models (ensemble + TTA).
"""
import argparse, math, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def load_train():
    df = pd.read_csv('data/train.csv')
    x = torch.tensor(df.drop(columns='label').values, dtype=torch.float32).view(-1, 1, 28, 28) / 255
    return x, torch.tensor(df['label'].values)

MEAN, STD = 0.1307, 0.3081
norm = lambda x: (x - MEAN) / STD

def augment(x, deg=10, shift=0.1, zoom=0.1):
    """Random rotation/shift/zoom on a batch, done on-device. No flips."""
    n = x.size(0)
    a = (torch.rand(n, device=x.device) * 2 - 1) * deg * math.pi / 180
    s = 1 + (torch.rand(n, device=x.device) * 2 - 1) * zoom
    t = (torch.rand(n, 2, device=x.device) * 2 - 1) * shift * 2
    theta = torch.stack([torch.stack([s * a.cos(), -s * a.sin(), t[:, 0]], 1),
                         torch.stack([s * a.sin(),  s * a.cos(), t[:, 1]], 1)], 1)
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)

def block(i, o): return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(*block(1, 32), *block(32, 32), nn.MaxPool2d(2), nn.Dropout(0.25),
                               *block(32, 64), *block(64, 64), nn.MaxPool2d(2), nn.Dropout(0.25),
                               *block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                               nn.Dropout(0.4), nn.Linear(128, 10))
    def forward(self, x): return self.f(norm(x))

@torch.no_grad()
def predict(model, x, tta=False):
    model.eval()
    out = []
    for i in range(0, len(x), 1000):
        xb = x[i:i + 1000].to(dev)
        p = F.softmax(model(xb), 1)
        if tta:
            g = torch.Generator(device='cpu').manual_seed(0)
            for _ in range(4): p = p + F.softmax(model(augment(xb, 6, 0.06, 0.06)), 1)
            p = p / 5
        out.append(p.cpu())
    return torch.cat(out)

def train_one(seed, x, y, epochs, val_frac=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    idx = torch.randperm(len(x))
    nv = int(len(x) * val_frac)
    va, tr = idx[:nv], idx[nv:]
    xtr, ytr, xva, yva = x[tr].to(dev), y[tr].to(dev), x[va], y[va]
    model = CNN().to(dev)
    bs = 128
    steps = epochs * math.ceil(len(xtr) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state = 0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs]
            loss = loss_fn(model(augment(xtr[b])), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = (predict(model, xva).argmax(1) == yva).float().mean().item()
        if acc >= best: best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f'seed {seed} epoch {ep + 1}/{epochs} val_acc {acc:.4f}', flush=True)
    model.load_state_dict(best_state)
    print(f'seed {seed} best val_acc {best:.4f}', flush=True)
    return model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30)
    args = ap.parse_args()
    os.makedirs('models', exist_ok=True)
    x, y = load_train()
    for seed in range(args.models):
        path = f'models/cnn_{seed}.pt'
        if os.path.exists(path): print('skip', path); continue
        torch.save(train_one(seed, x, y, args.epochs).state_dict(), path)
