"""Train a CNN ensemble on Kaggle Digit Recognizer and write submission.csv.

Usage: python train.py [--arch cnn|wide|res|big] [--models 5] [--first-seed 0] [--epochs 30] [--full]
                      [--pseudo test_probs.pt [--pseudo-thresh 0.9]]
Models are saved to models/ (<arch>_<seed>.pt, or full_<arch>_<seed>.pt with --full;
pl_ is added before <arch> with --pseudo). predict.py ensembles saved models with TTA
and writes test_probs.pt, which --pseudo uses: confident test predictions become extra
training images (never validation images).
"""
import argparse, glob, math, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
def find_data():
    """Local data/, or wherever Kaggle mounted the competition files under /kaggle/input."""
    for f in ['data/train.csv', *sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))]:
        if os.path.exists(f) and os.path.exists(f.replace('train.csv', 'test.csv')): return os.path.dirname(f)
    raise FileNotFoundError('train.csv/test.csv not found in data/ or /kaggle/input. '
                            'On Kaggle: + Add Input -> Competitions -> Digit Recognizer.')
DATA = None

def data_dir():
    global DATA
    DATA = DATA or find_data()
    return DATA

def load_train():
    df = pd.read_csv(f'{data_dir()}/train.csv')
    x = torch.tensor(df.drop(columns='label').values, dtype=torch.float32).view(-1, 1, 28, 28) / 255
    return x, torch.tensor(df['label'].values)

def load_test():
    return torch.tensor(pd.read_csv(f'{data_dir()}/test.csv').values, dtype=torch.float32).view(-1, 1, 28, 28) / 255

MEAN, STD = 0.1307, 0.3081
norm = lambda x: (x - MEAN) / STD

def augment(x, deg=10, shift=0.1, zoom=0.1, g=None):
    """Random rotation/shift/zoom on a batch, done on-device. No flips.
    Pass a CPU generator g for repeatable draws (used by TTA)."""
    n = x.size(0)
    rand = (lambda *s: torch.rand(*s, generator=g).to(x.device)) if g is not None else (lambda *s: torch.rand(*s, device=x.device))
    a = (rand(n) * 2 - 1) * deg * math.pi / 180
    s = 1 + (rand(n) * 2 - 1) * zoom
    t = (rand(n, 2) * 2 - 1) * shift * 2
    theta = torch.stack([torch.stack([s * a.cos(), -s * a.sin(), t[:, 0]], 1),
                         torch.stack([s * a.sin(),  s * a.cos(), t[:, 1]], 1)], 1)
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)

def block(i, o): return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]

class CNN(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.f = nn.Sequential(*block(1, w), *block(w, w), nn.MaxPool2d(2), nn.Dropout(0.25),
                               *block(w, 2 * w), *block(2 * w, 2 * w), nn.MaxPool2d(2), nn.Dropout(0.25),
                               *block(2 * w, 4 * w), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                               nn.Dropout(0.4), nn.Linear(4 * w, 10))
    def forward(self, x): return self.f(norm(x))

class ResBlock(nn.Module):
    """Two 3x3 convs plus a shortcut that carries the input straight through."""
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                               nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.short = nn.Identity() if i == o and stride == 1 else \
            nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))
    def forward(self, x): return F.relu(self.f(x) + self.short(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(*block(1, 32), ResBlock(32, 32), ResBlock(32, 64, 2), ResBlock(64, 64),
                               ResBlock(64, 128, 2), ResBlock(128, 128), nn.AdaptiveAvgPool2d(1),
                               nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 10))
    def forward(self, x): return self.f(norm(x))

def cbr(i, o, k, stride=1, pad=0): return [nn.Conv2d(i, o, k, stride, pad, bias=False), nn.BatchNorm2d(o), nn.ReLU()]

class Big(nn.Module):
    """Strided-conv CNN in the style of strong public Digit Recognizer solutions, widened.
    28 -> 26 -> 24 -> 12 (5x5 stride 2) -> 10 -> 8 -> 4 (5x5 stride 2) -> 1 (4x4)."""
    def __init__(self, w=64):
        super().__init__()
        self.f = nn.Sequential(*cbr(1, w, 3), *cbr(w, w, 3), *cbr(w, w, 5, 2, 2), nn.Dropout(0.4),
                               *cbr(w, 2 * w, 3), *cbr(2 * w, 2 * w, 3), *cbr(2 * w, 2 * w, 5, 2, 2), nn.Dropout(0.4),
                               *cbr(2 * w, 4 * w, 4), nn.Flatten(), nn.Dropout(0.4), nn.Linear(4 * w, 10))
    def forward(self, x): return self.f(norm(x))

ARCHS = {'cnn': CNN, 'wide': lambda: CNN(48), 'res': ResNet, 'big': Big}

def load_model(path):
    """Checkpoints are {'arch', 'state'}; older ones are a bare CNN state_dict."""
    ck = torch.load(path, map_location=dev)
    arch, state = (ck['arch'], ck['state']) if 'arch' in ck else ('cnn', ck)
    m = ARCHS[arch]().to(dev); m.load_state_dict(state)
    return m

@torch.no_grad()
def predict(model, x, tta=False):
    model.eval()
    out = []
    g = torch.Generator().manual_seed(0)
    for i in range(0, len(x), 1000):
        xb = x[i:i + 1000].to(dev)
        p = F.softmax(model(xb), 1)
        if tta:
            for _ in range(4): p = p + F.softmax(model(augment(xb, 6, 0.06, 0.06, g)), 1)
            p = p / 5
        out.append(p.cpu())
    return torch.cat(out)

def train_one(seed, x, y, epochs, val_frac=0.1, arch='cnn', extra=None):
    torch.manual_seed(seed); np.random.seed(seed)
    idx = torch.randperm(len(x))
    nv = int(len(x) * val_frac)
    va, tr = idx[:nv], idx[nv:]
    xtr, ytr, xva, yva = x[tr], y[tr], x[va], y[va]
    if extra is not None: xtr, ytr = torch.cat([xtr, extra[0]]), torch.cat([ytr, extra[1]])
    xtr, ytr = xtr.to(dev), ytr.to(dev)
    model = ARCHS[arch]().to(dev)
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
        if not nv:  # full-data run: no holdout, keep the last epoch
            print(f'{arch} seed {seed} epoch {ep + 1}/{epochs} (full data)', flush=True)
            continue
        acc = (predict(model, xva).argmax(1) == yva).float().mean().item()
        if acc >= best: best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f'{arch} seed {seed} epoch {ep + 1}/{epochs} val_acc {acc:.4f}', flush=True)
    if nv:
        model.load_state_dict(best_state)
        print(f'{arch} seed {seed} best val_acc {best:.4f}', flush=True)
    return model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--full', action='store_true', help='train on all data (no validation) -> models/full_<arch>_<seed>.pt')
    ap.add_argument('--arch', choices=ARCHS, default='cnn')
    ap.add_argument('--first-seed', type=int, default=0)
    ap.add_argument('--pseudo', help='test_probs.pt (or .csv, 10 columns) from predict.py: add confident test predictions to training')
    ap.add_argument('--pseudo-thresh', type=float, default=0.9,
                    help='min ensemble confidence; label smoothing caps it near 0.91, so 0.9 = confident')
    args = ap.parse_args()
    os.makedirs('models', exist_ok=True)
    x, y = load_train()
    extra = None
    if args.pseudo:
        assert os.path.isfile(args.pseudo), f'--pseudo file not found: {args.pseudo!r}'
        teacher = torch.tensor(pd.read_csv(args.pseudo).values, dtype=torch.float32) \
            if args.pseudo.endswith('.csv') else torch.load(args.pseudo)
        conf, lab = teacher.max(1)
        keep = conf >= args.pseudo_thresh
        extra = (load_test()[keep], lab[keep])
        print(f'pseudo-labels: {int(keep.sum())} of {len(keep)} test images at conf >= {args.pseudo_thresh}', flush=True)
    for seed in range(args.first_seed, args.first_seed + args.models):
        path = f"models/{'full_' if args.full else ''}{'pl_' if args.pseudo else ''}{args.arch}_{seed}.pt"
        if os.path.exists(path): print('skip', path); continue
        m = train_one(seed, x, y, args.epochs, val_frac=0 if args.full else 0.1, arch=args.arch, extra=extra)
        torch.save({'arch': args.arch, 'state': m.state_dict()}, path)
