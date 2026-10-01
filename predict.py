"""Ensemble models (with TTA) on the test set -> submission.csv.

Usage: python predict.py [glob ...] [--weight ARCH=SHARE ...]   (default: all models/*.pt)
With no --weight, every model gets an equal vote. --weight big=0.5 gives the big models
(averaged together) half the vote and all other models (averaged together) the rest.
Best so far (0.99703): python predict.py 'models/full_*.pt' --weight big=0.5
Also saves the final test probabilities to test_probs.pt (for train.py --pseudo).
"""
import argparse, glob, pandas as pd, torch
from train import load_model, load_test, predict

ap = argparse.ArgumentParser()
ap.add_argument('globs', nargs='*', default=['models/*.pt'])
ap.add_argument('--weight', action='append', default=[], metavar='ARCH=SHARE')
args = ap.parse_args()
weights = {a: float(s) for a, s in (w.split('=') for w in args.weight)}
assert sum(weights.values()) <= 1, 'shares must add up to at most 1'

x = load_test()
assert len(x) == 28000, f'expected 28000 test rows, got {len(x)}'
paths = sorted({p for g in args.globs for p in glob.glob(g)})
assert paths, 'no models found'
groups = {}  # arch named in --weight, or 'rest' -> list of model probabilities
for p in paths:
    ck = torch.load(p, map_location='cpu')
    arch = ck['arch'] if 'arch' in ck else 'cnn'
    groups.setdefault(arch if arch in weights else 'rest', []).append(predict(load_model(p), x, tta=True))
missing = [a for a in weights if a not in groups]
assert not missing, f'no models found for --weight {missing}'
shares = {**weights, 'rest': 1 - sum(weights.values())} if 'rest' in groups else weights
probs = sum(shares[g] * sum(ps) / len(ps) for g, ps in groups.items()) / sum(shares[g] for g in groups)
torch.save(probs, 'test_probs.pt')
pd.DataFrame({'ImageId': range(1, len(x) + 1), 'Label': probs.argmax(1).numpy()}).to_csv('submission.csv', index=False)
print(f'wrote submission.csv from {len(paths)} models: ' +
      ', '.join(f'{g} x{len(ps)} share {shares[g]:.2f}' for g, ps in groups.items()))
