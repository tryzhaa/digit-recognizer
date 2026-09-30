"""Ensemble models (with TTA) on the test set -> submission.csv.

Usage: python predict.py [glob ...]   (default: all models/*.pt)
Also saves the averaged test probabilities to test_probs.pt (for train.py --pseudo).
"""
import glob, sys, pandas as pd, torch
from train import load_model, load_test, predict

x = load_test()
assert len(x) == 28000, f'expected 28000 test rows, got {len(x)}'
paths = sorted({p for g in sys.argv[1:] or ['models/*.pt'] for p in glob.glob(g)})
assert paths, 'no models found'
probs = 0
for p in paths:
    probs = probs + predict(load_model(p), x, tta=True)
probs = probs / len(paths)
torch.save(probs, 'test_probs.pt')
pd.DataFrame({'ImageId': range(1, len(x) + 1), 'Label': probs.argmax(1).numpy()}).to_csv('submission.csv', index=False)
print(f'wrote submission.csv from {len(paths)} models')
