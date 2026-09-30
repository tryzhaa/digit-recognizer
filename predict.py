"""Ensemble all models in models/ (with TTA) on data/test.csv -> submission.csv."""
import glob, sys, pandas as pd, torch
from train import CNN, predict, dev

df = pd.read_csv('data/test.csv')
x = torch.tensor(df.values, dtype=torch.float32).view(-1, 1, 28, 28) / 255
assert len(x) == 28000, f'expected 28000 test rows, got {len(x)}'
paths = sorted(glob.glob('models/cnn_*.pt'))
probs = 0
for p in paths:
    m = CNN().to(dev); m.load_state_dict(torch.load(p, map_location=dev))
    probs = probs + predict(m, x, tta=True)
pd.DataFrame({'ImageId': range(1, len(x) + 1), 'Label': probs.argmax(1).numpy()}).to_csv('submission.csv', index=False)
print(f'wrote submission.csv from {len(paths)} models')
