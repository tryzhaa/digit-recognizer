"""Generate a figure of misclassified digits from validation set.

Usage: python visualize_errors.py [--output errors.png]
"""
import argparse
import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import model architectures and data loading from train.py
from train import load_train, ARCHS, MEAN, STD


def load_models(device='cpu'):
    """Load all trained models from models/ directory."""
    model_files = sorted(glob.glob('models/full_*.pt'))
    if not model_files:
        model_files = sorted(glob.glob('models/*.pt'))

    if not model_files:
        raise FileNotFoundError('No trained models found in models/ directory')

    models = []
    for path in model_files:
        try:
            ck = torch.load(path, map_location=device)
            # Checkpoints are {'arch': ..., 'state': ...}; older ones are bare state dicts (CNN)
            if isinstance(ck, dict) and 'arch' in ck:
                arch, state = ck['arch'], ck['state']
            else:
                arch, state = 'cnn', ck
            model = ARCHS[arch]().to(device)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
            print(f"Loaded {path} ({arch})")
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    return models


def predict_ensemble(x, models, device='cpu', batch_size=512):
    """Get ensemble predictions for a dataset."""
    all_probs = []
    x = x.to(device)

    for model in models:
        model = model.to(device)
        probs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                xb = x[i:i + batch_size]
                p = F.softmax(model(xb), dim=1)
                probs.append(p.cpu())
        all_probs.append(torch.cat(probs))

    avg_probs = torch.stack(all_probs).mean(0)
    return avg_probs.argmax(1)


def get_misclassified(x_val, y_val, models, device='cpu'):
    """Find misclassified examples in validation set."""
    preds = predict_ensemble(x_val, models, device)
    wrong = preds != y_val.cpu()
    indices = torch.where(wrong)[0]

    return [
        {'image': x_val[idx].cpu(), 'true': y_val[idx].item(), 'pred': preds[idx].item()}
        for idx in indices
    ]


def plot_misclassified(misclassified, output_path='errors.png'):
    """Plot misclassified digits in a grid."""
    n = len(misclassified)
    cols = 10
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(14, 2 * rows), dpi=100)
    fig.suptitle(f'{n} misclassified digits', fontsize=16, fontweight='bold', y=0.998)

    for i, item in enumerate(misclassified):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(item['image'].squeeze().numpy(), cmap='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{item["true"]}→{item["pred"]}', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='errors.png')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading data...")
    x_train, y_train = load_train()

    # Hold back last 10% as validation (same split as train.py seed 0)
    n_train = int(0.9 * len(x_train))
    x_val, y_val = x_train[n_train:], y_train[n_train:]
    print(f"Validation: {len(x_val)} images")

    print("Loading models...")
    models = load_models(device)
    print(f"Loaded {len(models)} models")

    print("Finding misclassified examples...")
    misclassified = get_misclassified(x_val, y_val, models, device)
    print(f"Found {len(misclassified)} misclassified digits")

    plot_misclassified(misclassified, args.output)
