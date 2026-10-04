# Digit Recognizer

## What this project is

This project teaches a computer to read handwritten digits (0–9). It's my entry for Kaggle's [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/) competition.

Kaggle gives 42,000 example images with the right answer, and 28,000 images without. The program learns from the first set, then guesses the digit in each of the 28,000 test images.

## Result

**99.70% correct** (Kaggle score 0.99703). That's 27,917 of the 28,000 test digits right, and 83 wrong.

### Try it live

**[🎨 Draw a digit on HuggingFace Spaces](https://huggingface.co/spaces/tryzhaa/digit-recognizer)** — see the model's prediction in real-time.

### Error analysis

The model's 83 mistakes are mostly **genuinely ambiguous digits** — many could reasonably be interpreted as multiple different numbers. You can visualize them:

```sh
python visualize_errors.py --output errors.png
```

This generates a grid of all misclassified digits with red borders, showing what the model predicted vs. the true label.

## How it works

The program uses neural networks that learn shapes, like loops and lines, instead of single pixels. It trains several of them with slightly different designs, then lets them vote on each digit. A group vote makes fewer mistakes than any one network alone.

| Step | Score |
| --- | --- |
| 5 small networks voting | 99.58% |
| Added 2 different kinds of network | 99.65% |
| Added bigger networks, gave them half the vote | **99.70%** |

---

## How to run it

### Setup

Install dependencies:

```sh
pip install -r requirements.txt
```

### On your computer

#### Interactive demo (drawing)

```sh
python app.py
```

Opens a local Gradio interface where you can draw digits and get live predictions.

#### Training & evaluation

Put `train.csv` and `test.csv` in `data/`, then:

```sh
python train.py --full --arch wide --models 3      # arch: cnn | wide | res | big
python predict.py                                   # all models/*.pt -> submission.csv + test_probs.pt
python predict.py 'models/full_*.pt' --weight big=0.5   # best: big models get half the vote
python train.py --full --pseudo test_probs.pt --arch res --models 3 --first-seed 10 --epochs 20
```

Leave out `--full` to hold back 10% for validation (prints `val_acc`).

### On Kaggle (free GPU)

1. Competition page → **Code → New Notebook**. Check the data panel on the right lists Digit Recognizer; if not, **+ Add Input → Competitions → Digit Recognizer**. `train.py` finds the files anywhere under `/kaggle/input`.
2. Notebook settings: **Accelerator → GPU**, **Internet → On** (both need a phone-verified account).
3. Run these cells:

```python
!git clone https://github.com/tryzhaa/digit-recognizer
%cd digit-recognizer
!nvidia-smi -L
```

```python
!python train.py --full --arch wide --models 3
!python train.py --full --arch res --models 3
!python predict.py
!cp submission.csv /kaggle/working/
```

   Strongest single design (1.3M parameters; best on a GPU):

```python
!python train.py --full --arch big --models 3 --first-seed 30 --epochs 45
```

4. For long runs, use **Save Version → Save & Run All**. Everything in `/kaggle/working/` (including `digit-recognizer/models/`) is kept as the version's output, and `submission.csv` can be submitted from there.

**Reusing local files:** upload them as a Kaggle Dataset (**+ Add Input** in the notebook), then point at `/kaggle/input/<dataset>/`. Upload the teacher probabilities as `.csv`, because Kaggle unpacks `.pt` files:

```python
!python train.py --full --pseudo /kaggle/input/<dataset>/test_probs_v9_0.99703.csv --arch big --models 3 --first-seed 10 --epochs 20
!python predict.py 'models/*.pt' '/kaggle/input/<dataset>/full_*.pt'   # full_* skips the test_probs file
```

---

## Deployment

The demo runs on [Render](https://render.com) as a web service, configured by
`render.yaml`. Pushes to `main` deploy automatically.

    https://digit-recognizer-x8cb.onrender.com

The instance is on Render's free plan, which shapes two things visitors notice:

- It spins down when idle, so the first request after a pause takes 50s or more.
- CPU is the binding constraint, so the demo serves a subset of the ensemble
  rather than all 27 models. `ENSEMBLE_SIZE` sets how many (default 4) — raise it
  for accuracy, lower it for speed. The served subset agrees with the full
  ensemble on 99.95% of test digits.

---

## GitHub visibility (for recruiters)

To make this project stand out:

1. **Repository About**: Add description "99.7% accuracy MNIST classifier using CNN ensemble. Train, evaluate, and draw digits live on Hugging Face Spaces."
2. **Topics**: Add `pytorch`, `cnn`, `kaggle`, `ensemble`, `mnist`, `neural-network`
3. **Pin this README** so visitors see the live demo link first
