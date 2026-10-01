# Digit Recognizer

CNN ensemble for Kaggle's [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/).
Best public score so far: **0.99703** (8 big CNNs holding half the vote, 15 CNN / wide CNN / ResNet models the other half; all trained on full data).

## Local

Put `train.csv` and `test.csv` in `data/`, then:

```sh
python train.py --full --arch wide --models 3      # arch: cnn | wide | res | big
python predict.py                                   # all models/*.pt -> submission.csv + test_probs.pt
python predict.py 'models/full_*.pt' --weight big=0.5   # best: big models get half the vote
python train.py --full --pseudo test_probs.pt --arch res --models 3 --first-seed 10 --epochs 20
```

Leave out `--full` to hold back 10% for validation (prints `val_acc`).

## On Kaggle (free GPU)

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

**Reusing local files:** upload them as a Kaggle Dataset (**+ Add Input** in the notebook), then point at `/kaggle/input/<dataset>/`:

```python
!python train.py --full --pseudo /kaggle/input/<dataset>/test_probs_v3_0.99653.pt --arch wide --models 3 --first-seed 10 --epochs 20
!python predict.py 'models/*.pt' '/kaggle/input/<dataset>/full_*.pt'   # full_* skips the test_probs file
```
