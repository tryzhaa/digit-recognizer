"""Interactive digit recognizer demo using Gradio.

Run locally: python app.py
Deploy to HF Spaces: git push space [this repo]
"""
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob


MEAN, STD = 0.1307, 0.3081


def load_models(device='cpu'):
    """Load all trained models from models/ directory."""
    from train import ARCHS
    model_files = sorted(glob.glob('models/full_*.pt'))
    if not model_files:
        model_files = sorted(glob.glob('models/*.pt'))

    if not model_files:
        print("Warning: No trained models found.")
        return None

    models = []
    for path in model_files:
        try:
            ck = torch.load(path, map_location=device)
            if isinstance(ck, dict) and 'arch' in ck:
                arch, state = ck['arch'], ck['state']
            else:
                arch, state = 'cnn', ck
            model = ARCHS[arch]().to(device)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    return models if models else None


HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
"""

FORCE_DARK = """
() => {
  document.documentElement.classList.add('dark');
  document.body.classList.add('dark');
}
"""

CSS = """
:root,
body,
body.dark,
.gradio-container,
.gradio-container.dark,
.dark {
  --ground: #08090d;
  --surface: #101218;
  --elevated: #161922;
  --line: #232736;
  --line-strong: #2e3346;
  --acc: #00e5a0;
  --acc-dim: #0a7f5c;
  --txt: #e8eaf0;
  --muted: #8b93a7;
  --faint: #7f879b;
  --ui: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
  color-scheme: dark;
}

/* Dark unconditionally: the surface ships one theme, not two. */
body,
.gradio-container,
gradio-app {
  background: var(--ground) !important;
  color: var(--txt) !important;
  font-family: var(--ui) !important;
}

.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 0 16px 72px !important;
}

footer,
.gradio-container > .main > .wrap > .footer,
.built-with,
.show-api {
  display: none !important;
}

/* Strip the stock chrome so our own surfaces read cleanly. */
.block:not(#readout),
.form,
.gr-box,
.gr-panel {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

/* ---------- browser surfaces ---------- */

::selection {
  background: var(--acc-dim);
  color: #f2fffa;
}

* {
  scrollbar-width: thin;
  scrollbar-color: var(--line-strong) transparent;
}

::-webkit-scrollbar {
  width: 11px;
  height: 11px;
}

::-webkit-scrollbar-track {
  background: var(--ground);
}

::-webkit-scrollbar-thumb {
  background: var(--line-strong);
  border: 3px solid var(--ground);
  border-radius: 99px;
}

::-webkit-scrollbar-thumb:hover {
  background: #3d4459;
}

:focus-visible {
  outline: 2px solid var(--acc);
  outline-offset: 2px;
  border-radius: 4px;
}

/* ---------- masthead ---------- */

#masthead {
  padding: 56px 0 8px;
}

#masthead .hd-line {
  display: flex;
  align-items: center;
  gap: 14px;
}

#masthead .hd-mark {
  width: 34px;
  height: 34px;
  flex: none;
  color: var(--acc);
}

#masthead h1 {
  margin: 0;
  font-family: var(--ui);
  font-size: clamp(1.9rem, 4.6vw, 2.9rem);
  font-weight: 600;
  letter-spacing: -0.03em;
  line-height: 1.02;
  color: var(--txt);
  text-wrap: balance;
}

#masthead p {
  margin: 18px 0 0;
  max-width: 62ch;
  font-size: 1.0625rem;
  line-height: 1.6;
  color: var(--muted);
}

#rule {
  height: 1px;
  margin: 36px 0 32px;
  background: linear-gradient(90deg, var(--line-strong), var(--line) 38%, transparent);
}

/* ---------- canvas well ---------- */

#stage {
  gap: 28px !important;
  flex-wrap: wrap !important;
  align-items: start !important;
}

/* Gradio pins columns to min-width:320px inline; two of them overflow a phone. */
#stage > .column {
  min-width: min(300px, 100%) !important;
}

#pad-wrap {
  position: relative;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--elevated);
  padding: 14px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.45), 0 16px 38px -14px rgba(0, 0, 0, 0.7);
  transition: border-color 220ms cubic-bezier(0.16, 1, 0.3, 1);
}

#pad-wrap:hover,
#pad-wrap:focus-within {
  border-color: var(--line-strong);
}

#pad-wrap label[data-testid="block-label"] {
  display: none !important;
}

/* The drawing surface is white on purpose: the stroke is black and
   predict_digit inverts the result into MNIST polarity. */
#pad-wrap .image-container,
#pad-wrap .pixi-target,
#pad-wrap canvas {
  border-radius: 10px !important;
}

/* No overflow clipping here: the editor toolbar is positioned against the
   top edge and gets cut off by it. */
#pad-wrap .image-container {
  background: #f4f5f7 !important;
}

#pad-wrap .empty {
  color: #6d7382 !important;
  font-family: var(--ui) !important;
  font-size: 0.875rem !important;
}

/* Gradio's PIXI editor toolbar. It is position:absolute, so anchor it to the
   canvas rather than the dark well, where it would sit dark-on-dark. */
#pad-wrap .image-container {
  position: relative !important;
}

#pad-wrap .icon-button-wrapper.top-panel {
  top: 10px !important;
  right: 10px !important;
}

#pad-wrap .icon-button-wrapper {
  --block-label-text-color: #c9cfdc;
  background: var(--surface) !important;
  border: 1px solid var(--line-strong) !important;
  border-radius: 10px !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.55), 0 12px 28px -12px rgba(0, 0, 0, 0.75) !important;
  padding: 3px !important;
  opacity: 1 !important;
  visibility: visible !important;
}

#pad-wrap .icon-button {
  border-radius: 7px !important;
  transition: color 160ms ease-out, background 160ms ease-out !important;
}

#pad-wrap .icon-button:not([disabled]):hover {
  color: var(--acc) !important;
  background: var(--elevated) !important;
}

#pad-wrap .icon-button[aria-pressed="true"],
#pad-wrap .icon-button.selected {
  color: var(--acc) !important;
  background: rgba(0, 229, 160, 0.13) !important;
}

.html-container,
.prose {
  width: 100% !important;
  max-width: none !important;
  padding: 0 !important;
}

#pad-note {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 4px 16px;
  margin-top: 14px;
  font-size: 0.8125rem;
  color: var(--faint);
}

#pad-note > span {
  white-space: nowrap;
}

#pad-note .pn-spec {
  font-family: var(--mono);
  font-size: 0.75rem;
  letter-spacing: 0.02em;
}

/* ---------- readout ---------- */

#readout {
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  background: linear-gradient(180deg, var(--surface), #0c0e14) !important;
  padding: 26px 26px 22px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.45), 0 16px 38px -14px rgba(0, 0, 0, 0.7) !important;
}

.dr-hero {
  display: flex;
  align-items: baseline;
  gap: 18px;
  min-height: 92px;
}

.dr-glyph {
  font-family: var(--ui);
  font-size: clamp(3.25rem, 10vw, 5.25rem);
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 0.86;
  color: var(--acc);
  font-variant-numeric: lining-nums;
}

.dr-conf {
  font-family: var(--mono);
  font-size: 0.9375rem;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  color: var(--txt);
}

.dr-hero.is-idle {
  align-items: center;
}

.dr-hero.is-idle .dr-conf {
  font-family: var(--ui);
  font-size: 1rem;
  color: var(--muted);
}

.dr-hero.is-err .dr-glyph {
  color: #ff8d7a;
}

.dr-note {
  margin: 0;
  font-size: 0.9375rem;
  line-height: 1.55;
  color: var(--muted);
}

.dr-err {
  font-family: var(--mono);
  font-size: 0.8125rem;
  line-height: 1.6;
  color: #ff8d7a;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}

.dr-table {
  margin-top: 26px;
  padding-top: 20px;
  border-top: 1px solid var(--line);
}

.dr-cap {
  font-family: var(--mono);
  font-size: 0.6875rem;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--faint);
  margin-bottom: 14px;
}

.dr-row {
  display: grid;
  grid-template-columns: 1.25rem 1fr 4.5rem;
  align-items: center;
  gap: 14px;
  padding: 3px 0;
}

.dr-digit {
  font-family: var(--mono);
  font-size: 0.875rem;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  color: var(--faint);
  text-align: center;
}

.dr-row.is-win .dr-digit {
  color: var(--acc);
  font-weight: 700;
}

.dr-track {
  position: relative;
  height: 7px;
  border-radius: 99px;
  background: #1a1e28;
  overflow: hidden;
}

.dr-fill {
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 99px;
  background: var(--acc-dim);
  transform-origin: left center;
  animation: dr-grow 560ms cubic-bezier(0.16, 1, 0.3, 1) both;
}

.dr-row.is-win .dr-fill {
  background: linear-gradient(90deg, var(--acc-dim), var(--acc));
}

.dr-pct {
  font-family: var(--mono);
  font-size: 0.8125rem;
  font-variant-numeric: tabular-nums;
  text-align: right;
  color: var(--muted);
}

.dr-row.is-win .dr-pct {
  color: var(--txt);
  font-weight: 600;
}

.dr-unit {
  color: var(--faint);
  margin-left: 1px;
}

.dr-row.is-idle .dr-fill {
  animation: none;
}

@keyframes dr-grow {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}

@media (prefers-reduced-motion: reduce) {
  .dr-fill { animation: none; }
}

/* ---------- spec sheet ---------- */

#about {
  margin-top: 64px;
  padding-top: 32px;
  border-top: 1px solid var(--line);
}

#about h2 {
  margin: 0 0 18px;
  font-size: 1.0625rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--txt);
}

#about p {
  margin: 0 0 26px;
  max-width: 68ch;
  font-size: 0.9375rem;
  line-height: 1.68;
  color: var(--muted);
}

#about dl {
  margin: 0;
  max-width: 640px;
}

#about div.sp-row {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 11px 0;
  border-top: 1px solid var(--line);
}

#about dt {
  flex: none;
  width: 11rem;
  font-size: 0.8125rem;
  color: var(--faint);
}

#about dd {
  margin: 0;
  font-family: var(--mono);
  font-size: 0.8125rem;
  font-variant-numeric: tabular-nums;
  color: var(--txt);
}

#about a {
  color: var(--acc);
  text-decoration: underline;
  text-decoration-color: var(--acc-dim);
  text-underline-offset: 3px;
  transition: text-decoration-color 180ms ease-out;
}

#about a:hover {
  text-decoration-color: var(--acc);
}

@media (max-width: 760px) {
  #masthead { padding-top: 36px; }
  #pad-wrap .wrap,
  #pad-wrap .image-container,
  #pad-wrap .sketchpad { min-height: 300px !important; }
  #readout { padding: 20px 18px 16px; }
  .dr-hero { min-height: 0; gap: 14px; }
  #about dt { width: 8.5rem; }
  #about div.sp-row { flex-wrap: wrap; gap: 4px 16px; }
}
"""

MARK = """
<svg class="hd-mark" viewBox="0 0 32 32" fill="none" aria-hidden="true">
  <rect x="1.6" y="1.6" width="28.8" height="28.8" rx="3.2" stroke="currentColor"
        stroke-width="1.3" opacity="0.42"/>
  <path d="M7 11.3h18M7 16h18M7 20.7h18" stroke="currentColor" stroke-width="1" opacity="0.15"/>
  <path d="M11 21.4c2.1-7.4 4.9-11 6.9-10.4 1.7.5 1.2 3.7-1.3 6.8-2.1 2.6-3.4 3.7-2.3 4.7 1.1 1 3.5.1 6.7-2.6"
        stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def _rows(probs=None, pred=None):
    """Ten distribution rows. With no probs, render the frame at rest."""
    out = []
    for d in range(10):
        if probs is None:
            out.append(
                f'<div class="dr-row is-idle"><span class="dr-digit">{d}</span>'
                f'<span class="dr-track"></span>'
                f'<span class="dr-pct">&mdash;</span></div>'
            )
            continue
        p = float(probs[d]) * 100
        win = ' is-win' if d == pred else ''
        # Floor the bar so a near-zero class still reads as a tick, not nothing.
        w = max(p, 0.5)
        out.append(
            f'<div class="dr-row{win}"><span class="dr-digit">{d}</span>'
            f'<span class="dr-track"><span class="dr-fill" style="width:{w:.4f}%"></span></span>'
            f'<span class="dr-pct">{p:.2f}<span class="dr-unit">%</span></span></div>'
        )
    return ''.join(out)


def _shell(hero, table=True, probs=None, pred=None):
    body = hero
    if table:
        body += ('<div class="dr-table"><div class="dr-cap">Distribution</div>'
                 + _rows(probs, pred) + '</div>')
    return body


def readout_idle():
    hero = ('<div class="dr-hero is-idle">'
            '<span class="dr-conf">Waiting for a stroke</span></div>')
    return _shell(hero)


def readout_result(probs, pred, conf):
    hero = (f'<div class="dr-hero"><span class="dr-glyph">{pred}</span>'
            f'<span class="dr-conf">{conf * 100:.1f}% confident</span></div>')
    return _shell(hero, probs=probs, pred=pred)


def readout_message(text):
    hero = ('<div class="dr-hero is-idle">'
            f'<span class="dr-conf">{text}</span></div>')
    return _shell(hero)


def readout_error(text):
    return (f'<div class="dr-hero is-idle is-err">'
            f'<span class="dr-conf">Prediction failed</span></div>'
            f'<div class="dr-table"><div class="dr-cap">Traceback</div>'
            f'<pre class="dr-err">{text}</pre></div>')


def predict_digit(image_array, models, device):
    """Predict digit from hand-drawn image."""
    if image_array is None:
        return readout_idle()

    # Gradio 4.x Sketchpad returns a dict with 'composite' key
    if isinstance(image_array, dict):
        composite = image_array.get('composite')
        layers = image_array.get('layers', [])
        image_array = composite if composite is not None else (layers[0] if layers else None)
    if image_array is None:
        return readout_idle()

    # Convert to grayscale float, inverted to match MNIST (white digit on black bg)
    if not isinstance(image_array, np.ndarray):
        image_array = np.array(image_array)
    # Use RGB channels regardless of whether RGBA or RGB
    rgb = image_array[:, :, :3].astype(np.float32)
    img = 1.0 - (np.mean(rgb, axis=2) / 255.0)  # invert: black stroke → white digit

    # Blank paper (including after the toolbar's clear) has nothing to read.
    if img.max() < 0.05:
        return readout_idle()

    # Resize to 28x28 if needed
    if img.shape != (28, 28):
        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((28, 28), Image.LANCZOS)
        img = np.array(pil_img) / 255.0

    # Normalize
    img = (img - MEAN) / STD

    # Convert to tensor
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    if models is None:
        return readout_message('No models loaded')

    # Ensemble prediction
    logits_list = []
    for model in models:
        model = model.to(device)
        with torch.no_grad():
            logits = model(x)
            logits_list.append(logits)

    avg_logits = torch.stack(logits_list).mean(0)
    probs = F.softmax(avg_logits, dim=1)[0].cpu().detach().numpy()

    pred = int(np.argmax(probs))
    return readout_result(probs, pred, probs[pred])


def create_demo(models, device):
    """Create and return Gradio interface."""

    def predict_fn(image):
        try:
            return predict_digit(image, models, device)
        except Exception as e:
            import traceback
            msg = f"{e}\n{traceback.format_exc()}"
            print(msg)
            import html
            return readout_error(html.escape(msg))

    n_models = len(models) if models else 0

    with gr.Blocks(title="Digit Recognizer") as demo:
        gr.HTML(
            f'<div id="masthead"><div class="hd-line">{MARK}'
            f'<h1>Handwritten digit recognizer</h1></div>'
            f'<p>Draw a digit from 0 to 9. A {n_models}-model convolutional ensemble '
            f'reads the stroke and reports its confidence across every class.</p></div>'
            f'<div id="rule"></div>'
        )

        with gr.Row(elem_id="stage"):
            with gr.Column(scale=1):
                with gr.Column(elem_id="pad-wrap"):
                    canvas = gr.Sketchpad(
                        label="Draw a digit here",
                        type="numpy",
                        value=None,
                        interactive=True,
                        show_label=False,
                        container=False,
                        layers=False,
                        transforms=(),
                        sources=(),
                        canvas_size=(360, 360),
                        fixed_canvas=True,
                        # Stroke stays black on white: predict_digit inverts to MNIST polarity.
                        brush=gr.Brush(colors=["#000000"], default_color="#000000",
                                       color_mode="fixed", default_size=22),
                        eraser=gr.Eraser(default_size=24),
                    )
                gr.HTML(
                    '<div id="pad-note"><span>Draw large and centered.</span>'
                    '<span class="pn-spec">downsampled to 28&times;28</span></div>'
                )

            with gr.Column(scale=1):
                readout = gr.HTML(readout_idle(), elem_id="readout")

        gr.HTML(
            '<section id="about"><h2>About this model</h2>'
            '<p>An ensemble of convolutional networks &mdash; standard, wide, residual, '
            'and large strided variants &mdash; each trained independently and averaged at '
            'inference. Training used rotation, shift, and zoom augmentation on the 42,000 '
            'labeled digits from the '
            '<a href="https://www.kaggle.com/competitions/digit-recognizer/" '
            'target="_blank" rel="noopener">Kaggle Digit Recognizer</a> set, with confident '
            'test predictions folded back in as pseudo-labels for a second round.</p>'
            '<dl>'
            '<div class="sp-row"><dt>Test accuracy</dt><dd>99.70%</dd></div>'
            '<div class="sp-row"><dt>Errors</dt><dd>83 / 28,000</dd></div>'
            f'<div class="sp-row"><dt>Ensemble size</dt><dd>{n_models} models</dd></div>'
            '<div class="sp-row"><dt>Training set</dt><dd>42,000 labeled digits</dd></div>'
            '<div class="sp-row"><dt>Input</dt><dd>28 &times; 28 grayscale</dd></div>'
            '</dl></section>'
        )

        canvas.change(predict_fn, inputs=canvas, outputs=readout)

    return demo


if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading models...")
    models = load_models(device)
    if models:
        print(f"Loaded {len(models)} models")
    else:
        print("No models found - running in demo mode")

    demo = create_demo(models, device)
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port,
                theme=gr.themes.Base(), css=CSS, head=HEAD, js=FORCE_DARK)
