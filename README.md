# Plant Disease Detector

A deep learning model that classifies plant leaf diseases from images, with Grad-CAM
visualization to show which regions of the leaf drove the prediction, wrapped in a
Gradio app for interactive use.

## Project Structure

```
plant-disease-detector/
├── README.md
├── .gitignore
├── requirements.txt
├── DEBUGGING_LOG.md
├── notebooks/
│   └── plant_disease_detector.ipynb   # final, self-written notebook
├── early-attempts/
│   ├── 01_data_leakage_early_attempt.ipynb
│   └── 02_gradcam_early_attempt.ipynb
├── app/
│   └── gradio_app.py
└── assets/
    ├── training_history.png
    └── gradcam_samples/
```

## Dataset

PlantVillage dataset, downloaded via the Kaggle API. Not committed to this repo —
see `notebooks/plant_disease_detector.ipynb` for the download and `split-folders`
setup steps.

## Model

- Backbone: `EfficientNetV2B0` (`include_preprocessing=True`, no manual `preprocess_input`)
- Training: Phase 1 (10 epochs, frozen base) → Fine-tuning (5 epochs, unfrozen), EarlyStopping enabled
- Final train/val split done once, correctly, before any preprocessing (see DEBUGGING_LOG.md
  for why this matters)

Results: _fill in after final training run — accuracy, loss, training curves in `assets/`_

## Grad-CAM

Visualizes which parts of the leaf the model used to make its prediction, using a
Grad-CAM implementation that automatically locates the last conv layer in
EfficientNetV2B0. Sample outputs in `assets/gradcam_samples/`.

## App

A Gradio interface wraps the trained model + Grad-CAM overlay for interactive testing.
Run with:

```
python app/gradio_app.py
```

## Learning Journey

This project went through a few iterations before the final version. Early on I
worked with AI assistance while learning TensorFlow/Keras and ran into two real bugs:

1. **Data leakage** — an early version scored 98%+ accuracy, which turned out to be
   because of two inconsistent train/test splits rather than one clean split.
2. **Grad-CAM gradient errors** — an early implementation threw `gradients are None`
   because the feature map and prediction weren't in the same gradient path.

Those early, AI-assisted attempts are kept in `early-attempts/` rather than deleted,
with notes on what went wrong. The final notebook in `notebooks/` was rewritten by me
from scratch once I understood both issues, without relying on AI to write the fix.
See `DEBUGGING_LOG.md` for the full writeup.

## Setup

```
pip install -r requirements.txt
```

Kaggle credentials (`kaggle.json`) are kept locally / in Google Drive and are never
committed — see `.gitignore`.
