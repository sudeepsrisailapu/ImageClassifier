# Chest X-Ray Image Classifier with Interpretability

A convolutional neural network (ResNet-18) trained on the [Kaggle Chest X-Ray (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset, with **Grad-CAM** visualizations that highlight the lung regions driving each prediction.

---

## Overview

| Item | Detail |
|------|--------|
| Task | Binary classification: `NORMAL` vs `PNEUMONIA` |
| Model | ResNet-18 pretrained on ImageNet (transfer learning) |
| Dataset | ~5,800 chest X-Ray images (Kaggle) |
| Interpretability | Grad-CAM heatmaps on `layer4` |
| Metrics | Accuracy, AUC-ROC, F1, Confusion Matrix |

---

## Project Structure

```
p4-2026-ImageClassifier/
├── data/                    # dataset directory (gitignored — see setup below)
├── outputs/                 # saved checkpoints, plots, Grad-CAM images (gitignored)
├── src/
│   ├── config.py            # all hyperparameters and paths
│   ├── dataset.py           # DataLoader + augmentation pipeline
│   ├── model.py             # ResNet-18 with custom classification head
│   ├── train.py             # training loop with early stopping
│   ├── evaluate.py          # metrics, ROC curve, confusion matrix
│   └── gradcam.py           # hook-based Grad-CAM + heatmap overlay
├── predict.py               # CLI: single-image inference + Grad-CAM output
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/p4-2026-ImageClassifier.git
cd p4-2026-ImageClassifier
pip install -r requirements.txt
```

### 2. Download the dataset

You need a [Kaggle account](https://www.kaggle.com) and API token (`~/.kaggle/kaggle.json`).

```bash
pip install kaggle
kaggle datasets download paultimothymooney/chest-xray-pneumonia -p data/ --unzip
```

This creates:
```
data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

---

## Usage

### Train

```bash
python src/train.py
```

Trains for up to 20 epochs with early stopping. Best checkpoint saved to `outputs/best_model.pth`.

### Evaluate

```bash
python src/evaluate.py
```

Prints classification report and saves:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`

### Predict on a single image

```bash
python predict.py --image data/chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg
```

Output:
```
Prediction : PNEUMONIA
Confidence : 94.3%
Grad-CAM   : outputs/gradcam_person1_bacteria_1.png
```

---

## Interpretability: Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients of the predicted class score flowing back into the final convolutional layer (`layer4`) to produce a coarse localization map. Brighter regions in the heatmap indicate areas the model relied on most for its prediction.

**Example output:**

> *(After running predict.py, Grad-CAM PNGs are saved to `outputs/` — bright activations typically cluster over infected lung regions in pneumonia cases.)*

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | TBD after training |
| AUC-ROC | TBD after training |
| F1 (Pneumonia) | TBD after training |

*(Update this table after running `src/evaluate.py`)*

---

## Key Design Decisions

- **Transfer learning**: ResNet-18 backbone frozen during warmup, then fine-tuned with differential learning rates (1e-4 for backbone, 1e-3 for head). Medical imaging benefits greatly from ImageNet features.
- **Class imbalance**: The training set has ~3x more PNEUMONIA than NORMAL samples. Handled via weighted random sampling in the DataLoader.
- **Grad-CAM target layer**: `layer4` (last conv block) provides the best trade-off between spatial resolution and semantic abstraction for 224×224 inputs.

---

## Dependencies

- [PyTorch](https://pytorch.org/) — model and training
- [torchvision](https://pytorch.org/vision/) — pretrained ResNet-18, transforms
- [OpenCV](https://opencv.org/) — Grad-CAM heatmap overlay
- [scikit-learn](https://scikit-learn.org/) — AUC-ROC, F1, confusion matrix
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — plots

---

## License

MIT
