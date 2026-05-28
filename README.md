# Chest X-Ray Image Classifier with Interpretability

A neural network (ResNet-18) that is trained on the [Kaggle Chest X-Ray (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset, with **Grad-CAM** visualizations that highlight the lung regions driving each prediction.

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


Trains for up to 20 epochs with early stopping. Best checkpoint saved to `outputs/best_model.pth`.

```bash
python src/evaluate.py
```

Prints classification report and saves:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`

### Predict on a single image

Example prediction:
```bash
python predict.py --image data/chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg
```

Example Output:
```
Prediction : PNEUMONIA
Confidence : 94.3%
Grad-CAM   : outputs/gradcam_person1_bacteria_1.png
```

---

## Interpretability: Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients of the predicted class score flowing back into the final convolutional layer (`layer4`) to produce a coarse localization map. Brighter regions in the heatmap indicate areas the model relied on most for its prediction.****

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | .83 or 83% |
| AUC-ROC | 0.9552870918255533 |
| F1 (Pneumonia) | .88 |

The model was able to correctly identify 389 of the 390 test images provided in the pneumonia dataset, yielding a 1.00 recall rate, indicating the model's sensitivity to pneumonia cases. On the other hand, the model yields a recall rate of .56 for NORMAL, reflecting a conservative bias that causes cases that might not be pneumonia to still be flagged as the illness. However, this seems appropriate in a medical situation where they would rather have a false alarm rather than letting the illness be waived off without 100% confidence. This tradeoff in the results could also be due to the large amount of pneumonia case images in comparison to patients without the illness.

## Confusion Matrix
<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/9a581ed7-b4f5-45a1-bf1d-4a797609ee23" />

## Prediction Results

Pneumonia: gradcam_person100_bacteria_475.png

<img width="224" height="224" alt="gradcam_person100_bacteria_475" src="https://github.com/user-attachments/assets/e7f57245-4635-4afe-ae51-c5545ef46750" />

Normal: gradcam_IM-0029-0001.png

<img width="224" height="224" alt="gradcam_IM-0029-0001" src="https://github.com/user-attachments/assets/7fd0c460-f5fa-42eb-b7ae-f012292cf6fc" />

The heatmaps above show which regions of the X-ray were used by the model when trying to predict whether the patient in the image had pneumonia or not. The brighter regions indicate which areas of the image had a stronger influence on the prediction.

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
