from model import build_model
from dataset import get_dataloaders
from config import MODEL_PATH, DEVICE, CLASS_NAMES
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate():
    _, _, test_loader = get_dataloaders()

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print(classification_report(all_labels, all_preds))
    print("AUC-ROC:", roc_auc_score(all_labels, all_probs))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()