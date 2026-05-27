import torchvision.models as models
import torch.nn as nn

def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


