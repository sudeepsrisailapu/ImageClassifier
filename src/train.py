from config import *
from dataset import *
from model import build_model
import torch
import torch.nn as nn

def train():
    model = build_model()

    backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
    optimizer = torch.optim.Adam([
        {"params": backbone_params,          "lr": LR_BACKBONE},
        {"params": model.fc.parameters(),   "lr": LR_HEAD}
    ])

    criterion = nn.CrossEntropyLoss()
    
    train_loader, val_loader, _ = get_dataloaders()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_val_loss = 0.0

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

        
        val_loss = total_val_loss / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load(MODEL_PATH))
    return model