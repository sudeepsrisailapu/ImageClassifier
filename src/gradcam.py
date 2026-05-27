from model import build_model
from config import DEVICE
import cv2
import numpy as np
import torch

activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations["layer4"] = output

def backward_hook(module, grad_in, grad_out):
    gradients["layer4"] = grad_out[0]

def gradcam(model, image_tensor, class_idx, original_img):
    model.to(DEVICE)
    model.eval()

    model.layer4.register_forward_hook(forward_hook)
    model.layer4.register_full_backward_hook(backward_hook)

    output = model(image_tensor.to(DEVICE))
    score = output[0, class_idx]
    model.zero_grad()
    score.backward()

    grads = gradients["layer4"]
    acts  = activations["layer4"]

    weights = grads.mean(dim=(2, 3))
    cam = (weights[:, :, None, None] * acts).sum(dim=1)
    cam = torch.relu(cam)              

    cam = cam.squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)

    return overlay





