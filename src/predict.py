import os
import argparse
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import *
from config import *
from gradcam import *

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input chest X-ray image")
    args = parser.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pil_image = Image.open(args.image).convert("RGB")
    image_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)
    original_img = cv2.resize(cv2.imread(args.image), (IMAGE_SIZE, IMAGE_SIZE))

    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))
        probs = F.softmax(output, dim=1)
        confidence, class_idx = probs.max(dim=1)

    predicted_class = CLASS_NAMES[class_idx.item()]
    confidence_percentage = confidence.item() * 100

    overlay = gradcam(model, image_tensor, class_idx.item(), original_img)
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    output_path = os.path.join("outputs", f"gradcam_{image_name}.png")
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, overlay)

    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence_percentage:.1f}%")
    print(f"Grad-CAM   : {output_path}")

if __name__ == "__main__":
    predict()
