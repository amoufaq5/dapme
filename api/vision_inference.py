# api/vision_inference.py

import io
import torch
from PIL import Image
from torchvision import transforms
from model_training.vision.vision_model import SimpleVisionModel

# Load model
model_path = "vision_diagnosis_model.pt"
vision_model = SimpleVisionModel(num_classes=2)
if torch.cuda.is_available():
    vision_model.load_state_dict(torch.load(model_path))
    vision_model.cuda()
else:
    vision_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
vision_model.eval()

# Dummy class names
classes = ["normal", "abnormal"]

def vision_predict(image_bytes: bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        outputs = vision_model(image)
        pred = torch.argmax(outputs, dim=1).item()
        return classes[pred]
