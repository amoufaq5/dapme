# model_training/vision/train_vision_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from vision_model import SimpleVisionModel

def train_vision_model(data_dir, epochs=5, lr=1e-3):
    # data_dir is a path with subfolders for each class: e.g. data_dir/flu, data_dir/pneumonia, ...
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = SimpleVisionModel(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    return model, dataset.classes

if __name__ == "__main__":
    model, classes = train_vision_model(data_dir="path_to_medical_images", epochs=5)
    torch.save(model.state_dict(), "vision_diagnosis_model.pt")
    print("Vision model saved as vision_diagnosis_model.pt")
    print("Classes:", classes)
