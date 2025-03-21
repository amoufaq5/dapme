# model_training/nlp/train_nlp_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nlp_model import SimpleNLPModel, collate_fn

class MedicalTextDataset(Dataset):
    def __init__(self, data, tokenizer, label_dict):
        self.data = data
        self.tokenizer = tokenizer
        self.label_dict = label_dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["abstract"]
        label = self.data[idx]["diagnosis_label"]  # e.g. "flu"
        tokens = self.tokenizer(text)
        return tokens, self.label_dict[label]

def train_model(data, tokenizer, label_dict, epochs=5, lr=1e-3):
    dataset = MedicalTextDataset(data, tokenizer, label_dict)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = SimpleNLPModel(vocab_size=10000, embed_dim=128, num_classes=len(label_dict))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            tokens, labels = batch
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    return model

if __name__ == "__main__":
    # Example usage
    # 1) Load from DB
    # 2) Preprocess data: label assignment, tokenization
    # 3) Train
    sample_data = [
        {"abstract": "Patient with fever, runny nose...", "diagnosis_label": "flu"},
        {"abstract": "Headache and stiff neck...", "diagnosis_label": "meningitis"},
        # ...
    ]

    # Mock tokenizer
    def simple_tokenizer(text):
        # Convert text into list of word indices; in reality, might use BERT tokenizer.
        # Just a naive approach for demonstration.
        words = text.lower().split()
        return [hash(w) % 10000 for w in words]  # mod to keep vocab_size = 10000
    
    label_dict = {"flu": 0, "meningitis": 1}  # etc.

    model = train_model(sample_data, simple_tokenizer, label_dict)
    torch.save(model.state_dict(), "nlp_diagnosis_model.pt")
    print("Model saved as nlp_diagnosis_model.pt")
