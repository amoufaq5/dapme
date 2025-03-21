# model_training/nlp/nlp_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNLPModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, num_classes=2):
        super(SimpleNLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is a list of token index lists, we need to handle variable lengths
        # We'll just sum embeddings for simplicity.
        embedded = []
        for tokens in x:
            emb = self.embedding(torch.tensor(tokens, dtype=torch.long))
            emb_sum = emb.mean(dim=0)  # average of embeddings
            embedded.append(emb_sum)
        embedded = torch.stack(embedded)
        out = F.relu(self.fc1(embedded))
        out = self.fc2(out)
        return out

def collate_fn(batch):
    tokens_list, labels_list = zip(*batch)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    return tokens_list, labels_tensor
