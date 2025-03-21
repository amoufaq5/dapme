# api/nlp_inference.py

import torch
from model_training.nlp.nlp_model import SimpleNLPModel

# Load model
model_path = "nlp_diagnosis_model.pt"  # Make sure this is available
nlp_model = SimpleNLPModel(vocab_size=10000, embed_dim=128, num_classes=2)
if torch.cuda.is_available():
    nlp_model.load_state_dict(torch.load(model_path))
    nlp_model.cuda()
else:
    nlp_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
nlp_model.eval()

# Dummy label dict
label_dict = {0: "flu", 1: "meningitis"}

def simple_tokenizer(text):
    words = text.lower().split()
    return [hash(w) % 10000 for w in words]

def nlp_predict(text):
    """
    Return predicted label from the loaded NLP model.
    """
    with torch.no_grad():
        tokens = [simple_tokenizer(text)]
        outputs = nlp_model(tokens)
        pred = torch.argmax(outputs, dim=1).item()
        return label_dict[pred]
