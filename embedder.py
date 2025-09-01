from bertopic.backend import BaseEmbedder
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class ParsBERTEmbedder(BaseEmbedder):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        self.model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

    def embed(self, docs, verbose=False):
        embs = []
        for d in docs:
            inp = self.tokenizer(d, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            with torch.no_grad():
                out = self.model(**inp)
            embs.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
        return np.array(embs)
