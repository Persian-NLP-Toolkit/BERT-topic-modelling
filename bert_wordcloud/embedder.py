from bertopic.backend import BaseEmbedder
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"


class ParsBERTEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 16):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.batch_size = batch_size

    def embed(self, docs, verbose=False):
        embs = []
        for start in range(0, len(docs), self.batch_size):
            batch = list(docs[start : start + self.batch_size])
            inp = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            with torch.no_grad():
                out = self.model(**inp)
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        if not embs:
            return np.zeros((0, self.model.config.hidden_size))
        return np.vstack(embs)
