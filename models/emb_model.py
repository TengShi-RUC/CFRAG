import torch
import torch.nn as nn
from transformers import AutoModel


class EmbModel(nn.Module):

    def __init__(self, model_path, pooling='average', normalize=True) -> None:
        super().__init__()
        self.emb_model = AutoModel.from_pretrained(model_path)
        self.emb_dim = self.emb_model.config.hidden_size
        self.pooling = pooling
        self.normalize = normalize

    def forward(self, **kwargs):
        model_output = self.emb_model(**kwargs)
        attention_mask = kwargs['attention_mask']

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)

        if self.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling == "cls":
            emb = last_hidden[:, 0]
        else:
            raise ValueError('Pooling Error')

        if self.normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb
