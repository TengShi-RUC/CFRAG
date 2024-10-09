import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

import utils
from arguments import DataArguments, ModelArguments, TrainingArguments


class UserEncoder(nn.Module):

    def __init__(self, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.device = train_args.device


        if data_args.freeze_emb:
            self.corpus_embedding = nn.Embedding.from_pretrained(torch.load(
                data_args.corpus_emb_path),
                                                                 padding_idx=0,
                                                                 freeze=True)
        else:
            self.corpus_embedding = nn.Embedding.from_pretrained(torch.load(
                data_args.corpus_emb_path),
                                                                 padding_idx=0,
                                                                 freeze=False)
        logging.info("load corpus embedding from: {} freeze: {}".format(
            data_args.corpus_emb_path, data_args.freeze_emb))

        self.corpus_trans = nn.Linear(self.corpus_embedding.weight.shape[1],
                                      model_args.emb_dim)

        self.pos = PositionalEmbedding(data_args.max_profile_len,
                                       model_args.emb_dim)
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=model_args.emb_dim,
            nhead=model_args.num_heads,
            dim_feedforward=model_args.emb_dim,
            dropout=model_args.dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformerEncoderLayer, num_layers=model_args.num_layers)

        self.infoNCE = InfoNCE(
            batch_size=train_args.per_device_train_batch_size,
            hidden_dim=model_args.emb_dim,
            sim_metric=model_args.sim_metric,
            sim_map=model_args.sim_map,
            sim_activate=model_args.sim_activate,
            infoNCE_temp=model_args.infoNCE_temp,
            infoNCE_temp_learn=model_args.infoNCE_temp_learn,
            device=self.device)
        self.to(self.device)

    def encode_corpus(self, corpus):

        corpus_emb = self.corpus_embedding(corpus)

        if self.corpus_embedding.weight.shape[1] != self.model_args.emb_dim:
            corpus_emb = self.corpus_trans(corpus_emb)

        return corpus_emb

    def forward(self, corpus, corpus_mask):
        corpus = corpus.to(self.device)
        corpus_mask = corpus_mask.to(self.device)

        corpus_emb = self.encode_corpus(corpus).reshape(
            (corpus_mask.shape[0], self.data_args.max_profile_len, -1))

        corpus_emb += self.pos(corpus_emb)
        corpus_encoded = self.transformer_encoder(
            src=corpus_emb, src_key_padding_mask=corpus_mask)

        corpus_encoded = corpus_encoded.masked_fill(corpus_mask.unsqueeze(2),
                                                    0)
        corpus_emb_mean = corpus_encoded.sum(dim=1) / (~corpus_mask).sum(
            dim=1, keepdim=True)
        return corpus_emb_mean

    def loss(self, corpus_1, corpus_1_mask, corpus_2, corpus_2_mask):
        corpus_1_emb = self.forward(corpus_1, corpus_1_mask)
        corpus_2_emb = self.forward(corpus_2, corpus_2_mask)
        loss = self.infoNCE(corpus_1_emb, corpus_2_emb)
        return {"total_loss": loss}

    def get_user_emb(self, corpus, corpus_mask):
        return self.forward(corpus, corpus_mask)

    def save_model(self):
        model_path = self.train_args.model_path
        utils.check_dir(model_path)
        logging.info("save model to: {}".format(model_path))
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path=None):
        logging.info("load model from: {}".format(model_path))
        self.load_state_dict(torch.load(model_path, map_location=self.device))


class InfoNCE(nn.Module):

    def __init__(self, batch_size, hidden_dim, sim_metric, sim_map,
                 sim_activate, infoNCE_temp, infoNCE_temp_learn,
                 device) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.sim_metric = sim_metric
        self.sim_map = sim_map
        self.sim_activate = sim_activate

        if infoNCE_temp_learn:
            self.infoNCE_temp = nn.Parameter(torch.ones([]) * infoNCE_temp)
        else:
            self.infoNCE_temp = infoNCE_temp

        if sim_map:
            self.weight_matrix = nn.Parameter(
                torch.randn((hidden_dim, hidden_dim)))
            nn.init.xavier_normal_(self.weight_matrix)

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(self.batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, his_1_mean: torch.Tensor, his_2_mean: torch.Tensor):
        batch_size = his_1_mean.size(0)
        N = 2 * batch_size

        if self.sim_metric == 'cosine':
            his_1_mean = F.normalize(his_1_mean, p=2, dim=-1)
            his_2_mean = F.normalize(his_2_mean, p=2, dim=-1)

        z = torch.cat([his_1_mean, his_2_mean], dim=0)

        if self.sim_map:
            sim = torch.mm(torch.mm(z, self.weight_matrix), z.T)
        else:
            sim = torch.mm(z, z.T)

        # logging.info(f"type: {type(self.sim_activate)}")
        if self.sim_activate is None:
            sim = sim / self.infoNCE_temp
        elif self.sim_activate == 'tanh':
            sim = torch.tanh(sim) / self.infoNCE_temp
        else:
            logging.info("sim_activate: {}".format(self.sim_activate))
            raise NotImplementedError

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)

        return info_nce_loss


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


