import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class RetrieverModel(nn.Module):

    def __init__(self,
                 ret_type,
                 model_path,
                 base_model_path,
                 user2id,
                 user_emb_path,
                 batch_size,
                 device,
                 max_length=512,
                 pooling='average',
                 normalize=True):
        super().__init__()

        self.pooling = pooling
        self.normalize = normalize
        self.use_user = False

        if ret_type == 'dense_tune':
            with open(os.path.join(model_path, '1_Pooling/config.json'),
                      'r') as f:
                model_config = json.load(f)

            self.load_user_emb = True

            if "use_user" in model_config.keys():
                self.use_user = model_config['use_user']
            if "persona_weight" in model_config.keys():
                self.persona_weight = model_config['persona_weight']
            if "user_emb_path" in model_config.keys():
                self.user_emb_path = model_config['user_emb_path']
            if "freeze_user_emb" in model_config.keys():
                self.load_user_emb = False
                self.freeze_user_emb = model_config['freeze_user_emb']

            if model_config['pooling_mode_cls_token']:
                self.pooling = 'cls'
            elif model_config['pooling_mode_mean_tokens']:
                self.pooling = 'average'

        if self.use_user:
            self.user2id = user2id
            self.model = AutoModel.from_pretrained(base_model_path)
            emb_dim = self.model.config.hidden_size
            if self.load_user_emb:
                self.user_embedding = torch.load(self.user_emb_path).to(device)
                self.user_map = nn.Linear(self.user_embedding.shape[1],
                                          emb_dim)
            else:
                self.user_embedding = nn.Embedding.from_pretrained(
                    torch.load(self.user_emb_path))
                self.user_map = nn.Linear(self.user_embedding.weight.shape[1],
                                          emb_dim)
            assert os.path.abspath(
                self.user_emb_path) == os.path.abspath(user_emb_path)

            self.load_state_dict(
                torch.load(os.path.join(model_path, 'model.pt')))
        else:
            self.model = AutoModel.from_pretrained(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length

        print("load retriever from: {}".format(model_path))
        print("load tokenizer from: {}".format(model_path))
        print("retriever pooling: {}".format(self.pooling))

    def sentence_embedding(self, hidden_state, mask):
        if self.pooling == 'average':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.pooling == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state,
                                         features['attention_mask'])
        if self.normalize:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, query_emb: torch.Tensor,
                           docs_emb: torch.Tensor, user: int):
        semantic_score = torch.matmul(query_emb, docs_emb.T).squeeze(0)
        if self.use_user:
            if self.load_user_emb:
                user_emb = self.user_embedding[[self.user2id[user]]]
            else:
                user_emb = self.user_embedding(
                    torch.LongTensor([self.user2id[user]]).to(self.device))
            user_emb = self.user_map(user_emb)
            user_emb = F.normalize(user_emb, dim=-1)

            persona_score = torch.matmul(user_emb,
                                         docs_emb.transpose(-2, -1)).squeeze(0)

            score = self.persona_weight * persona_score + (
                1 - self.persona_weight) * semantic_score
            return score

        else:
            return semantic_score

    @torch.no_grad()
    def retrieve_topk_dense(self, corpus, profile, query, user, topk):
        query_tokens = self.tokenizer([query],
                                      padding=True,
                                      truncation=True,
                                      max_length=self.max_length,
                                      return_tensors='pt').to(self.device)

        query_emb = self.encode(query_tokens)
        scores = []

        for batch_idx in range(0, len(corpus), self.batch_size):
            batch_corpus = corpus[batch_idx:batch_idx + self.batch_size]

            tokens_corpus = self.tokenizer(batch_corpus,
                                           padding=True,
                                           truncation=True,
                                           max_length=self.max_length,
                                           return_tensors='pt').to(self.device)

            docs_emb = self.encode(tokens_corpus)

            temp_scores = self.compute_similarity(query_emb, docs_emb, user)

            scores.extend(temp_scores.tolist())

        topk_values, topk_indices = torch.topk(torch.tensor(scores),
                                               min(topk, len(scores)))

        return [profile[m]
                for m in topk_indices.tolist()], topk_values.tolist()
