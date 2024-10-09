import json
import os
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class ReRankerModel(nn.Module):

    def __init__(self,
                 rerank_type,
                 model_path,
                 base_model_path,
                 user2id,
                 user_emb_path,
                 batch_size,
                 device,
                 max_length=512):
        super().__init__()

        self.use_user = False
        if rerank_type == 'cross_tune':
            with open(os.path.join(model_path, 'user_config/config.json'),
                      'r') as f:
                user_config = json.load(f)

            self.load_user_emb = True

            if "use_user" in user_config.keys():
                self.use_user = user_config['use_user']
            if "user_emb_path" in user_config.keys():
                self.user_emb_path = user_config['user_emb_path']
            if "freeze_user_emb" in user_config.keys():
                self.load_user_emb = False
                self.freeze_user_emb = user_config['freeze_user_emb']

        if self.use_user:
            self.user2id = user2id
            self.hf_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path)
            emb_dim = self.hf_model.config.hidden_size
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

            self.pred_mlp = nn.Sequential(nn.Linear(emb_dim * 2,
                                                    emb_dim), nn.Tanh(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.Tanh(), nn.Linear(emb_dim, 1))

            self.load_state_dict(
                torch.load(os.path.join(model_path, 'model.pt')))
        else:
            self.hf_model = AutoModelForSequenceClassification.from_pretrained(
                model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("load reranker from: {}".format(model_path))
        print("load tokenizer from: {}".format(model_path))

        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length

    def compute_score(self, q_d_inputs: Dict[str, torch.Tensor], user: int):
        q_d_outputs: SequenceClassifierOutput = self.hf_model(
            **q_d_inputs, output_hidden_states=True, return_dict=True)
        semantic_score = q_d_outputs.logits.squeeze(-1)
        if self.use_user:
            u_id = self.user2id[user]

            if self.load_user_emb:
                user_emb = self.user_embedding[[u_id]]
            else:
                user_emb = self.user_embedding(
                    torch.LongTensor([u_id]).to(self.device))
            u_emb = self.user_map(user_emb)

            q_d_reps = q_d_outputs.hidden_states[-1][:, 0, :]
            u_q_d_input = torch.cat(
                [u_emb.expand(q_d_reps.shape[0], -1), q_d_reps], dim=-1)
            total_score = self.pred_mlp(u_q_d_input).squeeze(-1)
            return total_score

        else:
            return semantic_score

    @torch.no_grad()
    def rerank_topk(self, corpus, profile, query, user, topk):
        truncate_query_token = self.tokenizer(query,
                                              max_length=self.max_length // 2,
                                              truncation=True)
        query = self.tokenizer.batch_decode(
            [truncate_query_token['input_ids']], skip_special_tokens=True)[0]

        for i in range(len(corpus)):
            truncate_doc_token = self.tokenizer(corpus[i],
                                                max_length=self.max_length //
                                                2,
                                                truncation=True)
            corpus[i] = self.tokenizer.batch_decode(
                [truncate_doc_token['input_ids']], skip_special_tokens=True)[0]

        corpus_pairs = [[query, x] for x in corpus]
        scores = []

        for batch_idx in range(0, len(corpus), self.batch_size):
            batch_pairs = corpus_pairs[batch_idx:batch_idx + self.batch_size]
            tokens_pairs = self.tokenizer(batch_pairs,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors='pt').to(self.device)
            batch_scores = self.compute_score(tokens_pairs, user)
            scores.extend(batch_scores.tolist())

        topk_values, topk_indices = torch.topk(torch.tensor(scores),
                                               min(topk, len(scores)))
        selected_profs = [profile[m] for m in topk_indices.tolist()]

        return selected_profs, topk_values.tolist()
