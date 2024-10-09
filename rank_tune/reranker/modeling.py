import logging
import os
from typing import Dict

import torch
import torch.nn.functional as F
from arguments import DataArguments, ModelArguments
from arguments import ReRankerTrainingArguments as TrainingArguments
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module):

    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments,
                 data_args: DataArguments, train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.config = self.hf_model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if train_args.use_user:
            if train_args.freeze_user_emb:
                self.user_embedding = nn.Embedding.from_pretrained(torch.load(
                    train_args.user_emb_path),
                                                                   freeze=True)
            else:
                self.user_embedding = nn.Embedding.from_pretrained(
                    torch.load(train_args.user_emb_path), freeze=False)

            emb_dim = self.hf_model.config.hidden_size
            self.user_map = nn.Linear(self.user_embedding.weight.shape[1],
                                      emb_dim)

            self.pred_mlp = nn.Sequential(nn.Linear(emb_dim * 2,
                                                    emb_dim), nn.Tanh(),
                                          nn.Linear(emb_dim, emb_dim),
                                          nn.Tanh(), nn.Linear(emb_dim, 1))

    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def compute_score(self, q_d_inputs: Dict[str, torch.Tensor],
                      u_id: torch.Tensor):

        q_d_outputs: SequenceClassifierOutput = self.hf_model(
            **q_d_inputs, output_hidden_states=True, return_dict=True)
        semantic_score = q_d_outputs.logits
        semantic_score = semantic_score.view(
            self.train_args.per_device_train_batch_size,
            self.data_args.num_profile)
        if self.train_args.use_user:
            u_emb = self.user_map(self.user_embedding(u_id))

            q_d_reps = q_d_outputs.hidden_states[-1][:, 0, :]
            q_d_reps = q_d_reps.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.num_profile, -1)
            u_q_d_input = torch.cat([
                u_emb.unsqueeze(1).expand(-1, self.data_args.num_profile, -1),
                q_d_reps
            ],
                                    dim=-1)
            total_score = self.pred_mlp(u_q_d_input).squeeze(-1)

            return q_d_outputs, total_score

        else:
            return q_d_outputs, semantic_score

    def forward(self, batch):
        user_id, batch_input_ids, batch_attention_mask, \
            batch_scores, batch_masks = batch

        q_d_inputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask
        }

        ranker_out, logits = self.compute_score(q_d_inputs, user_id)

        if self.training:
            logits = logits.masked_fill(batch_masks, -torch.inf)
            teacher_scores = F.softmax(batch_scores /
                                       self.model_args.teacher_temperature,
                                       dim=-1)
            student_scores = F.log_softmax(logits /
                                           self.model_args.temperature,
                                           dim=-1)
            distill_loss = F.kl_div(student_scores,
                                    teacher_scores,
                                    reduction="batchmean")
            loss = distill_loss

            return SequenceClassifierOutput(loss=loss, **ranker_out)
        else:
            return ranker_out

    @classmethod
    def from_pretrained(cls, model_args: ModelArguments,
                        data_args: DataArguments,
                        train_args: TrainingArguments, *args, **kwargs):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            *args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        if self.train_args.use_user:
            state_dict = self.state_dict()
            state_dict = type(state_dict)({
                k: v.clone().cpu()
                for k, v in state_dict.items()
            })
            torch.save(state_dict, os.path.join(output_dir, "model.pt"))
        else:
            state_dict = self.hf_model.state_dict()
            state_dict = type(state_dict)({
                k: v.clone().cpu()
                for k, v in state_dict.items()
            })
            self.hf_model.save_pretrained(output_dir, state_dict=state_dict)
