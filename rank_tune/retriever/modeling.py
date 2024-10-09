import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from arguments import DataArguments, ModelArguments
from arguments import RetrieverTrainingArguments as TrainingArguments
from torch import Tensor, nn
from transformers import AutoModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_args.model_name_or_path)
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if train_args.use_user:
            if train_args.freeze_user_emb:
                self.user_embedding = nn.Embedding.from_pretrained(torch.load(
                    train_args.user_emb_path),
                                                                   freeze=True)
            else:
                self.user_embedding = nn.Embedding.from_pretrained(
                    torch.load(train_args.user_emb_path), freeze=False)

            emb_dim = self.model.config.hidden_size
            self.user_map = nn.Linear(self.user_embedding.weight.shape[1],
                                      emb_dim)

        self.normlized = train_args.normlized
        self.sentence_pooling_method = train_args.sentence_pooling_method
        self.use_inbatch_neg = train_args.use_inbatch_neg

        self.negatives_cross_device = train_args.negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError(
                    'Distributed training has not been initialized for representation all gather.'
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state,
                                         features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps: torch.Tensor, p_reps: torch.Tensor,
                           u_id: torch.Tensor):
        q_reps = q_reps.unsqueeze(1)
        semantic_score = torch.matmul(q_reps, p_reps.transpose(-2,
                                                               -1)).squeeze(1)
        if self.train_args.use_user:
            u_emb = self.user_map(self.user_embedding(u_id))
            u_emb = F.normalize(u_emb, dim=-1)
            persona_score = torch.matmul(u_emb.unsqueeze(1),
                                         p_reps.transpose(-2, -1)).squeeze(1)

            score = self.train_args.persona_weight * persona_score + (
                1 - self.train_args.persona_weight) * semantic_score

            return score

        else:
            return semantic_score

    def forward(self, inputs):
        user_id, q_input_ids, q_token_type_ids, q_attention_mask, \
            d_input_ids, d_token_type_ids, d_attention_mask, \
                batch_scores,batch_masks = inputs

        q_reps = self.encode({
            "input_ids": q_input_ids,
            "token_type_ids": q_token_type_ids,
            "attention_mask": q_attention_mask
        })
        p_reps = self.encode({
            "input_ids": d_input_ids,
            "token_type_ids": d_token_type_ids,
            "attention_mask": d_attention_mask
        })

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            p_reps = p_reps.view(self.train_args.per_device_train_batch_size,
                                 self.data_args.num_profile, -1)

            scores = self.compute_similarity(q_reps, p_reps, user_id)
            scores = scores.masked_fill(batch_masks, -torch.inf)

            teacher_scores = F.softmax(batch_scores /
                                       self.model_args.teacher_temperature,
                                       dim=-1)
            student_scores = F.log_softmax(scores /
                                           self.model_args.temperature,
                                           dim=-1)
            distill_loss = F.kl_div(student_scores,
                                    teacher_scores,
                                    reduction="batchmean")
            loss = distill_loss
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        if self.train_args.use_user:
            state_dict = self.state_dict()
            state_dict = type(state_dict)({
                k: v.clone().cpu()
                for k, v in state_dict.items()
            })
            torch.save(state_dict, os.path.join(output_dir, "model.pt"))
        else:
            state_dict = self.model.state_dict()
            state_dict = type(state_dict)({
                k: v.clone().cpu()
                for k, v in state_dict.items()
            })
            self.model.save_pretrained(output_dir, state_dict=state_dict)
