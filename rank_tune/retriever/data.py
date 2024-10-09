import json
import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from arguments import DataArguments
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainDataCollator():

    args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch):
        batch = [x[0] for x in batch]
        query_input = [x['query'] for x in batch]
        doc_input = sum([x['docs'] for x in batch], [])

        query_tokenized = self.tokenizer(query_input,
                                         truncation=True,
                                         max_length=self.args.query_max_len,
                                         padding=True,
                                         return_tensors='pt')
        doc_tokenized = self.tokenizer(doc_input,
                                       truncation=True,
                                       max_length=self.args.passage_max_len,
                                       padding=True,
                                       return_tensors='pt')

        batch_scores = torch.stack([b['scores'] for b in batch], dim=0)
        user_id = torch.LongTensor([b['user_id'] for b in batch])
        batch_masks = torch.stack([b['masks'] for b in batch], dim=0)

        return user_id, query_tokenized['input_ids'], query_tokenized[
            'token_type_ids'], query_tokenized[
                'attention_mask'], doc_tokenized['input_ids'], doc_tokenized[
                    'token_type_ids'], doc_tokenized[
                        'attention_mask'], batch_scores, batch_masks


class TrainDataset(Dataset):

    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        logger.info("load data")
        if os.path.isdir(args.train_data):
            self.data = []
            for file in sorted(os.listdir(args.train_data)):
                logger.info("load file: {}".format(file))
                with open(os.path.join(args.train_data, file), 'r') as f:
                    temp_dataset = json.load(f)
                self.data.extend(temp_dataset)
        else:
            logger.info("load file: {}".format(
                os.path.basename(args.train_data)))
            with open(args.train_data, 'r') as f:
                self.data = json.load(f)

        with open(os.path.join(args.user_vocab_path, 'user2id.pkl'),
                  'rb') as file:
            self.user2id = pickle.load(file)

        self.dataset = []
        logger.info("preprocess data")
        for group_idx in tqdm(range(len(self.data))):
            group = self.data[group_idx]
            group_data = group['data']
            user_id = group['user_id']

            query = group_data[0]['query']
            sort_group = sorted(group_data,
                                key=lambda x:
                                (x[args.main_metric], x['user_sim']),
                                reverse=True)

            cur_docs = []
            cur_scores = []
            cur_masks = []
            for i in range(max(len(sort_group), args.num_profile)):
                if i < len(sort_group):
                    cur_docs.append(sort_group[i]['doc'])
                    if args.main_metric in ['MSE', 'MAE', 'RMSE']:
                        cur_scores.append(-sort_group[i][args.main_metric])
                    else:
                        cur_scores.append(sort_group[i][args.main_metric])
                    cur_masks.append(False)
                else:
                    cur_docs.append('')
                    cur_scores.append(-np.inf)
                    cur_masks.append(True)
            self.dataset.append({
                "user_id": self.user2id[user_id],
                "query": query,
                "docs": cur_docs,
                "scores": cur_scores,
                "masks": cur_masks
            })

        self.total_len = len(self.dataset)
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        cur_data = self.dataset[index]

        if self.args.query_instruction_for_retrieval is not None:
            query = cur_data[
                'query'] + self.args.query_instruction_for_retrieval
        else:
            query = cur_data['query']

        if self.args.passage_instruction_for_retrieval is not None:
            docs = [
                self.args.passage_instruction_for_retrieval + d
                for d in cur_data['docs']
            ]
        else:
            docs = cur_data['docs']

        return [{
            "user_id": cur_data['user_id'],
            "query": query,
            "docs": docs,
            "scores": torch.tensor(cur_data['scores']),
            "masks": torch.tensor(cur_data['masks'])
        }]
