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
        batch_input = sum([x['input'] for x in batch], [])
        batch_tokenized = self.tokenizer(batch_input,
                                         truncation=True,
                                         max_length=self.args.max_len,
                                         padding=True,
                                         return_tensors='pt')

        batch_scores = torch.stack([b['scores'] for b in batch], dim=0)
        batch_masks = torch.stack([b['masks'] for b in batch], dim=0)
        user_id = torch.LongTensor([b['user_id'] for b in batch])

        return user_id, batch_tokenized['input_ids'], batch_tokenized[
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
        data = self.dataset[index]

        query = data['query']
        truncate_query_token = self.tokenizer(query,
                                              max_length=self.args.max_len //
                                              2,
                                              truncation=True)
        query = self.tokenizer.batch_decode(
            [truncate_query_token['input_ids']], skip_special_tokens=True)[0]

        docs = data['docs']
        for i in range(len(docs)):
            truncate_doc_token = self.tokenizer(docs[i],
                                                max_length=self.args.max_len //
                                                2,
                                                truncation=True)
            docs[i] = self.tokenizer.batch_decode(
                [truncate_doc_token['input_ids']], skip_special_tokens=True)[0]

        return [{
            "user_id": data['user_id'],
            "input": [[query, x] for x in docs],
            "docs": docs,
            "scores": torch.tensor(data['scores']),
            "masks": torch.tensor(data['masks'])
        }]
