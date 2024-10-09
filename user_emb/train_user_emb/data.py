import logging
import math
import os.path
import pickle
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from arguments import DataArguments


@dataclass
class TrainDataCollator:

    args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch):
        corpus_1 = torch.LongTensor([x['corpus_1'] for x in batch])
        corpus_1_mask = torch.tensor([x['corpus_1_mask'] for x in batch])
        corpus_2 = torch.LongTensor([x['corpus_2'] for x in batch])
        corpus_2_mask = torch.tensor([x['corpus_2_mask'] for x in batch])

        
        return {
            "corpus_1": corpus_1,
            "corpus_1_mask": corpus_1_mask,
            "corpus_2": corpus_2,
            "corpus_2_mask": corpus_2_mask
        }


class TrainDataset(Dataset):

    def __init__(self, args: DataArguments):

        with open(os.path.join(args.vocab_path, 'corpus2id.pkl'),
                  'rb') as file:
            self.corpus2id = pickle.load(file)
        logging.info("num corpus: {}".format(len(self.corpus2id)))
        self.pad_token = self.corpus2id['<pad>']
        logging.info("pad token: {} id: {}".format("<pad>", self.pad_token))

        if args.freeze_emb:
            self.mask_token = self.corpus2id['']
            logging.info("mask token: {} id: {}".format('', self.mask_token))
        else:
            self.mask_token = self.corpus2id['<mask>']
            logging.info("mask token: {} id: {}".format(
                '<mask>', self.mask_token))

        with open(os.path.join(args.vocab_path, 'user_vocab.pkl'),
                  'rb') as file:
            self.user_vocab = pickle.load(file)
        new_user_vocab = [
            self.user_vocab[i]
            for i in range(min(args.max_samples, len(self.user_vocab)))
        ]
        self.user_vocab = new_user_vocab

        logging.info("num users: {}".format(len(self.user_vocab)))
        profile_lens = [
            len(self.user_vocab[idx]['profile'])
            for idx in range(len(self.user_vocab))
        ]
        logging.info("profile len mean: {} max:{} min:{}".format(
            np.mean(profile_lens), np.max(profile_lens), np.min(profile_lens)))

        self.args = args


    def __len__(self):
        return len(self.user_vocab)

    def __getitem__(self, index):
        data = self.user_vocab[index]

        corpus = data['corpus_ids']
        corpus = corpus[-self.args.max_profile_len:]
        aug_corpus_1, corpus_1_mask = self.aug_seq(corpus)
        aug_corpus_2, corpus_2_mask = self.aug_seq(corpus)

        return {
            "corpus_1": aug_corpus_1,
            "corpus_1_mask": corpus_1_mask,
            "corpus_2": aug_corpus_2,
            "corpus_2_mask": corpus_2_mask
        }

    def aug_seq(self, corpus):
        seqs_len = len(corpus)
        if seqs_len > 1:
            aug_type = random.choice(range(3))
            if aug_type == 0:
                # crop
                num_left = math.floor(seqs_len * self.args.crop_ratio)
                crop_begin = random.randint(0, seqs_len - num_left)
                aug_seqs = corpus[crop_begin:crop_begin + num_left]
            elif aug_type == 1:
                # mask
                num_mask = math.floor(seqs_len * self.args.mask_ratio)
                mask_index = random.sample(range(seqs_len), k=num_mask)
                aug_seqs = []
                for i in range(seqs_len):
                    if i in mask_index:
                        aug_seqs.append(self.mask_token)
                    else:
                        aug_seqs.append(corpus[i])
            elif aug_type == 2:
                # reorder
                num_reorder = math.floor(seqs_len * self.args.reorder_ratio)
                reorder_begin = random.randint(0, seqs_len - num_reorder)
                aug_seqs = corpus[:]
                sub_seqs = corpus[reorder_begin:reorder_begin + num_reorder]
                random.shuffle(sub_seqs)
                aug_seqs[reorder_begin:reorder_begin + num_reorder] = sub_seqs
        else:
            aug_seqs = corpus
        masks = [False] * len(aug_seqs)
        if len(aug_seqs) < self.args.max_profile_len:
            aug_seqs += [self.pad_token
                         ] * (self.args.max_profile_len - len(aug_seqs))
            masks += [True] * (self.args.max_profile_len - len(masks))
        return aug_seqs, masks


@dataclass
class TestDataCollator:

    args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch):
        corpus = torch.LongTensor([x['corpus'] for x in batch])
        corpus_mask = torch.tensor([x['corpus_mask'] for x in batch])

        return {"corpus": corpus, "corpus_mask": corpus_mask}


class TestDataset(Dataset):

    def __init__(self, args: DataArguments):
        with open(os.path.join(args.vocab_path, 'corpus2id.pkl'),
                  'rb') as file:
            self.corpus2id = pickle.load(file)
        self.pad_token = self.corpus2id['<pad>']

        with open(os.path.join(args.vocab_path, 'user_vocab.pkl'),
                  'rb') as file:
            self.user_vocab = pickle.load(file)
        new_user_vocab = [
            self.user_vocab[i]
            for i in range(min(args.max_samples, len(self.user_vocab)))
        ]
        self.user_vocab = new_user_vocab
        self.args = args


    def __len__(self):
        return len(self.user_vocab)

    def __getitem__(self, index):
        data = self.user_vocab[index]

        corpus = data['corpus_ids']
        corpus = corpus[-self.args.max_profile_len:]
        masks = [False] * len(corpus)
        if len(corpus) < self.args.max_profile_len:
            corpus += [self.pad_token
                       ] * (self.args.max_profile_len - len(corpus))
            masks += [True] * (self.args.max_profile_len - len(masks))
        return {"corpus": corpus, "corpus_mask": masks}

