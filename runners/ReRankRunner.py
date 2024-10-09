import copy
import json
import os
import pickle

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from models.reranker import ReRankerModel
from prompts.pre_process import load_get_corpus_fn


class ReRanker:

    @staticmethod
    def parse_args(parser):
        parser.add_argument(
            "--rerank_type",
            default="cross_tune",
            choices=['direct', 'bm25', 'cross', 'cross_tune', 'optim'])

        parser.add_argument("--base_reranker_path",
                            default="LLMs/bge-reranker-base")
        parser.add_argument("--reranker_checkpoint",
                            default="bge-reranker-base/20241009-123120")

        parser.add_argument("--input_source", default="bge-base-en-v1.5_5")
        parser.add_argument("--input_file",
                            default="20241009-122157_user-6_20241009-120906")

        parser.add_argument("--model_name",
                            default="Meta-Llama-3-8B-Instruct",
                            choices=[
                                'Llama-2-7b-chat-hf',
                                'Meta-Llama-3-8B-Instruct', 'Qwen1.5-7B-Chat'
                            ])

        parser.add_argument("--user_emb_path", default="20241009-120906.pt")
        parser.add_argument("--user_vocab_path", default="")

        return parser

    def __init__(self, opts):
        self.task = opts.task
        self.get_corpus = load_get_corpus_fn(self.task)
        self.use_date = opts.source.endswith('date')

        self.topk = opts.topk
        self.device = opts.device
        self.batch_size = opts.batch_size
        self.rerank_type = opts.rerank_type
        self.data_split = opts.data_split

        self.load_user(opts)

        if self.rerank_type == 'cross' or self.rerank_type == 'cross_tune':
            if self.rerank_type == 'cross':
                self.reranker_checkpoint = opts.base_reranker_path
                self.result_name = f"{self.reranker_checkpoint.split('/')[-1]}_rerank_{self.topk}"
            elif self.rerank_type == 'cross_tune':
                self.reranker_checkpoint = os.path.join(
                    opts.output_addr, f"train/{opts.source}",
                    opts.reranker_checkpoint)
                self.result_name = f"{'/'.join(self.reranker_checkpoint.split('/')[-2:])}_rerank_{self.topk}"

            self.reranker = ReRankerModel(
                rerank_type=self.rerank_type,
                model_path=self.reranker_checkpoint,
                base_model_path=opts.base_reranker_path,
                user2id=self.user2id,
                user_emb_path=self.user_emb_path,
                batch_size=self.batch_size,
                device=self.device,
                max_length=opts.max_length,
            ).eval().to(self.device)

        self.input_file = opts.input_file
        self.output_addr = opts.output_addr
        self.source = opts.source
        self.input_source = opts.input_source
        file_path = os.path.join(self.output_addr, opts.data_split,
                                 self.source, self.input_source, 'retrieval',
                                 f"{self.input_file}.json")

        self.dataset = json.load(open(file_path, 'r'))
        print("orig datasize:{}".format(len(self.dataset)))
        self.dataset = self.dataset[opts.begin_idx:opts.end_idx]

    def load_user(self, opts):
        opts.user_vocab_path = os.path.join(opts.data_addr,
                                            f"dev/{opts.source}")
        vocab_addr = opts.user_vocab_path

        opts.user_emb_path = os.path.join(opts.data_addr,
                                          f"dev/{opts.source}/user_emb",
                                          opts.user_emb_path)
        self.user_emb_path = opts.user_emb_path
        self.user_emb_name = '.'.join(
            os.path.basename(self.user_emb_path).split('.')[:-1])
        self.user_embedding = torch.load(self.user_emb_path).to(self.device)

        with open(os.path.join(vocab_addr, 'user_vocab.pkl'), 'rb') as file:
            self.user_vocab = pickle.load(file)

        with open(os.path.join(vocab_addr, 'user2id.pkl'), 'rb') as file:
            self.user2id = pickle.load(file)

        assert self.user_embedding.shape[0] == len(self.user_vocab)
        assert len(self.user_vocab) == len(self.user2id)

    def run(self):
        results = []
        for idx, data in enumerate(tqdm(self.dataset)):
            rerank_profs, rerank_scores = self.rerank_topk(
                data['query'], data['retrieval'], data['user_id'])

            new_cur_reranked = []
            for prof_idx, profile in enumerate(rerank_profs):
                cur_profile = copy.deepcopy(profile)
                cur_profile['rerank_score'] = rerank_scores[prof_idx]
                new_cur_reranked.append(cur_profile)

            results.append({
                "input": data['input'],
                "query": data['query'],
                "output": data['output'],
                "user_id": data['user_id'],
                "retrieval": new_cur_reranked
            })

        if self.rerank_type == 'bm25' or self.rerank_type == 'direct':
            output_addr = os.path.join(
                self.output_addr, self.data_split, self.source,
                self.input_source, f"{self.rerank_type}_rerank_{self.topk}")
        elif self.rerank_type == 'cross' or self.rerank_type == 'cross_tune':
            output_addr = os.path.join(self.output_addr, self.data_split,
                                       self.source, self.input_source,
                                       self.result_name)
        elif self.rerank_type == 'optim':
            output_addr = os.path.join(
                self.output_addr, self.data_split, self.source,
                self.input_source,
                f"optim_rerank_{self.topk}_{self.optim_metric}")
        if not os.path.exists(output_addr):
            os.makedirs(output_addr)

        result_path = os.path.join(output_addr, f"{self.input_file}.json")

        print("save file to: {}".format(result_path))
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    @torch.no_grad()
    def rerank_topk(self, query, profile, user):
        corpus = self.get_corpus(profile, self.use_date)

        if self.rerank_type == 'direct':
            scores = [x['score'] for x in profile]
            top_n = np.argsort(scores)[::-1][:self.topk]
            top_n_scores = [scores[i] for i in top_n]
            selected_profs = [profile[i] for i in top_n]
        elif self.rerank_type == 'bm25':
            bm25 = BM25Okapi([x.split() for x in corpus])
            scores = bm25.get_scores(query.split())
            top_n = np.argsort(scores)[::-1][:self.topk]
            top_n_scores = [scores[i] for i in top_n]
            selected_profs = [profile[i] for i in top_n]
        elif self.rerank_type == 'cross' or self.rerank_type == 'cross_tune':
            selected_profs, top_n_scores = self.reranker.rerank_topk(
                corpus, profile, query, user, self.topk)

        return selected_profs, top_n_scores
