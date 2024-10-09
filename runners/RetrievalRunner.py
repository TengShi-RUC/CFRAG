import copy
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from models.retriever import RetrieverModel
from prompts.pre_process import load_get_corpus_fn, load_get_query_fn


class Retriever:

    @staticmethod
    def parse_args(parser):
        parser.add_argument("--ret_type",
                            default="dense_tune",
                            choices=[
                                'zero_shot', 'random', 'recency', 'bm25',
                                'dense', 'dense_tune'
                            ])

        parser.add_argument("--base_retriever_path",
                            default="LLMs/bge-base-en-v1.5")
        parser.add_argument("--retriever_checkpoint",
                            default="bge-base-en-v1.5/20241009-122157")
        parser.add_argument("--retriever_pooling", default="average")
        parser.add_argument("--retriever_normalize", type=int, default=1)

        parser.add_argument("--retrieve_user", type=int, default=1)

        parser.add_argument("--user_emb_path", default="20241009-120906.pt")
        parser.add_argument("--user_vocab_path", default="")

        parser.add_argument("--user_topk", type=int, default=6)

        return parser

    def __init__(self, opts) -> None:
        self.task = opts.task
        self.get_query = load_get_query_fn(self.task)
        self.get_corpus = load_get_corpus_fn(self.task)
        self.use_date = opts.source.endswith('date')
        self.llm_name = opts.llm_name

        self.data_addr = opts.data_addr
        self.output_addr = opts.output_addr
        self.data_split = opts.data_split
        self.source = opts.source
        self.ret_type = opts.ret_type
        self.topk = opts.topk
        self.retrieve_user = opts.retrieve_user
        self.device = opts.device

        self.load_user(opts)

        if self.ret_type == 'dense' or self.ret_type == 'dense_tune':
            self.batch_size = opts.batch_size

            if self.ret_type == 'dense':
                self.retriever_checkpoint = opts.base_retriever_path
            elif self.ret_type == 'dense_tune':
                opts.retriever_checkpoint = os.path.join(
                    opts.output_addr, f"train/{opts.source}",
                    opts.retriever_checkpoint)
                self.retriever_checkpoint = opts.retriever_checkpoint

            self.retriever = RetrieverModel(
                ret_type=self.ret_type,
                model_path=self.retriever_checkpoint,
                base_model_path=opts.base_retriever_path,
                user2id=self.user2id,
                user_emb_path=opts.user_emb_path,
                batch_size=self.batch_size,
                device=self.device,
                max_length=opts.max_length,
                pooling=opts.retriever_pooling,
                normalize=opts.retriever_normalize).eval().to(self.device)

        input_path = os.path.join(self.data_addr, opts.data_split, self.source,
                                  'rank_merge.json')

        self.dataset = json.load(open(input_path, 'r'))
        print("orig datasize:{}".format(len(self.dataset)))
        self.dataset = self.dataset[opts.begin_idx:opts.end_idx]

    def load_user(self, opts):
        opts.user_vocab_path = os.path.join(opts.data_addr,
                                            f"dev/{opts.source}")
        vocab_addr = opts.user_vocab_path

        with open(os.path.join(vocab_addr, 'user_vocab.pkl'), 'rb') as file:
            self.user_vocab = pickle.load(file)

        with open(os.path.join(vocab_addr, 'user2id.pkl'), 'rb') as file:
            self.user2id = pickle.load(file)

        assert len(self.user_vocab) == len(self.user2id)

        opts.user_emb_path = os.path.join(opts.data_addr,
                                          f"dev/{opts.source}/user_emb",
                                          opts.user_emb_path)

        self.user_emb_path = opts.user_emb_path

        if self.retrieve_user:
            self.user_emb_name = '.'.join(
                os.path.basename(self.user_emb_path).split('.')[:-1])
            self.user_embedding = torch.load(self.user_emb_path).to(
                self.device)
            self.user_topk = opts.user_topk

            assert self.user_embedding.shape[0] == len(self.user_vocab)

    def run(self):
        if self.ret_type == 'zero_shot':
            sub_dir = self.ret_type
            file_name = "base"
        else:
            if self.ret_type in ['random', 'recency', 'bm25']:
                sub_dir = f"{self.ret_type}_{self.topk}"
                file_name = "base"
            elif self.ret_type == 'dense':
                sub_dir = f"{self.retriever_checkpoint.split('/')[-1]}_{self.topk}"
                file_name = "base"
            elif self.ret_type == 'dense_tune':
                retriever_name = self.retriever_checkpoint.split('/')[-2]
                train_time = self.retriever_checkpoint.split('/')[-1]
                sub_dir = f"{retriever_name}_{self.topk}"
                file_name = f"{train_time}"

            if self.retrieve_user:
                file_name += '_user-{}_{}'.format(self.user_topk,
                                                  self.user_emb_name)

        results = []
        for data in tqdm(self.dataset):
            query, selected_profs = self.retrieve_topk(data['input'],
                                                       data['user_id'])
            results.append({
                "input": data['input'],
                "query": query,
                "output": data['output'],
                "user_id": data['user_id'],
                "retrieval": selected_profs
            })

        output_addr = os.path.join(self.output_addr, self.data_split,
                                   self.source, sub_dir, 'retrieval')

        if not os.path.exists(output_addr):
            os.makedirs(output_addr)

        result_path = os.path.join(output_addr, f"{file_name}.json")
        print("save file to: {}".format(result_path))
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    def retrieve_topk(self, inp, user):
        all_profiles = self.retrieve_user_topk(user)

        query = self.get_query(inp)
        all_retrieved = []
        for i in range(len(all_profiles)):
            cur_corpus = self.get_corpus(all_profiles[i], self.use_date)
            cur_retrieved, cur_scores = self.retrieve_topk_one_user(
                cur_corpus, query, all_profiles[i], user, self.topk)
            new_cur_retrieved = []
            for data_idx, data in enumerate(cur_retrieved):
                cur_data = copy.deepcopy(data)
                if self.task.startswith('LaMP_3'):
                    cur_data['rate'] = cur_data['score']
                cur_data['score'] = cur_scores[data_idx]
                new_cur_retrieved.append(cur_data)
            all_retrieved.extend(new_cur_retrieved)
        return query, all_retrieved

    def retrieve_topk_one_user(self, corpus, query, profile, user, topk):
        if self.ret_type == "bm25":
            bm25 = BM25Okapi([x.split() for x in corpus])
            scores = bm25.get_scores(query.split())
            top_n = np.argsort(scores)[::-1][:topk]
            top_n_scores = [scores[i] for i in top_n]
            selected_profs = [profile[i] for i in top_n]
        elif self.ret_type == "dense" or self.ret_type == 'dense_tune':
            selected_profs, top_n_scores = self.retriever.retrieve_topk_dense(
                corpus, profile, query, user, topk)
        elif self.ret_type == "random":
            selected_profs = random.choices(profile, k=topk)
            top_n_scores = [1.0] * topk
        elif self.ret_type == "recency":
            profile = sorted(
                profile,
                key=lambda x: tuple(map(int,
                                        str(x['date']).split("-"))))
            randked_profile = profile[::-1]
            selected_profs = randked_profile[:topk]
            top_n_scores = [1.0] * topk
        elif self.ret_type == 'zero_shot':
            selected_profs = []
            top_n_scores = []

        return selected_profs, top_n_scores

    def retrieve_user_topk(self, user):
        user_id = self.user2id[user]
        if self.retrieve_user:
            cur_user_emb = self.user_embedding[[user_id]]
            sims = F.cosine_similarity(cur_user_emb, self.user_embedding)
            topk_values, topk_indices = torch.topk(sims, self.user_topk * 2)

            top_k_user_id = topk_indices.tolist()[:self.user_topk]
            topk_scores = [sims[i].item() for i in top_k_user_id]
        else:
            topk_scores = [1]
            top_k_user_id = [user_id]

        topk_profile = []
        for idx, user_idx in enumerate(top_k_user_id):
            cur_profile = self.user_vocab[user_idx]['profile']
            new_profile = []
            for data in cur_profile:
                new_data = copy.deepcopy(data)
                new_data['user_sim'] = topk_scores[idx]
                new_profile.append(new_data)
            topk_profile.append(new_profile)

        return topk_profile
