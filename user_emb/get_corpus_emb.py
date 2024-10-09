import sys

sys.path.append('.')

import argparse
import os
import pickle

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from models.emb_model import EmbModel


@torch.no_grad()
def get_emb(emb_model, tokenizer, batch_size, device, corpus, max_length):
    batched_corpus = [
        corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)
    ]
    all_embs = None
    for batch in tqdm(batched_corpus):
        tokens_batch = tokenizer(batch,
                                 padding=True,
                                 truncation=True,
                                 max_length=max_length,
                                 return_tensors='pt').to(device)
        batch_emb = emb_model(**tokens_batch).cpu()
        all_embs = batch_emb if all_embs is None else torch.cat(
            (all_embs, batch_emb), dim=0)
    return all_embs


def get_corpus_emb(corpus_vocab, emb_model: EmbModel, tokenizer, batch_size,
                   device, max_length):
    # skip 0 for pad, using <mask> init mask
    corpus_list = [
        corpus_vocab[i]['corpus'] for i in range(1, len(corpus_vocab))
    ]
    corpus_emb = get_emb(emb_model, tokenizer, batch_size, device, corpus_list,
                         max_length)
    return torch.cat([torch.zeros(1, corpus_emb.shape[1]), corpus_emb], dim=0)


parser = argparse.ArgumentParser()

parser.add_argument("--CUDA_VISIBLE_DEVICES", default='0,1')
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--emb_model_path", default='LLMs/bge-base-en-v1.5')
parser.add_argument("--emb_model_pooling", default='average')
parser.add_argument("--emb_type", default='mean')
parser.add_argument("--emb_model_normalize", type=int, default=1)
parser.add_argument("--max_length", type=int, default=512)

parser.add_argument("--task", default='LaMP_2_time')
parser.add_argument("--source", default='recency')
parser.add_argument("--stage", default='dev', choices=['dev'])

if __name__ == "__main__":
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES
    opts.input_path = os.path.join("data", opts.task, opts.stage, opts.source)

    opts.corpus_emb_path = os.path.join(
        "data", opts.task, opts.stage, opts.source, "corpus_emb",
        f"{opts.emb_model_path.split('/')[-1]}_{opts.emb_type}.pt")
    for flag, value in opts.__dict__.items():
        print('{}: {}'.format(flag, value))

    emb_model = EmbModel(opts.emb_model_path, opts.emb_model_pooling,
                         opts.emb_model_normalize).eval().to(opts.device)
    emb_tokenizer = AutoTokenizer.from_pretrained(opts.emb_model_path)

    with open(os.path.join(opts.input_path, "corpus_vocab.pkl"), 'rb') as file:
        corpus_vocab = pickle.load(file)
    print("corpus size: {}".format(len(corpus_vocab)))
    print("get corpus embedding")
    corpus_emb = get_corpus_emb(corpus_vocab, emb_model, emb_tokenizer,
                                opts.batch_size, opts.device, opts.max_length)

    with open(os.path.join(opts.input_path, "user_vocab.pkl"), 'rb') as file:
        user_vocab = pickle.load(file)

    print("corpus emb: {}".format(corpus_emb.shape))
    if not os.path.exists(os.path.dirname(opts.corpus_emb_path)):
        os.makedirs(os.path.dirname(opts.corpus_emb_path))
    torch.save(corpus_emb, opts.corpus_emb_path)
