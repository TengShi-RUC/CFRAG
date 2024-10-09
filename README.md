# CFRAG
This is the official implementation of the paper "Retrieval Augmented Generation with Collaborative Filtering for Personalized Text Generation" based on PyTorch.

# Quick Start

## 0. Download Data
Download [LaMP](https://lamp-benchmark.github.io/download) data.


## 1. User Embedding Training

```bash
cd ~
python data/preprocess_profile.py --data_phase train
python data/preprocess_profile.py --data_phase dev
python user_emb/get_user_set.py
python user_emb/get_corpus_emb.py

cd user_emb/train_user_emb
python run.py
```


## 2. Retriever Training

### Get Retriever Training Data

```bash
cd ~
python ranking.py --rank_stage retrieval --data_split train --ret_type dense --base_retriever_path base_retriever_path --user_emb_path user_emb_path
python generation/generate_point.py --source retrieval --file_name file_name 
```

### Retriever Training

```bash
cd rank_tune/retriever
python run.py 
```

## 3. ReRanker Training

### Get ReRanker Training Data

```bash
python ranking.py --rank_stage retrieval --data_split train --ret_type dense_tune --retriever_checkpoint retriever_checkpoint --user_emb_path user_emb_path
python generation/generate_point.py --source retrieval --file_name file_name 
``` 

### ReRanker Training

```bash
cd rank_tune/reranker
python run.py
```


## 4. Get Test Result

```bash
cd ~
python ranking.py --rank_stage retrieval --data_split dev --ret_type dense_tune --retriever_checkpoint retriever_checkpoint --user_emb_path user_emb_path
python ranking.py --rank_stage rerank --data_split dev --rerank_type cross_tune --reranker_checkpoint reranker_checkpoint --user_emb_path user_emb_path
python generation/generate.py --source source --file_name file_name 
```
