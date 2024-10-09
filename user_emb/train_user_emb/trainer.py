import gc
import logging
import time
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import utils
from arguments import DataArguments, ModelArguments, TrainingArguments
from data import *
from modeling import UserEncoder


class Trainer:

    def __init__(self, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path)
        self.model = UserEncoder(model_args=model_args,
                                 data_args=data_args,
                                 train_args=train_args)

        num_train = 0
        num_total = 0
        for p in self.model.parameters():
            num_total += p.numel()
            if p.requires_grad:
                num_train += p.numel()
        logging.info('Number of total parameters: %d', num_total)
        logging.info('Number of trainable parameters: %d', num_train)

        self.build_optimizer()

        self.train_loader = self.getDataLoader(
            TrainDataset(data_args),
            TrainDataCollator(data_args, self.tokenizer),
            train_args.per_device_train_batch_size,
            shuffle=True)

        self.test_loader = self.getDataLoader(
            TestDataset(data_args),
            TestDataCollator(data_args, self.tokenizer),
            train_args.per_device_eval_batch_size,
            shuffle=False)

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_args.learning_rate,
            weight_decay=self.train_args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=self.train_args.patience,
            min_lr=self.train_args.min_lr,
            verbose=True)

    def getDataLoader(self, dataset, collator, batch_size: int,
                      shuffle: bool) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=self.train_args.num_workers,
            pin_memory=True,
            prefetch_factor=batch_size // self.train_args.num_workers + 1,
            worker_init_fn=utils.worker_init_fn,
            persistent_workers=True,
            collate_fn=collator)
        return dataloader

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) - criterion.index(
                max(criterion)) > self.train_args.early_stop:
            return True
        return False

    def train(self):
        loss_result = []
        last_lr = self.train_args.learning_rate
        for epoch in range(self.train_args.num_train_epochs):
            gc.collect()
            torch.cuda.empty_cache()

            epoch_loss = self.train_epoch(epoch)
            logging.info("epoch:{} mean loss:{:.4f}".format(epoch, epoch_loss))

            loss_result.append(-epoch_loss)
            self.scheduler.step(-epoch_loss)
            new_lr = self.scheduler.get_last_lr()[0]
            if last_lr != new_lr:
                logging.info("reducing lr from:{} to:{}".format(
                    last_lr, new_lr))
                last_lr = new_lr

            if max(loss_result) == loss_result[-1]:
                self.model.save_model()
                self.get_user_emb()

            if self.train_args.early_stop > 0 and self.eval_termination(
                    loss_result):
                logging.info('Early stop at %d based on dev result.' %
                             (epoch + 1))
                break

    def train_epoch(self, epoch):
        self.model.train()
        logging.info(" ")
        logging.info("Epoch: {}".format(epoch))
        print("\nEpoch: {}".format(epoch))

        loss_list = []
        loss_dict = {}
        start = time.time()
        for step, batch in enumerate(tqdm(self.train_loader)):
            loss = self.model.loss(**batch)

            total_loss = loss['total_loss']

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            for k, v in loss.items():
                if k in loss_dict.keys():
                    loss_dict[k].append(v.item())
                else:
                    loss_dict[k] = [v.item()]

            loss_list.append(total_loss.item())

            if step > 0 and step % self.train_args.print_interval == 0:
                logging.info("epoch:{:d} step:{:d} time:{:.2f}s {}".format(
                    epoch, step,
                    time.time() - start, " ".join([
                        "{}:{:.4f}".format(k,
                                           np.mean(v).item())
                        for k, v in loss_dict.items()
                    ])))
        logging.info("total time: {:.2f}s".format(time.time() - start))
        return np.mean(loss_list).item()

    @torch.no_grad()
    def get_user_emb(self):
        self.model.eval()

        user_emb = None
        start = time.time()
        for step, batch in enumerate(self.test_loader):
            batch_user_emb = self.model.get_user_emb(**batch).detach().cpu()
            user_emb = batch_user_emb if user_emb is None else torch.cat(
                [user_emb, batch_user_emb], dim=0)

        logging.info("get user embedding time used:{}s".format(time.time() -
                                                               start))
        torch.save(user_emb, self.train_args.user_emb_path)
