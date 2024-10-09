import datetime
import logging
import os

import utils
from arguments import DataArguments, ModelArguments, TrainingArguments
from trainer import Trainer
from transformers import HfArgumentParser


def log_args(args):
    for flag, value in sorted(args.__dict__.items(), key=lambda x: x[0]):
        logging.info('{}: {} {}'.format(flag, value, type(value)))
    logging.info("")


if __name__ == "__main__":
    global_start_time = datetime.datetime.now()

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    train_args: TrainingArguments

    os.environ['CUDA_VISIBLE_DEVICES'] = train_args.CUDA_VISIBLE_DEVICES
    data_args.vocab_path = os.path.join(data_args.base_path, "data",
                                        data_args.task, data_args.stage,
                                        data_args.source)
    if train_args.time is None:
        cur_time = datetime.datetime.now()
        train_args.time = cur_time.strftime(r"%Y%m%d-%H%M%S")

    out_path = os.path.join(data_args.base_path, "data", data_args.task,
                            data_args.stage, data_args.source, "user_emb")

    data_args.corpus_emb_path = os.path.join(
        data_args.base_path, "data", data_args.task, data_args.stage,
        data_args.source, "corpus_emb",
        f"{model_args.model_name_or_path.split('/')[-1]}_mean.pt")

    train_args.log_path = os.path.join(out_path, f"log/{train_args.time}.log")
    train_args.model_path = os.path.join(out_path,
                                         f"checkpoints/{train_args.time}.pt")
    train_args.user_emb_path = os.path.join(out_path, f"{train_args.time}.pt")
    utils.setup_seed(train_args.seed)
    utils.set_logging(train_args)

    logging.info("Training/evaluation parameters")
    log_args(train_args)
    logging.info("Model parameters")
    log_args(model_args)
    logging.info("Data parameters")
    log_args(data_args)

    trainer = Trainer(model_args=model_args,
                      data_args=data_args,
                      train_args=train_args)
    trainer.train()

    global_end_time = datetime.datetime.now()
    print("runnning used time:{}".format(global_end_time - global_start_time))
