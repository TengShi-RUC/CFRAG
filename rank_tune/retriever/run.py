import logging
import os
from pathlib import Path
import datetime

from arguments import DataArguments, ModelArguments
from arguments import RetrieverTrainingArguments as TrainingArguments
from modeling import BiEncoderModel
from trainer import BiTrainer
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

from data import TrainDataCollator, TrainDataset

logger = logging.getLogger(__name__)


def log_args(args):
    for flag, value in args.__dict__.items():
        logger.info('{}: {} {}'.format(flag, value, type(value)))
    logger.info("")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    cur_time = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        model_args.model_name_or_path.split('/')[-1], cur_time)

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info("Training/evaluation parameters")
    log_args(training_args)
    logger.info("Model parameters")
    log_args(model_args)
    logger.info("Data parameters")
    log_args(data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    model = BiEncoderModel(model_args=model_args,
                           data_args=data_args,
                           train_args=training_args)

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDataset(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        data_collator=TrainDataCollator(args=data_args,
                                                        tokenizer=tokenizer),
                        tokenizer=tokenizer)

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
