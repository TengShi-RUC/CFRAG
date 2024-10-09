import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='../../LLMs/bge-base-en-v1.5',
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from s3"
        })

    temperature: float = field(default=1)
    teacher_temperature: float = field(default=1)


@dataclass
class DataArguments:
    train_data: str = field(
        default=
        "../../Meta-Llama-3-8B-Instruct_outputs/LaMP_2_time/train/recency/bge-base-en-v1.5_5/retrieval/point_base_user-6_20241009-120906_vllm_new-64/predictions_0-100.json",
        metadata={"help": "Path to corpus"})

    user_vocab_path: str = field(default="../../data/LaMP_2_time/dev/recency")

    query_max_len: int = field(
        default=512,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=512,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    main_metric: str = field(default="accuracy",
                             metadata={"help": "main metric"})
    num_profile: int = field(default=30)

    query_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for query"})
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"})

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(
                f"cannot find file: {self.train_data}, please set a true path")


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    fp16: bool = field(
        default=True,
        metadata={
            "help": "Whether to use fp16 (mixed) precision instead of 32-bit"
        },
    )

    freeze_user_emb: bool = field(default=True)
    user_emb_path: str = field(
        default="../../data/LaMP_2_time/dev/recency/user_emb/20241009-120906.pt"
    )
    use_user: bool = field(default=True)
    persona_weight: float = field(default=0.1)

    output_dir: str = field(
        default=
        '../../Meta-Llama-3-8B-Instruct_outputs/LaMP_2_time/train/recency/',
        metadata={
            "help":
            "The output directory where the model predictions and checkpoints will be written."
        })
    learning_rate: float = field(
        default=6e-5,
        metadata={"help": "The initial learning rate for AdamW."})
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."
        })
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help":
            "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    dataloader_drop_last: bool = field(
        default=False,
        metadata={
            "help":
            "Drop the last incomplete batch if it is not divisible by the batch size."
        })
    weight_decay: float = field(
        default=1e-5,
        metadata={"help": "Weight decay for AdamW if we apply some."})
    logging_steps: float = field(
        default=10,
        metadata={
            "help":
            ("Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
             "If smaller than 1, will be interpreted as ratio of total training steps."
             )
        },
    )

    negatives_cross_device: bool = field(
        default=False, metadata={"help": "share negatives across devices"})
    fix_position_embedding: bool = field(
        default=False,
        metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(
        default='mean',
        metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(
        default=False,
        metadata={"help": "use passages in the same batch as negatives"})
