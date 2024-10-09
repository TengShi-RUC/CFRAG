from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='../../LLMs/bge-base-en-v1.5',
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })

    corpus_encode_batch_size: int = field(default=128)
    emb_pooling: str = field(default='average')
    emb_normalize: bool = field(default=True)
    emb_dim: int = field(default=768)

    infoNCE_temp: float = field(default=0.1)
    infoNCE_temp_learn: bool = field(default=False)
    sim_metric: str = field(default='cosine')  # [cosine, matmul]
    sim_map: bool = field(default=False)
    sim_activate: str = field(default=None)  # [None, tanh]

    num_layers: int = field(default=1)
    num_heads: int = field(default=2)
    dropout: float = field(default=0.1)


@dataclass
class DataArguments:
    task: str = field(default="LaMP_2_time")
    stage: str = field(default='dev')
    source: str = field(default='recency')
    base_path: str = field(default="../../")
    vocab_path: str = field(default=None)
    corpus_vocab_path: str = field(default=None)
    corpus_emb_path: str = field(default=None)
    max_samples: int = field(default=1000000)

    max_profile_len: int = field(default=100)
    max_corpus_len: int = field(default=512)

    crop_ratio: float = field(default=0.7)
    mask_ratio: float = field(default=0.3)
    reorder_ratio: float = field(default=0.3)

    freeze_emb: bool = field(default=False)


@dataclass
class TrainingArguments:
    CUDA_VISIBLE_DEVICES: str = field(default='0,1')

    seed: int = field(default=20240701)
    time: str = field(default=None)
    device: str = field(default='cuda:0')
    model_path: str = field(default=None)
    log_path: str = field(default=None)
    user_emb_path: str = field(default=None)

    num_workers: int = field(default=8)

    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "The initial learning rate for AdamW."})
    patience: int = field(
        default=3,
        metadata={
            "help":
            "Number of epochs with no improvement after which learning rate will be reduced"
        })
    min_lr: float = field(default=1e-7)

    early_stop: int = field(default=5)
    print_interval: int = field(default=10)

    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(
        default=64,
        metadata={
            "help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."
        })
    per_device_eval_batch_size: int = field(default=64)
    weight_decay: float = field(
        default=1e-5,
        metadata={"help": "Weight decay for AdamW if we apply some."})
