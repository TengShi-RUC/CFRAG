import datetime
import logging
import os
import pickle
import random
from typing import Any, Dict, List, NoReturn

import numpy as np
import torch
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


GLOBAL_SEED = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


GLOBAL_WORKER_ID = None


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def batch_to_gpu(batch: dict, device) -> dict:
    for c in batch:
        if isinstance(batch[c], torch.Tensor):
            batch[c] = batch[c].to(device)
        elif isinstance(batch[c], List):
            batch[c] = [[p.to(device)
                         for p in k] if isinstance(k, List) else k.to(device)
                        for k in batch[c]]

    return batch


def check_dir(file_name: str):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_logging(args):
    log_path = args.log_path
    check_dir(log_path)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO,
                        filename=log_path,
                        filemode='w')
