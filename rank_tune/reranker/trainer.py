import json
import logging
import os
from typing import Optional

import torch
from modeling import CrossEncoder
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


class CETrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if self.is_world_process_zero():
            config = {
                "use_user": self.args.use_user,
                "user_emb_path": os.path.abspath(self.args.user_emb_path),
                "freeze_user_emb": self.args.freeze_user_emb
            }
            config_dir = os.path.join(output_dir, "user_config")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            with open(os.path.join(config_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=4)

    def compute_loss(self, model: CrossEncoder, inputs):
        return model(inputs)['loss']
