import json

from transformers.trainer import *


class BiTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(f'MODEL {self.model.__class__.__name__} '
                                      f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            config = {
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": False,
                "use_user": self.args.use_user,
                "persona_weight": self.args.persona_weight,
                "user_emb_path": os.path.abspath(self.args.user_emb_path),
                "freeze_user_emb": self.args.freeze_user_emb
            }
            if self.args.sentence_pooling_method == 'cls':
                config["pooling_mode_cls_token"] = True
            else:
                config["pooling_mode_mean_tokens"] = True
            config_dir = os.path.join(output_dir, "1_Pooling")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            with open(os.path.join(config_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=4)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
