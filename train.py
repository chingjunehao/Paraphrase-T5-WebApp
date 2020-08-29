import argparse
import logging
import os

import random
import numpy as np
import torch

import pytorch_lightning as pl

from T5FineTuner import T5FineTuner
from ParaphraseDataLoader import ParaphraseDataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)



logger = logging.getLogger(__name__)
class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

args_dict = dict(
    train_path = "paraphrase_data/paws_quora_train_data.txt",
    val_path = "paraphrase_data/paws_quora_val_data.txt",
    output_dir="t5_paraphrase_pq", 
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=6,
    eval_batch_size=6,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1', 
    max_grad_norm=1.0, 
    seed=42,
)



args = argparse.Namespace(**args_dict)
print(args_dict)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)


print(torch.cuda.device_count())



print ("Initialize model")
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
print ("Training model")
trainer.fit(model)
print ("Training finished")


print ("Saving model")
model.model.save_pretrained(args.output_dir)
print ("Saved model")