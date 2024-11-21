import logging
import os
import random

import hydra
import numpy as np
import tensorflow as tf
import wandb
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# Device setting
DEVICE = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"


def to_device(x, device=DEVICE):
    if tf.is_tensor(x):
        with tf.device(device):
            return tf.identity(x)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="/gpu:0"):
    """Convert each field of the batch to TensorFlow tensor on the appropriate device"""
    return {
        k: to_device(v, device=device) for k, v in batch.items()
    }


def stitched_sequence_generator(dataset):
    for idx in range(len(dataset)):
        yield dataset[idx]
        # sp = dataset[idx]
        # yield {
        #     "actions": sp.actions,
        #     "conditions": {
        #         "state": sp.conditions["state"]
        #     }
        # }


class EMA(tf.Module):
    """Exponential Moving Average (EMA) implementation in TensorFlow."""

    def __init__(self, decay):
        super().__init__()
        self.decay = decay

    def update_model_average(self, ma_model, current_model):
        for ma_var, cur_var in zip(ma_model.trainable_variables, current_model.trainable_variables):
            ma_var.assign(self.update_average(ma_var, cur_var))

    def update_average(self, old, new):
        return old * self.decay + (1 - self.decay) * new


class PreTrainAgent(tf.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # WandB initialization
        self.use_wandb = cfg.wandb is not None
        if self.use_wandb:
            wandb.login(key="8dfecd7a3ef1cc5bdea4d0a0d55fdfac3a136629")
            wandb.init(
                # entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Build model and EMA
        self.model = hydra.utils.instantiate(cfg.model)
        self.ema = EMA(cfg.ema.decay)
        # self.ema_model = tf.keras.models.clone_model(self.model)
        self.ema_model = hydra.utils.instantiate(cfg.model)
        # self.ema_model.set_weights(self.model.get_weights())

        # Training parameters
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.update_ema_freq = cfg.train.update_ema_freq
        self.epoch_start_ema = cfg.train.epoch_start_ema
        self.val_freq = cfg.train.get("val_freq", 100)

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq

        # Dataset and dataloader
        stitched_sequence_dataset = hydra.utils.instantiate(cfg.train_dataset)
        self.dataset_train = tf.data.Dataset.from_generator(
            lambda: stitched_sequence_generator(stitched_sequence_dataset),
            output_signature=stitched_sequence_dataset.element_spec()
        )
        self.dataloader_train = self.dataset_train.batch(self.batch_size)
        self.dataloader_val = None

        # Split dataset for validation
        if "train_split" in cfg.train and cfg.train.train_split < 1:
            train_size = int(cfg.train.train_split * len(self.dataset_train))
            val_size = len(self.dataset_train) - train_size
            self.dataset_train, self.dataset_val = tf.keras.utils.split_dataset(self.dataset_train,
                                                                                [train_size, val_size])
            self.dataloader_val = tf.data.Dataset.from_tensor_slices(self.dataset_val)\
                                  .batch(self.batch_size).prefetch(2).cache("cache")

        # Optimizer and learning rate scheduler
        self.lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=cfg.train.learning_rate,
            first_decay_steps=cfg.train.lr_scheduler.first_cycle_steps,
            t_mul=1.0,
            m_mul=1.0,
            alpha=cfg.train.lr_scheduler.min_lr / cfg.train.learning_rate
        )
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.lr_scheduler,
            weight_decay=cfg.train.weight_decay
        )

        self.reset_parameters()

    def run(self):
        raise NotImplementedError("Define a run method for the specific training loop.")

    def reset_parameters(self):
        """Copies current model parameters to the EMA model."""
        self.ema_model.set_weights(self.model.get_weights())

    def step_ema(self):
        """Updates the EMA model parameters."""
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_model(self, epoch):
        """Saves the model and EMA model to disk."""
        save_path = os.path.join(self.checkpoint_dir, f"state_{epoch}.h5")
        self.model.save_weights(save_path)
        self.ema_model.save_weights(save_path.replace("state_", "ema_state_"))
        log.info(f"Saved model to {save_path}")

    def load_model(self, epoch):
        """Loads the model and EMA model from disk."""
        load_path = os.path.join(self.checkpoint_dir, f"state_{epoch}.h5")
        self.model.load_weights(load_path)
        self.ema_model.load_weights(load_path.replace("state_", "ema_state_"))
        log.info(f"Loaded model from {load_path}")

    def train_epoch(self):
        for step, batch in enumerate(self.dataloader_train):
            batch = to_device(batch)
            with tf.GradientTape() as tape:
                # Forward pass and loss computation
                predictions = self.model(batch["inputs"], training=True)
                loss = self.compute_loss(predictions, batch["targets"])

            # Backpropagation
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Update EMA model periodically
            if step % self.update_ema_freq == 0:
                self.step_ema()

            # Logging and wandb integration
            if self.use_wandb and step % self.log_freq == 0:
                wandb.log({"loss": loss.numpy(), "lr": self.lr_scheduler(self.optimizer.iterations).numpy()})
                log.info(f"Step {step}, Loss: {loss.numpy()}")

    def compute_loss(self, predictions, targets):
        """Computes the training loss (override for custom loss)."""
        return tf.keras.losses.MeanSquaredError()(targets, predictions)
