"""
Pre-training diffusion policy

"""

import logging
import wandb
import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        timer = Timer()
        self.epoch = 1
        
        for _ in range(self.n_epochs):
            # train
            loss_train_epoch = []
            n_batch = 0
            for batch_train in self.dataloader_train:
                # if self.dataset_train.device == "cpu":
                batch_train = batch_to_device(batch_train)

                with tf.GradientTape() as tape:
                    loss_train = self.model.loss(**batch_train)
                    
                gradients = tape.gradient(loss_train, self.model.network.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.network.trainable_variables)
                )
                loss_train_epoch.append(loss_train.numpy())
                n_batch += 1
                log.info(f"Epoch {self.epoch}, Batch {n_batch}")
                
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val, infos_val = self.model.loss(**batch_val)
                    loss_val_epoch.append(loss_val.numpy())

            loss_val = np.mean(loss_val_epoch) if loss_val_epoch else None

            # update ema
            if self.epoch % self.update_ema_freq == 0:
                self.step_ema()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1
