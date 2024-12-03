"""
Parent PPO fine-tuning agent class.

"""

from typing import Optional
import tensorflow as tf
import logging
# import tf.keras.optimizers.schedules.CosineDecayRestarts as CosineAnnealingWarmupRestarts
from util.scheduler import CosineAnnealingWarmupRestarts2

log = logging.getLogger(__name__)
from agent.finetune.train_agent import TrainAgent
from util.reward_scaling import RunningRewardScaler


class TrainPPOAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Batch size for logprobs calculations after an iteration --- prevent out of memory if using a single batch
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        assert (
            self.logprob_batch_size % self.n_envs == 0
        ), "logprob_batch_size must be divisible by n_envs"

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Wwarm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # use cosine scheduler with linear warmup
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts2(
            initial_learning_rate=0.001,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        # Optimizer
        self.actor_optimizer = tf.keras.optimizers.Adam(
            # self.model.actor_ft.parameters(),
            learning_rate=self.actor_lr_scheduler,
            decay=cfg.train.actor_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts2(
            initial_learning_rate=0.001,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            # self.model.critic.parameters(),
            learning_rate=self.critic_lr_scheduler,
            decay=cfg.train.critic_weight_decay,
        )

        # Generalized advantage estimation
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)

        # If specified, stop gradient update once KL difference reaches it
        self.target_kl: Optional[float] = cfg.train.target_kl

        # Number of times the collected data is used in gradient update
        self.update_epochs: int = cfg.train.update_epochs

        # Entropy loss coefficient
        self.ent_coef: float = cfg.train.get("ent_coef", 0)

        # Value loss coefficient
        self.vf_coef: float = cfg.train.get("vf_coef", 0)

        # Whether to use running reward scaling
        self.reward_scale_running: bool = cfg.train.reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(self.n_envs)

        # Scaling reward with constant
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1)

        # Use base policy
        self.use_bc_loss: bool = cfg.train.get("use_bc_loss", False)
        self.bc_loss_coeff: float = cfg.train.get("bc_loss_coeff", 0)

    def reset_actor_optimizer(self):
        """Not used anywhere currently"""
        new_scheduler = CosineAnnealingWarmupRestarts2(**self.actor_lr_scheduler.get_config())
        self.actor_lr_scheduler = new_scheduler
        new_optimizer = tf.keras.optimizers.Adam(
            # self.model.actor_ft.parameters(),
            learning_rate=self.actor_lr_scheduler,
            decay=self.cfg.train.actor_weight_decay,
        )
        new_optimizer.load_state_dict(self.actor_optimizer.state_dict())
        self.actor_optimizer = new_optimizer
        log.info("Reset actor optimizer")
