import tensorflow as tf
import numpy as np
import os
import logging

class TrainPPODiffusionAgent(tf.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model  # Instance of PPODiffusion
        self.reward_horizon = cfg.get("reward_horizon", self.model.horizon_steps)

        # Optimizers
        self.actor_optimizer = tf.optimizers.Adam(
            learning_rate=cfg.train.actor_lr
        )
        self.critic_optimizer = tf.optimizers.Adam(
            learning_rate=cfg.train.critic_lr
        )

        if self.model.learn_eta:
            self.eta_optimizer = tf.optimizers.Adam(
                learning_rate=cfg.train.eta_lr
            )
            self.eta_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=cfg.train.eta_lr,
                decay_steps=cfg.train.eta_decay_steps,
                alpha=cfg.train.eta_lr_decay_alpha
            )
            
        # Logging
        self.log = logging.getLogger(__name__)

    def run(self):
        run_results = []
        itr = 0
        total_steps = 0

        while itr < self.cfg.train.num_iterations:
            eval_mode = (itr % self.cfg.train.eval_frequency == 0)
            self.model.training = not eval_mode

            # Reset environment and collect trajectories
            obs_trajs, reward_trajs, actions = self.collect_trajectories(eval_mode)

            # Evaluate reward performance
            avg_reward = np.mean(reward_trajs)
            self.log.info(f"Iteration {itr}: Avg reward {avg_reward}")

            if not eval_mode:
                # Compute PPO Losses and apply gradients
                self.train_step(obs_trajs, reward_trajs, actions)

            # Save model periodically
            if itr % self.cfg.train.save_frequency == 0:
                self.save_model()

            # Update counters
            itr += 1
            total_steps += len(obs_trajs)

    def collect_trajectories(self, eval_mode):
        obs_trajs = []
        reward_trajs = []
        actions = []

        for _ in range(self.cfg.env.num_steps):
            # Reset and collect experience for a batch of episodes
            obs = self.reset_env()
            done = False

            while not done:
                # Sample actions
                action = self.model(obs, deterministic=eval_mode)

                # Interact with environment
                next_obs, reward, done, _ = self.env.step(action)

                # Store results
                obs_trajs.append(obs)
                reward_trajs.append(reward)
                actions.append(action)

                # Update obs
                obs = next_obs

        return obs_trajs, reward_trajs, actions

    @tf.function
    def train_step(self, obs_trajs, reward_trajs, actions):
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(obs_trajs, reward_trajs)

        # Perform policy and critic update
        with tf.GradientTape() as tape:
            # Compute PPO loss and regularization terms
            pg_loss, v_loss, entropy = self.model.loss(
                obs_trajs, actions, returns, advantages
            )
            total_loss = pg_loss + v_loss * self.cfg.train.value_loss_coef + entropy * self.cfg.train.entropy_coef

        # Apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def save_model(self):
        # Save model and checkpoints
        pass

    def reset_env(self):
        # Reset environment to get initial observation
        pass

    def compute_advantages(self, obs_trajs, reward_trajs):
        # Use GAE or Monte Carlo returns to compute advantages
        pass
