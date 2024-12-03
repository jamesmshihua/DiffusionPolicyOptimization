from model.diffusion.diffusion_vpg_g import VPGDiffusion
import tensorflow as tf
# from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import math

class PPODiffusion(VPGDiffusion):
    def __init__(
        self,
        gamma_denoising,
        clip_ploss_coef,
        clip_ploss_coef_base=1e-3,
        clip_ploss_coef_rate=3,
        clip_vloss_coef=None,
        clip_advantage_lower_quantile=0,
        clip_advantage_upper_quantile=1,
        norm_adv=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # PPO-specific parameters
        self.gamma_denoising = gamma_denoising
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        self.norm_adv = norm_adv

    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        # New logprobs with entropy for denoising steps
        newlogprobs, eta = self.get_logprobs_subsample(
            obs, chains_prev, chains_next, denoising_inds, get_ent=True
        )
        entropy_loss = -tf.reduce_mean(eta)
        newlogprobs = tf.clip_by_value(newlogprobs, clip_value_min=-5, clip_value_max=2)
        oldlogprobs = tf.clip_by_value(oldlogprobs, clip_value_min=-5, clip_value_max=2)

        # Apply reward horizon limitation
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

        # Mean over dimensions
        newlogprobs = tf.reduce_mean(tf.reshape(newlogprobs, (-1,)))
        oldlogprobs = tf.reduce_mean(tf.reshape(oldlogprobs, (-1,)))

        # Optional behavioral cloning loss
        bc_loss = 0
        if use_bc_loss:
            samples = self.forward(
                cond=obs, deterministic=False, return_chain=True, use_base_policy=True
            )
            bc_logprobs = self.get_logprobs(
                obs, samples.chains, get_ent=False, use_base_policy=False
            )
            bc_logprobs = tf.clip_by_value(bc_logprobs, clip_value_min=-5, clip_value_max=2)
            bc_loss = -tf.reduce_mean(bc_logprobs)

        # Normalize advantages
        if self.norm_adv:
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        # Quantile clipping for advantages
        advantage_min = tfp.stats.percentile(advantages, self.clip_advantage_lower_quantile * 100)
        advantage_max = tfp.stats.percentile(advantages, self.clip_advantage_upper_quantile * 100)
        advantages = tf.clip_by_value(advantages, clip_value_min=advantage_min, clip_value_max=advantage_max)

        # Discounted denoising
        discount_factors = tf.pow(
            self.gamma_denoising, self.ft_denoising_steps - tf.cast(denoising_inds, tf.float32) - 1
        )
        advantages *= discount_factors

        # Calculate the ratio for PPO
        logratio = newlogprobs - oldlogprobs
        ratio = tf.exp(logratio)

        # Clip policy loss coefficient
        t = tf.cast(denoising_inds, tf.float32) / (self.ft_denoising_steps - 1)
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (tf.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = t

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * tf.clip_by_value(ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef)
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

        # Value loss
        newvalues = tf.squeeze(self.critic(obs), axis=-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = tf.square(newvalues - returns)
            v_clipped = oldvalues + tf.clip_by_value(
                newvalues - oldvalues, -self.clip_vloss_coef, self.clip_vloss_coef
            )
            v_loss_clipped = tf.square(v_clipped - returns)
            v_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss_unclipped, v_loss_clipped))
        else:
            v_loss = 0.5 * tf.reduce_mean(tf.square(newvalues - returns))

        # Compute additional metrics
        approx_kl = tf.reduce_mean((ratio - 1) - logratio)
        clipfrac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > clip_ploss_coef, tf.float32))
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl,
            tf.reduce_mean(ratio),
            bc_loss,
            tf.reduce_mean(eta)
        )

    def get_config(self):
        # config = dict()
        config = super().get_config()
        config["gamma_denoising"] = self.gamma_denoising
        config["clip_ploss_coef"] = self.clip_ploss_coef
        config["clip_ploss_coef_base"] = self.clip_ploss_coef_base
        config["clip_ploss_coef_rate"] = self.clip_ploss_coef_rate
        config["clip_vloss_coef"] = self.clip_vloss_coef
        config["clip_advantage_lower_quantile"] = self.clip_advantage_lower_quantile
        config["clip_advantage_upper_quantile"] = self.clip_advantage_upper_quantile
        config["norm_adv"] = self.norm_adv
        return config