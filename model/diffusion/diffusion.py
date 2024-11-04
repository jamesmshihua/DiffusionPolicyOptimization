import tensorflow as tf
# import tensorflow_probability as tfp
import logging
import numpy as np
from collections import namedtuple

log = logging.getLogger(__name__)

from model.diffusion.sampling import (
    extract,
    cosine_beta_schedule,
    make_timesteps,
)

Sample = namedtuple("Sample", "trajectories chains")


class DiffusionModel(tf.Module):
    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="/gpu:0",
        # Clipping values
        denoised_clip_value=1.0,
        randn_clip_value=10.0,
        final_action_clip_value=None,
        eps_clip_value=None,
        # DDPM parameters
        denoising_steps=100,
        predict_epsilon=True,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Clipping values
        self.denoised_clip_value = denoised_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.randn_clip_value = randn_clip_value
        self.eps_clip_value = eps_clip_value

        # Set up models
        self.network = network
        if network_path is not None:
            self.network.load_weights(network_path)
            log.info(f"Loaded model from {network_path}")
        log.info(
            f"Number of network parameters: {np.sum([np.prod(v.shape) for v in self.network.trainable_variables])}"
        )

        # DDPM parameters
        self.betas = cosine_beta_schedule(denoising_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = tf.concat(
            [tf.constant([1.0], dtype=tf.float32), self.alphas_cumprod[:-1]], axis=0
        )
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod - 1)
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = tf.math.log(tf.clip_by_value(self.ddpm_var, 1e-20, float("inf")))
        self.ddpm_mu_coef1 = self.betas * tf.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_mu_coef2 = (1.0 - self.alphas_cumprod_prev) * tf.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        # DDIM parameters
        if use_ddim:
            assert predict_epsilon, "DDIM requires predicting epsilon."
            if ddim_discretize == "uniform":
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = tf.range(0, ddim_steps) * step_ratio
            else:
                raise ValueError("Unknown discretization method for DDIM.")
            self.ddim_alphas = tf.gather(self.alphas_cumprod, self.ddim_t)
            self.ddim_alphas_sqrt = tf.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = tf.concat(
                [tf.constant([1.0], dtype=tf.float32), self.alphas_cumprod[:-1]],
                axis=0,
            )
            self.ddim_sqrt_one_minus_alphas = tf.sqrt(1.0 - self.ddim_alphas)

            # Initialize fixed sigmas for inference
            ddim_eta = 0
            self.ddim_sigmas = ddim_eta * tf.sqrt(
                (1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas)
                * (1 - self.ddim_alphas / self.ddim_alphas_prev)
            )

    def p_mean_var(self, x, t, cond, index=None, network_override=None):
        noise = network_override(x, t, cond=cond) if network_override else self.network(x, t, cond=cond)

        if self.predict_epsilon:
            if self.use_ddim:
                alpha = extract(self.ddim_alphas, index, tf.shape(x))
                alpha_prev = extract(self.ddim_alphas_prev, index, tf.shape(x))
                sqrt_one_minus_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, tf.shape(x))
                x_recon = (x - sqrt_one_minus_alpha * noise) / tf.sqrt(alpha)
            else:
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, tf.shape(x)) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, tf.shape(x)) * noise
                )
        else:
            x_recon = noise

        if self.denoised_clip_value is not None:
            x_recon = tf.clip_by_value(x_recon, -self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                noise = (x - tf.sqrt(alpha) * x_recon) / sqrt_one_minus_alpha

        if self.use_ddim and self.eps_clip_value is not None:
            noise = tf.clip_by_value(noise, -self.eps_clip_value, self.eps_clip_value)

        if self.use_ddim:
            sigma = extract(self.ddim_sigmas, index, tf.shape(x))
            dir_xt = tf.sqrt(1.0 - alpha_prev - sigma**2) * noise
            mu = tf.sqrt(alpha_prev) * x_recon + dir_xt
            var = sigma**2
            logvar = tf.math.log(var)
        else:
            mu = (
                extract(self.ddpm_mu_coef1, t, tf.shape(x)) * x_recon
                + extract(self.ddpm_mu_coef2, t, tf.shape(x)) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, tf.shape(x))
        return mu, logvar

    def forward(self, cond, deterministic=True):
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = tf.shape(sample_data)[0]

        x = tf.random.normal((B, self.horizon_steps, self.action_dim))
        t_all = self.ddim_t if self.use_ddim else tf.reverse(tf.range(self.denoising_steps), axis=[0])

        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t)
            index_b = make_timesteps(B, i)
            mean, logvar = self.p_mean_var(x=x, t=t_b, cond=cond, index=index_b)
            std = tf.sqrt(tf.exp(logvar))

            if self.use_ddim:
                std = tf.zeros_like(std) if deterministic else std
            else:
                std = tf.where(t == 0, tf.zeros_like(std), tf.clip_by_value(std, 1e-3, float("inf")))

            noise = tf.clip_by_value(tf.random.normal(tf.shape(x)), -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = tf.clip_by_value(x, -self.final_action_clip_value, self.final_action_clip_value)
        
        return Sample(x, None)

    def loss(self, x, *args):
        batch_size = tf.shape(x)[0]
        t = tf.random.uniform((batch_size,), minval=0, maxval=self.denoising_steps, dtype=tf.int32)
        return self.p_losses(x, *args, t)

    def p_losses(self, x_start, cond, t):
        noise = tf.random.normal(tf.shape(x_start))
        x_noisy = self.q_sample(x_start, t, noise=noise)
        x_recon = self.network(x_noisy, t, cond=cond)
        
        if self.predict_epsilon:
            return tf.reduce_mean(tf.square(x_recon - noise))
        else:
            return tf.reduce_mean(tf.square(x_recon - x_start))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = tf.random.normal(tf.shape(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    def get_config(self):
        # config = super().get_config()
        # config.update(
        config = {'network': self.network,  # You may need to handle this separately if it's a custom object
                  'horizon_steps': self.horizon_steps,
                  'obs_dim': self.obs_dim,
                  'action_dim': self.action_dim,
                  'network_path': None,  # Optionally store or handle this
                  'device': self.device,
                  'denoised_clip_value': self.denoised_clip_value,
                  'randn_clip_value': self.randn_clip_value,
                  'final_action_clip_value': self.final_action_clip_value,
                  'eps_clip_value': self.eps_clip_value,
                  'denoising_steps': self.denoising_steps,
                  'predict_epsilon': self.predict_epsilon,
                  'use_ddim': self.use_ddim,
                #   'ddim_discretize': self.ddim_discretize,
                  'ddim_steps': self.ddim_steps}
        # )
        return config

    @classmethod
    def from_config(cls, config):
        network = config.pop("network")
        return cls(network=network, **config)