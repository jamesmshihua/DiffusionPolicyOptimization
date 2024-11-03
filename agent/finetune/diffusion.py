import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DiffusionModel(tf.Module):
    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        denoising_steps=100,
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        use_ddim=False,
        ddim_steps=None,
    ):
        super(DiffusionModel, self).__init__()
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.network = network  # Assuming network is a tf.keras.Model

        # Define DDPM parameters
        self.betas = self.cosine_beta_schedule(denoising_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        self.alphas_cumprod_prev = tf.concat(
            [tf.ones((1,)), self.alphas_cumprod[:-1]], axis=0
        )
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod - 1)
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = tf.math.log(tf.clip_by_value(self.ddpm_var, 1e-20, float("inf")))

    def cosine_beta_schedule(self, num_steps):
        return tf.convert_to_tensor(np.linspace(0.0001, 0.02, num_steps), dtype=tf.float32)

    def p_mean_var(self, x, t, cond):
        noise = self.network([x, t, cond])
        if self.use_ddim:
            alpha = tf.gather(self.ddim_alphas, t)
            alpha_prev = tf.gather(self.ddim_alphas_prev, t)
            sqrt_one_minus_alpha = tf.gather(self.ddim_sqrt_one_minus_alphas, t)
            x_recon = (x - sqrt_one_minus_alpha * noise) / tf.sqrt(alpha)
        else:
            x_recon = self.sqrt_recip_alphas_cumprod[t] * x - self.sqrt_recipm1_alphas_cumprod[t] * noise
        return x_recon

    @tf.function
    def sample(self, cond):
        B = tf.shape(cond['state'])[0]
        x = tf.random.normal((B, self.horizon_steps, self.action_dim))

        t_all = tf.range(self.denoising_steps - 1, -1, -1) if not self.use_ddim else self.ddim_t

        for i, t in enumerate(t_all):
            mean, logvar = self.p_mean_var(x, t, cond)
            std = tf.exp(0.5 * logvar)
            std = tf.where(t == 0, tf.zeros_like(std), std)  # Set std to zero at the final step
            noise = tf.clip_by_value(tf.random.normal(tf.shape(x)), -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = tf.clip_by_value(x, -self.final_action_clip_value, self.final_action_clip_value)

        return x

    def loss(self, x, cond):
        t = tf.random.uniform((tf.shape(x)[0],), maxval=self.denoising_steps, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x))
        x_noisy = self.q_sample(x, t, noise)
        x_recon = self.network([x_noisy, t, cond])
        return tf.reduce_mean(tf.square(x_recon - noise))

    def q_sample(self, x_start, t, noise):
        return self.sqrt_alphas_cumprod[t] * x_start + self.sqrt_one_minus_alphas_cumprod[t] * noise
