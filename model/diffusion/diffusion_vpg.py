from model.diffusion.diffusion import DiffusionModel
import tensorflow as tf
from tensorflow_probability import distributions as tfd
# import copy

class VPGDiffusion(DiffusionModel):
    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        eta=None,
        learn_eta=False,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs
        )
        
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."
        
        # Fine-tuning denoising steps
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d
        self.ft_denoising_steps_t = ft_denoising_steps_t
        self.ft_denoising_steps_cnt = 0

        # Minimum std values for stability
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.learn_eta = learn_eta
        # Eta - learnable parameter for DDIM sampling
        self.eta = tf.Variable(eta if eta is not None else 0.0, trainable=learn_eta, dtype=tf.float32)

        # Rename actor and create fine-tuning copy
        self.actor = actor  # Main actor network
        self.actor_ft = tf.keras.models.clone_model(actor)
        # self.actor_ft.set_weights(self.actor.get_weights())

        # Turn off gradients for original actor
        self.actor.trainable = False

        # Critic (value function) for policy gradient updates
        self.critic = critic

    def step(self):
        """ Anneals min_sampling_denoising_std and fine-tuning denoising steps. """
        # Update min_sampling_denoising_std if it's learnable
        if isinstance(self.min_sampling_denoising_std, tf.Variable):
            self.min_sampling_denoising_std.assign(self.min_sampling_denoising_std - self.ft_denoising_steps_d)
        
        # Update fine-tuning denoising steps every interval
        self.ft_denoising_steps_cnt += 1
        if self.ft_denoising_steps_d > 0 and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0:
            self.ft_denoising_steps = max(0, self.ft_denoising_steps - self.ft_denoising_steps_d)

            # Replace actor with fine-tuned actor
            self.actor = self.actor_ft
            self.actor_ft = tf.keras.models.clone_model(self.actor)

    def p_mean_var(self, x, t, cond, use_base_policy=False, deterministic=False):
        """ Overridden function to include fine-tuning. """
        # Get noise prediction
        noise = self.actor(x, t, cond=cond) if use_base_policy else self.actor_ft(x, t, cond=cond)

        # Calculate reconstruction
        if self.predict_epsilon:
            if self.use_ddim:
                alpha = tf.gather(self.ddim_alphas, t)
                sqrt_one_minus_alpha = tf.gather(self.ddim_sqrt_one_minus_alphas, t)
                x_recon = (x - sqrt_one_minus_alpha * noise) / tf.sqrt(alpha)
            else:
                x_recon = self.sqrt_recip_alphas_cumprod[t] * x - self.sqrt_recipm1_alphas_cumprod[t] * noise
        else:
            x_recon = noise

        # Clip for numerical stability
        x_recon = tf.clip_by_value(x_recon, -self.denoised_clip_value, self.denoised_clip_value)
        if self.use_ddim and not deterministic:
            std = tf.sqrt(self.eta * (1 - alpha) / alpha)
        else:
            std = tf.sqrt(self.betas[t])
        return x_recon, std

    def call(self, cond, deterministic=False, return_chain=True, use_base_policy=False):
        """ Sampling forward pass, returning trajectory or full chain if specified. """
        B = tf.shape(cond['state'])[0]
        x = tf.random.normal((B, self.horizon_steps, self.action_dim), dtype=tf.float32)
        
        t_all = tf.range(self.denoising_steps - 1, -1, -1) if not self.use_ddim else self.ddim_t
        chain = [] if return_chain else None

        for t in t_all:
            mean, std = self.p_mean_var(x, t, cond, use_base_policy=use_base_policy, deterministic=deterministic)
            noise = tf.random.normal(tf.shape(x)) * std
            x = mean + noise

            if return_chain:
                chain.append(x)

        return (x, chain) if return_chain else x

    def c_loss(self, cond, chains, reward):
        """ Calculates REINFORCE loss for actor-critic training. """
        # Get critic's value for baseline
        value = tf.squeeze(self.critic(cond), axis=-1)
        advantage = reward - value

        # Compute log-probabilities
        logprobs = self.get_logprobs(cond, chains)
        logprobs = tf.reduce_sum(logprobs, axis=-1)  # Sum across action dimensions
        logprobs = tf.reduce_mean(logprobs, axis=-1)  # Mean over horizon steps

        # REINFORCE loss calculation
        loss_actor = -tf.reduce_mean(logprobs * advantage)
        loss_critic = tf.reduce_mean(tf.square(value - reward))
        return loss_actor, loss_critic

    def get_logprobs(self, cond, chains):
        """ Compute log-probabilities of actions in the denoised chain. """
        chains_prev, chains_next = chains[:, :-1], chains[:, 1:]
        chains_prev = tf.reshape(chains_prev, (-1, self.horizon_steps, self.action_dim))
        chains_next = tf.reshape(chains_next, (-1, self.horizon_steps, self.action_dim))

        # Predict next mean and std
        mean, logvar, _ = self.p_mean_var(chains_prev, cond=cond)
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, min=self.min_logprob_denoising_std, clip_value_max=float('inf'))

        # Calculate log-probability
        dist = tfd.Normal(mean, std)
        log_prob = dist.log_prob(chains_next)
        return log_prob

    def get_config(self):
        # config = dict()
        config = super().get_config()
        config["actor"] = self.actor
        config["critic"] = self.critic
        config["ft_denoising_steps"] = self.ft_denoising_steps
        config["ft_denoising_steps_d"] = self.ft_denoising_steps_d
        config["ft_denoising_steps_t"] = self.ft_denoising_steps_t
        config["network_path"] = self.network_path
        config["min_sampling_denoising_std"] = self.min_sampling_denoising_std
        config["min_logprob_denoising_std"] = self.min_logprob_denoising_std
        config["eta"] = self.eta
        config["learn_eta"] = self.learn_eta
        return config