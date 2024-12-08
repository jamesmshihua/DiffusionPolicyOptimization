"""
Policy gradient with diffusion policy. VPG: vanilla policy gradient

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

import copy
import tensorflow as tf
import logging

log = logging.getLogger(__name__)
# import torch.nn.functional as F

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract
# from torch.distributions import Normal
import tensorflow_probability as tfp

class VPGDiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        # eta in DDIM
        eta=None,
        learn_eta=False,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."

        # Number of denoising steps to use with fine-tuned model. Thus denoising_step - ft_denoising_steps is the number of denoising steps to use with original model.
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d  # annealing step size
        self.ft_denoising_steps_t = ft_denoising_steps_t  # annealing interval
        self.ft_denoising_steps_cnt = 0

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

        # Minimum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std = min_logprob_denoising_std

        # Learnable eta
        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta.to(self.device)
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False
                logging.info("Turned off gradients for eta")

        # Re-name network to actor
        self.actor = self.network

        if network_path is not None:
            # checkpoint = torch.load(
            #     network_path, map_location=self.device, weights_only=True
            # )
            # if "ema" not in checkpoint:  # load trained RL model
            #     self.load_state_dict(checkpoint["model"], strict=False)
            #     logging.info("Loaded critic from %s", network_path)
            dummy_x_noisy = tf.zeros((10, 4, 3))
            dummy_t = tf.zeros((10,))
            dummy_cond = {
                "state": tf.zeros((10,1,11))
            }
            
            self.actor(dummy_x_noisy, dummy_t, cond=dummy_cond)
            self.actor.load_weights(self.network_path)

        # Make a copy of the original model
        self.actor_ft = copy.deepcopy(self.actor)
        self.actor_ft(dummy_x_noisy, dummy_t, cond=dummy_cond)
        self.actor_ft.load_weights(self.network_path)
        logging.info("Cloned model for fine-tuning")

        # Turn off gradients for original model
        for param in self.actor.trainable_variables:
            param.trainable = False
        logging.info("Turned off gradients of the pretrained network")
        logging.info(
            f"Number of finetuned parameters: {sum(tf.reduce_prod(p.shape) for p in self.actor_ft.trainable_variables)}"
        )

        # Value function
        self.critic = critic
        self.critic(dummy_cond)
        
    # ---------- Sampling ----------#

    def step(self):
        """
        Anneal min_sampling_denoising_std and fine-tuning denoising steps

        Current configs do not apply annealing
        """
        # anneal min_sampling_denoising_std
        if type(self.min_sampling_denoising_std) is not float:
            self.min_sampling_denoising_std.step()

        # anneal denoising steps
        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(
                0, self.ft_denoising_steps - self.ft_denoising_steps_d
            )

            # update actor
            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            # for param in self.actor.parameters():
            #     param.requires_grad = False
            logging.info(
                f"Finished annealing fine-tuning denoising steps to {self.ft_denoising_steps}"
            )

    def get_min_sampling_denoising_std(self):
        if type(self.min_sampling_denoising_std) is float:
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std()

    # override
    @tf.function
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):
        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = tf.identity(index >= (self.ddim_steps - self.ft_denoising_steps))
        else:
            ft_indices = tf.identity(t < self.ft_denoising_steps)

        # Use base policy to query expert model, e.g. for imitation loss
        # actor = self.actor if use_base_policy else self.actor_ft

        # overwrite noise for fine-tuning steps
        # if tf.reduce_sum(tf.cast(ft_indices, tf.int32)) > 0:
        if ft_indices[0] == True:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            # noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            if use_base_policy:
                noise_ft = self.actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            else:
                noise_ft = self.actor_ft(x[ft_indices], t[ft_indices], cond=cond_ft)
            # noise[ft_indices] = noise_ft
            noise = noise_ft

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            # x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            x_recon = tf.clip_by_value(x_recon, -self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            # noise.clamp_(-self.eps_clip_value, self.eps_clip_value)
            tf.clip_by_value(noise, -self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:
                etas = tf.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                # etas = self.eta(cond).unsqueeze(1)  # B x 1 x (Da or 1)
                etas = tf.expand_dims(self.eta(cond), axis=1)
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            # dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            dir_xt_coef = tf.sqrt(tf.clip_by_value(1.0 - alpha_prev - sigma**2, 0, 1e6))
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = tf.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = tf.ones_like(mu)  # always one for DDPM
        return mu, logvar, etas

    # override
    # @torch.no_grad()
    @tf.function
    def call(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Forward pass for sampling actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: If true, then std=0 with DDIM, or with DDPM, use normal schedule (instead of clipping at a higher value)
            return_chain: whether to return the entire chain of denoised actions
            use_base_policy: whether to use the frozen pre-trained policy instead
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
                chain: (B, K + 1, Ta, Da)
        """
        # device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Get updated minimum sampling denoising std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        # Loop
        x = tf.random.normal((B, self.horizon_steps, self.action_dim))
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t)
            index_b = make_timesteps(B, i)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = tf.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                if deterministic:
                    std = tf.zeros_like(std)
                else:
                    std = tf.clip_by_value(std, min_sampling_denoising_std, 1e6)
            else:
                if deterministic and t == 0:
                    std = tf.zeros_like(std)
                elif deterministic:  # still keep the original noise
                    std = tf.clip_by_value(std, 1e-3, 1e6)
                else:  # use higher minimum noise
                    std = tf.clip_by_value(std, min_sampling_denoising_std, 1e6)
            # noise = tf.randn_like(x).clamp_(
            #     -self.randn_clip_value, self.randn_clip_value
            # )
            noise = tf.clip_by_value(tf.random.normal(tf.shape(x)), -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                # x = torch.clamp(
                #     x, -self.final_action_clip_value, self.final_action_clip_value
                # )
                x = tf.clip_by_value(x, -self.final_action_clip_value, self.final_action_clip_value)

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                    self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = tf.stack(chain, axis=1)
        return Sample(x, chain)

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of the entire chain of denoised actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B x K, Ta, Da)
            entropy (if get_ent=True):  (B x K, Ta)
        """
        # Repeat cond for denoising_steps, flatten batch and time dimensions
        # cond = {
        #     key: cond[key]
        #     .unsqueeze(1)
        #     .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
        #     .flatten(start_dim=0, end_dim=1)
        #     for key in cond
        # }  # less memory usage than einops?
        
        for key in cond:
            temp = cond[key]
            temp = tf.expand_dims(temp, axis=1)
            temp = tf.tile(temp, multiples=[1, self.ft_denoising_steps] + [1] * (len(temp.shape) - 2))
            temp = tf.reshape(temp, (-1, *temp.shape[2:]))
            cond[key] = temp

        # Repeat t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = tf.identity(tf.range(self.ft_denoising_steps - 1, -1, -1))
            # 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0
        # t_all = t_single.repeat(chains.shape[0], 1).flatten()
        t_all = tf.reshape(
            tf.tile(tf.expand_dims(t_single, 0), multiples=(chains.shape[0], 1)), (1,-1)
        )[0]

        if self.use_ddim:
            indices_single = tf.range(
                start=self.ddim_steps - self.ft_denoising_steps,
                limit=self.ddim_steps,
            ).gpu()
            indices = tf.tile(tf.expand_dims(indices_single, 0), multiples=(chains.shape[0], 1))
        else:
            indices = None

        # Split chains
        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]

        # Flatten first two dimensions
        chains_prev = tf.reshape(chains_prev, (-1, self.horizon_steps, self.action_dim))
        chains_next = tf.reshape(chains_next, (-1, self.horizon_steps, self.action_dim))

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, self.min_logprob_denoising_std, 1e6)
        dist = tfp.distributions.normal.Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of random samples of denoised chains.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B, Ta, Da)
            entropy (if get_ent=True):  (B, Ta)
            denoising_indices: (B, )
        """
        # Sample t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = tf.identity(tf.range(self.ft_denoising_steps-1, -1, -1))
            # 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0
        t_all = tf.gather(t_single, denoising_inds)
        if self.use_ddim:
            ddim_indices_single = tf.identity(tf.range(self.ddim_steps - self.ft_denoising_steps, self.ddim_steps))  # only used for DDIM
            ddim_indices = tf.gather(ddim_indices_single, denoising_inds)
        else:
            ddim_indices = None

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, self.min_logprob_denoising_std, 1e6)
        dist = tfp.distributions.normal.Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def c_loss(self, cond, chains, reward):
        """
        REINFORCE loss. Not used right now.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            reward (to go): (b,)
        """
        # Get advantage
        # with torch.no_grad():
        value = tf.squeeze(self.critic(cond))
        advantage = reward - value

        # Get logprobs for denoising steps from T-1 to 0
        logprobs, eta = self.get_logprobs(cond, chains, get_ent=True)
        # (n_steps x n_envs x K) x Ta x (Do+Da)

        # Ignore obs dimension, and then sum over action dimension
        logprobs = logprobs[:, :, : self.action_dim].sum(-1)
        # -> (n_steps x n_envs x K) x Ta

        # -> (n_steps x n_envs) x K x Ta
        logprobs = logprobs.reshape((-1, self.denoising_steps, self.horizon_steps))

        # Sum/avg over denoising steps
        logprobs = logprobs.mean(-2)  # -> (n_steps x n_envs) x Ta

        # Sum/avg over horizon steps
        logprobs = logprobs.mean(-1)  # -> (n_steps x n_envs)

        # Get REINFORCE loss
        loss_actor = tf.reduce_mean(-logprobs * advantage)

        # Train critic to predict state value
        pred = tf.squeeze(self.critic(cond))
        loss_critic = tf.keras.losses.MSE(pred, reward)
        return loss_actor, loss_critic, eta
