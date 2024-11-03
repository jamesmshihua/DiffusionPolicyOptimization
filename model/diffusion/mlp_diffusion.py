import tensorflow as tf
import logging
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

log = logging.getLogger(__name__)

class DiffusionMLP(tf.Module):
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        output_dim = action_dim * horizon_steps

        self.time_embedding = tf.keras.Sequential([
            SinusoidalPosEmb(time_dim),
            tf.keras.layers.Dense(time_dim * 2),
            tf.keras.layers.Activation("tanh"),  # Assuming Mish can be replaced with tanh for simplicity
            tf.keras.layers.Dense(time_dim),
        ])

        mlp_class = ResidualMLP if residual_style else MLP
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + action_dim * horizon_steps + cond_dim
        self.mlp_mean = mlp_class(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def __call__(self, x, time, cond, **kwargs):
        B, Ta, Da = x.shape

        # flatten chunk
        x = tf.reshape(x, [B, -1])

        # flatten history
        state = tf.reshape(cond["state"], [B, -1])

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # append time and cond
        time = tf.reshape(time, [B, 1])
        time_emb = self.time_embedding(time)
        time_emb = tf.reshape(time_emb, [B, self.time_dim])
        x = tf.concat([x, time_emb, state], axis=-1)

        # mlp head
        out = self.mlp_mean(x)
        return tf.reshape(out, [B, Ta, Da])


# class VisionDiffusionMLP(tf.Module):
#     """With ViT backbone"""

#     def __init__(
#         self,
#         backbone,
#         action_dim,
#         horizon_steps,
#         cond_dim,
#         img_cond_steps=1,
#         time_dim=16,
#         mlp_dims=[256, 256],
#         activation_type="Mish",
#         out_activation_type="Identity",
#         use_layernorm=False,
#         residual_style=False,
#         spatial_emb=0,
#         visual_feature_dim=128,
#         dropout=0,
#         num_img=1,
#         augment=False,
#     ):
#         super().__init__()

#         # vision
#         self.backbone = backbone
#         self.augment = augment
#         self.num_img = num_img
#         self.img_cond_steps = img_cond_steps
#         self.aug = RandomShiftsAug(pad=4) if augment else None

#         if spatial_emb > 0:
#             assert spatial_emb > 1, "spatial_emb must be greater than 1"
#             if num_img > 1:
#                 self.compress1 = SpatialEmb(
#                     num_patch=self.backbone.num_patch,
#                     patch_dim=self.backbone.patch_repr_dim,
#                     prop_dim=cond_dim,
#                     proj_dim=spatial_emb,
#                     dropout=dropout,
#                 )
#                 self.compress2 = deepcopy(self.compress1)
#             else:
#                 self.compress = SpatialEmb(
#                     num_patch=self.backbone.num_patch,
#                     patch_dim=self.backbone.patch_repr_dim,
#                     prop_dim=cond_dim,
#                     proj_dim=spatial_emb,
#                     dropout=dropout,
#                 )
#             visual_feature_dim = spatial_emb * num_img
#         else:
#             self.compress = tf.keras.Sequential([
#                 tf.keras.layers.Dense(visual_feature_dim),
#                 tf.keras.layers.LayerNormalization(),
#                 tf.keras.layers.Dropout(dropout),
#                 tf.keras.layers.ReLU()
#             ])

#         # diffusion
#         input_dim = time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
#         output_dim = action_dim * horizon_steps

#         self.time_embedding = tf.keras.Sequential([
#             SinusoidalPosEmb(time_dim),
#             tf.keras.layers.Dense(time_dim * 2),
#             tf.keras.layers.Activation("tanh"),  # Assuming Mish can be replaced with tanh for simplicity
#             tf.keras.layers.Dense(time_dim),
#         ])
        
#         mlp_class = ResidualMLP if residual_style else MLP
#         self.mlp_mean = mlp_class(
#             [input_dim] + mlp_dims + [output_dim],
#             activation_type=activation_type,
#             out_activation_type=out_activation_type,
#             use_layernorm=use_layernorm,
#         )
#         self.time_dim = time_dim

#     def __call__(self, x, time, cond, **kwargs):
#         B, Ta, Da = x.shape
#         _, T_rgb, C, H, W = cond["rgb"].shape

#         # flatten chunk
#         x = tf.reshape(x, [B, -1])

#         # flatten history
#         state = tf.reshape(cond["state"], [B, -1])

#         # Take recent images
#         rgb = cond["rgb"][:, -self.img_cond_steps:]

#         # concatenate images in cond by channels
#         if self.num_img > 1:
#             rgb = tf.reshape(rgb, [B, T_rgb, self.num_img, 3, H, W])
#             rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
#         else:
#             rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

#         # convert rgb to float32 for augmentation
#         rgb = tf.cast(rgb, tf.float32)

#         # get vit output - pass in two images separately
#         if self.num_img > 1:
#             rgb1 = rgb[:, 0]
#             rgb2 = rgb[:, 1]
#             if self.augment:
#                 rgb1 = self.aug(rgb1)
#                 rgb2 = self.aug(rgb2)
#             feat1 = self.backbone(rgb1)
#             feat2 = self.backbone(rgb2)
#             feat1 = self.compress1(feat1, state)
#             feat2 = self.compress2(feat2, state)
#             feat = tf.concat([feat1, feat2], axis=-1)
#         else:  # single image
#             if self.augment:
#                 rgb = self.aug(rgb)
#             feat = self.backbone(rgb)
#             feat = self.compress(feat) if not isinstance(self.compress, SpatialEmb) else self.compress(feat, state)

#         cond_encoded = tf.concat([feat, state], axis=-1)

#         # append time and cond
#         time = tf.reshape(time, [B, 1])
#         time_emb = self.time_embedding(time)
#         time_emb = tf.reshape(time_emb, [B, self.time_dim])
#         x = tf.concat([x, time_emb, cond_encoded], axis=-1)

#         # mlp
#         out = self.mlp_mean(x)
#         return tf.reshape(out, [B, Ta, Da])

