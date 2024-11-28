"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main

"""

import tensorflow as tf
import tensorflow_addons as tfa
from model.common.grid_sampler import nearest_sampler


class SpatialEmb(tf.keras.Model):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):
        super().__init__()

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(proj_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.weight = tf.Variable(tf.random.normal([1, num_proj, proj_dim]), trainable=True)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def call(self, feat: tf.Tensor, prop: tf.Tensor):
        feat = tf.transpose(feat, perm=[0, 2, 1])

        if self.prop_dim > 0:
            repeated_prop = tf.tile(tf.expand_dims(prop, 1), [1, tf.shape(feat)[1], 1])
            feat = tf.concat([feat, repeated_prop], axis=-1)

        y = self.input_proj(feat)
        z = tf.reduce_sum(self.weight * y, axis=1)
        z = self.dropout(z)
        return z


class RandomShiftsAug(tf.keras.Model):
    def __init__(self, pad):
        self.pad = pad

    def call(self, x):
        n, c, h, w = x.size()
        assert h == w, "Input height and width must be the same."
        padding = tf.constant([[self.pad, self.pad], [self.pad, self.pad]])
        x = tf.pad(x, padding, mode='SYMMETRIC')
        eps = 1.0 / (h + 2 * self.pad)
        arange = tf.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad)[:h]
        arange = tf.tile(arange[tf.newaxis, :], [h, 1])[:, :, tf.newaxis]
        base_grid = tf.concat([arange, tf.transpose(arange, perm=(1, 0, 2))], axis=-1)
        base_grid = tf.tile(base_grid[tf.newaxis, ...], [n, 1, 1, 1])

        shift = tf.random.uniform(
            shape=(n, 1, 1, 2), minval=0, maxval=2 * self.pad + 1, dtype=tf.float32
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        # x must have a shape (B, H, W, C)
        # (B, C, H, W) - > (B, H, W, C)
        x = tf.transpose(x, perm=(0, 2, 3, 1))
        return nearest_sampler(x, grid)

# # test random shift
# if __name__ == "__main__":
#     from PIL import Image
#     import requests
#     import numpy as np
#
#     image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
#     image = Image.open(requests.get(image_url, stream=True).raw)
#     image = image.resize((96, 96))
#
#     image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
#     aug = RandomShiftsAug(pad=4)
#     image_aug = aug(image)
#     image_aug = image_aug.squeeze().permute(1, 2, 0).numpy()
#     image_aug = Image.fromarray(image_aug.astype(np.uint8))
#     image_aug.show()
