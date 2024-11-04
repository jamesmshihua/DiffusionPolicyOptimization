import math
import tensorflow as tf

class SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class Downsample1d(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(dim, kernel_size=3, strides=2, padding='same')

    def call(self, x):
        return self.conv(x)


class Upsample1d(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv1DTranspose(dim, kernel_size=4, strides=2, padding='same')

    def call(self, x):
        return self.conv(x)


class Conv1dBlock(tf.keras.layers.Layer):
    """
    Conv1D --> GroupNorm --> Mish
    """

    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        n_groups=None,
        activation_type="Mish",
        eps=1e-5,
    ):
        super().__init__()

        if activation_type == "Mish":
            self.act = tf.keras.activations.mish
        elif activation_type == "ReLU":
            self.act = tf.keras.activations.relu
        else:
            raise ValueError("Unknown activation type for Conv1dBlock")

        self.conv = tf.keras.layers.Conv1D(
            out_channels, kernel_size, padding='same'
        )

        self.group_norm = None
        if n_groups is not None:
            self.group_norm = tf.keras.layers.GroupNormalization(
                groups=n_groups, epsilon=eps
            )

    def call(self, x):
        x = self.conv(x)

        if self.group_norm is not None:
            # Reshape for group normalization
            batch_size, channels, horizon = tf.shape(x)
            x = tf.reshape(x, (batch_size, channels // self.group_norm.groups, self.group_norm.groups, horizon))
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            x = tf.reshape(x, (batch_size, channels, horizon))  # Reshape back to original shape
            x = self.group_norm(x)

        x = self.act(x)
        return x
