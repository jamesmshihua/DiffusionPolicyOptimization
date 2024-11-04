import tensorflow as tf
import math


class SinusoidalPosEmb(tf.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.expand_dims(x, axis=1) * tf.expand_dims(emb, axis=0)
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class Downsample1d(tf.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(dim, kernel_size=3, strides=2, padding='same')

    def __call__(self, x):
        return self.conv(x)


class Upsample1d(tf.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv1DTranspose(dim, kernel_size=4, strides=2, padding='same')

    def __call__(self, x):
        return self.conv(x)


class Conv1dBlock(tf.Module):
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
            # self.act = tf.keras.layers.Activation('mish')  # Custom Mish may need to be defined
        elif activation_type == "ReLU":
            self.act = tf.keras.activations.mish()
            self.act = tf.keras.activations.ReLU()
        else:
            raise ValueError("Unknown activation type for Conv1dBlock")

        self.conv = tf.keras.layers.Conv1D(
            out_channels, kernel_size, padding='same'
        )

        if n_groups is not None:
            self.group_norm = tf.keras.layers.GroupNormalization(groups=n_groups, axis=-1, epsilon=eps)
        else:
            self.group_norm = None

    def __call__(self, x):
        x = self.conv(x)
        if self.group_norm is not None:
            x = self.group_norm(x)
        x = self.act(x)
        return x
