import tensorflow as tf
from collections import OrderedDict
import logging

# Define activation dictionary with TensorFlow equivalents
activation_dict = {
    "ReLU": tf.keras.activations.relu,
    "ELU": tf.keras.activations.elu,
    "GELU": tf.keras.activations.gelu,
    "Tanh": tf.keras.activations.tanh,
    "Mish": tf.keras.activations.mish,
    "Identity": tf.keras.activations.linear,
    "Softplus": tf.keras.activations.softplus
}


# activation_dict = {
#     "ReLU": tf.keras.layers.ReLU(),
#     "ELU": tf.keras.layers.ELU(),
#     "GELU": tf.keras.layers.GELU(),
#     "Tanh": tf.keras.layers.Activation("tanh"),
#     "Mish": tf.keras.layers.Activation("tanh"),  # TensorFlow doesn’t have native Mish; consider alternatives
#     "Identity": tf.keras.layers.Activation("linear"),
#     "Softplus": tf.keras.layers.Softplus(),
# }
# activation_dict = {
#     "ReLU": tf.keras.layers.ReLU(),
#     "ELU": tf.keras.layers.ELU(),
#     "GELU": tf.keras.layers.GELU(),
#     "Tanh": tf.keras.layers.Activation("tanh"),
#     "Mish": tf.keras.layers.Activation("tanh"),  # TensorFlow doesn’t have native Mish; consider alternatives
#     "Identity": tf.keras.layers.Activation("linear"),
#     "Softplus": tf.keras.layers.Softplus(),
# }


class MLP(tf.keras.Model):
    def __init__(
            self,
            dim_list,
            append_dim=0,
            append_layers=None,
            activation_type="Tanh",
            out_activation_type="Identity",
            use_layernorm=False,
            use_layernorm_final=False,
            dropout=0,
            use_drop_final=False,
            verbose=False,
    ):
        super(MLP, self).__init__()

        self.module_list = []
        self.append_layers = append_layers
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim

            # Linear layer
            layers = [tf.keras.layers.Dense(o_dim, input_shape=(i_dim,))]

            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(tf.keras.layers.LayerNormalization())

            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(tf.keras.layers.Dropout(dropout))

            # add activation function
            act = (
                activation_dict[activation_type]
                if idx != num_layer - 1
                else activation_dict[out_activation_type]
            )
            layers.append(act)

            # re-construct module
            module = tf.keras.Sequential(layers)
            self.module_list.append(module)

        if verbose:
            logging.info("MLP Layer Structure:", self.module_list)

    def call(self, x, append=None):
        for layer_ind, m in enumerate(self.module_list):
            if append is not None and layer_ind in self.append_layers:
                x = tf.concat([x, append], axis=-1)
            x = m(x)
        return x


class ResidualMLP(tf.keras.Model):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
            self,
            dim_list,
            activation_type="Mish",
            out_activation_type="Identity",
            use_layernorm=False,
            use_layernorm_final=False,
            dropout=0,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0, "Residual layers should be an even number."

        self.layers = [tf.keras.layers.Dense(hidden_dim, input_dim=dim_list[0])]
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(tf.keras.layers.Dense(dim_list[-1], input_dim=hidden_dim))

        if use_layernorm_final:
            self.layers.append(tf.keras.layers.LayerNormalization())
        self.layers.append(activation_dict[out_activation_type])

    def call(self, x, **kwargs):
        for m in self.layers:
            x = m(x)
        return x


class TwoLayerPreActivationResNetLinear(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(hidden_dim)
        self.l2 = tf.keras.layers.Dense(hidden_dim)
        self.act = activation_dict[activation_type]

        if use_layernorm:
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def call(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input
