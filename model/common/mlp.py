import tensorflow as tf
from collections import OrderedDict
import logging

# Define activation dictionary with TensorFlow equivalents
# activation_dict = {
#     "ReLU": tf.keras.activations.relu,
#     "ELU": tf.keras.activations.elu,
#     "GELU": tf.keras.activations.gelu,
#     "Tanh": tf.keras.activations.tanh,
#     "Mish": tf.keras.activations.mish,
#     "Identity": tf.keras.activations.linear,
#     "Softplus": tf.keras.activations.softplus
# }
activation_dict = {
    "ReLU": tf.keras.layers.ReLU(),
    "ELU": tf.keras.layers.ELU(),
    "GELU": tf.keras.layers.GELU(),
    "Tanh": tf.keras.layers.Activation("tanh"),
    "Mish": tf.keras.layers.Activation("tanh"),  # TensorFlow doesn’t have native Mish; consider alternatives
    "Identity": tf.keras.layers.Activation("linear"),
    "Softplus": tf.keras.layers.Softplus(),
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

        self.append_layers = append_layers or []
        self.module_list = []

        num_layers = len(dim_list) - 1
        for idx in range(num_layers):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim

            # Linear layer
            layer = [tf.keras.layers.Dense(o_dim, input_shape=(i_dim,))]

            # Layer normalization
            if use_layernorm and (idx < num_layers - 1 or use_layernorm_final):
                layer.append(tf.keras.layers.LayerNormalization())

            # Dropout
            if dropout > 0 and (idx < num_layers - 1 or use_drop_final):
                layer.append(tf.keras.layers.Dropout(dropout))

            # Activation
            activation = (
                activation_dict[activation_type]
                if idx != num_layers - 1
                else activation_dict[out_activation_type]
            )
            layer.append(activation)

            # Store the layer
            self.module_list.append(tf.keras.Sequential(layer))

        if verbose:
            print("MLP Layer Structure:", self.module_list)

    def call(self, x, append=None):
        for layer_ind, layer in enumerate(self.module_list):
            if append is not None and layer_ind in self.append_layers:
                x = tf.concat([x, append], axis=-1)
            x = layer(x)
        return x


class ResidualMLP(tf.keras.Model):
    """
    Simple multi-layer perceptron network with residual connections.
    The residual layers are based on the IBC paper implementation, which uses
    2 residual layers with pre-activation, with or without dropout and normalization.
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

        # Input layer
        self.input_layer = tf.keras.layers.Dense(hidden_dim)

        # Residual layers
        self.residual_blocks = []
        for _ in range(1, num_hidden_layers, 2):
            self.residual_blocks.append(
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
            )

        # Output layer
        self.output_layer = tf.keras.layers.Dense(dim_list[-1])

        if use_layernorm_final:
            self.layernorm_final = tf.keras.layers.LayerNormalization()
        else:
            self.layernorm_final = None

        self.out_activation = activation_dict[out_activation_type]

    def call(self, x, **kwargs):
        # Pass through the input layer
        x = self.input_layer(x)

        # Apply each residual block
        for block in self.residual_blocks:
            x = block(x)

        # Pass through output layer
        x = self.output_layer(x)

        # Optionally apply final layer normalization
        if self.layernorm_final:
            x = self.layernorm_final(x)

        # Apply output activation
        x = self.out_activation(x)

        return x


class TwoLayerPreActivationResNetLinear(tf.keras.layers.Layer):
    """
    Residual block consisting of two linear layers with pre-activation.
    """

    def __init__(self, hidden_dim, activation_type="Mish", use_layernorm=False, dropout=0):
        super(TwoLayerPreActivationResNetLinear, self).__init__()
        self.l1 = tf.keras.layers.Dense(hidden_dim)
        self.l2 = tf.keras.layers.Dense(hidden_dim)
        self.act = activation_dict[activation_type]

        if use_layernorm:
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.norm1 = None
            self.norm2 = None

        if dropout > 0:
            self.dropout = tf.keras.layers.Dropout(dropout)
        else:
            self.dropout = None

    def call(self, x):
        x_input = x

        # Pre-activation and first linear layer
        if self.norm1:
            x = self.norm1(x)
        x = self.act(x)
        x = self.l1(x)

        # Optional dropout after first linear layer
        if self.dropout:
            x = self.dropout(x)

        # Pre-activation and second linear layer
        if self.norm2:
            x = self.norm2(x)
        x = self.act(x)
        x = self.l2(x)

        # Residual connection
        return x + x_input
