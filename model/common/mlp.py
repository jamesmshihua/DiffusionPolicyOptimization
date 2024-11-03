import tensorflow as tf
from collections import OrderedDict
import logging

# Define activation dictionary with TensorFlow equivalents
activation_dict = {
    "ReLU": tf.keras.layers.ReLU(),
    "ELU": tf.keras.layers.ELU(),
    "GELU": tf.keras.layers.GaussianDropout(),
    "Tanh": tf.keras.layers.Activation("tanh"),
    "Mish": tf.keras.layers.Activation("mish"),  # Custom Mish may be required
    "Identity": tf.keras.layers.Activation("linear"),
    "Softplus": tf.keras.layers.Activation("softplus"),
}

class MLP(tf.Module):
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
        super().__init__()
        
        # Initialize model layer list
        self.moduleList = []
        self.append_layers = append_layers
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in append_layers:
                i_dim += append_dim
            linear_layer = tf.keras.layers.Dense(o_dim)
            
            # Construct layer list
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", tf.keras.layers.LayerNormalization()))
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", tf.keras.layers.Dropout(dropout)))

            # Add activation function
            act = activation_dict[activation_type] if idx != num_layer - 1 else activation_dict[out_activation_type]
            layers.append(("act_1", act))

            # Create a sequential model for each layer and add it to module list
            module = tf.keras.Sequential([layer[1] for layer in layers])
            self.moduleList.append(module)
        
        if verbose:
            logging.info(self.moduleList)

    def __call__(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = tf.concat((x, append), axis=-1)
            x = m(x)
        return x


class ResidualMLP(tf.Module):
    """
    Simple multi-layer perceptron network with residual connections.
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
        super().__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0

        # Define model layers
        self.layers = [tf.keras.layers.Dense(hidden_dim)]
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
        self.layers.append(tf.keras.layers.Dense(dim_list[-1]))
        if use_layernorm_final:
            self.layers.append(tf.keras.layers.LayerNormalization())
        self.layers.append(activation_dict[out_activation_type])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TwoLayerPreActivationResNetLinear(tf.Module):
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
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-06)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-06)
        
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def __call__(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input
