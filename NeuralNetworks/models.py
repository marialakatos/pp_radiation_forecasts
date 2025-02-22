import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from losses import custom_loss_crps_wrapper_fct

"""
Neural Network builder

This function constructs a feedforward neural network with customizable architecture, activation functions,  
and an optional compilation step.

Args:
    n_features (int): Number of input features.
    n_outputs (int): Number of output neurons.
    hidden_nodes (int or list): Number of nodes in each hidden layer. Can be a single integer or a list of integers.
    compile_model (bool, optional): If True, compiles the model. Default is False.
    optimizer (str, optional): Optimizer to use if compiling. Default is 'adam'.
    lr (float, optional): Learning rate for the optimizer. Default is 0.01.
    loss (callable, optional): Loss function to use if compiling. Default is custom_loss_crps_wrapper_fct(LB=0).
    activation (str, optional): Activation function for hidden layers. Default is 'relu'.

Returns:
    keras.Model: A compiled or uncompiled Keras model, depending on `compile_model`.
"""

def build_hidden_model(n_features, n_outputs, hidden_nodes, compile_model=False,
                       optimizer='adam', lr=0.01, loss=custom_loss_crps_wrapper_fct(LB=0),
                       activation='relu'):

    if type(hidden_nodes) is not list:
        hidden_nodes = [hidden_nodes]

    inp = Input(shape=(n_features,))
    x = inp

    for h in hidden_nodes:
        x = Dense(h, activation=activation)(x)

    x = Dense(n_outputs, activation='linear')(x)
    model = Model(inputs=inp, outputs=x)

    if compile_model:
        opt = tf.keras.optimizers.get({'class_name': optimizer, 'config': {'learning_rate': lr}})
        model.compile(optimizer=opt, loss=loss)

    return model


