import keras


def TCN_Residual_Block(x, filters=None, kernel_size=None,
                       dilation_rate=None, dropout_rate=None):
    """
    TCN residual block based on residual block
    Note: This residual block consists of two 1-D dilation causal convolution layers
    which have the same parameters (e.g. kernel_size, dilation_rate, dropout_rate).
    ----------------------------------------------------------------------------------
    Args:
         x (tf.Tensor);         Input
         filters (int);         Number of filters for 1-D convolution layer
         kernel_size (int);     Size of convolution window for 1-D convolution layer
         dilation_rate (int);   Dilation rate of 1-D convolution layer
         dropout_rate (float);  Dropout rate of TCN residual block
    Returns:
         y (tf.Tensor);         Outputs
    """

    # The first dilation causal convolution layer
    y = keras.layers.Conv1D(filters=filters,
                            kernel_size=kernel_size,
                            padding='causal',
                            dilation_rate=dilation_rate)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation(activation='relu')(y)
    y = keras.layers.Dropout(rate=dropout_rate)(y)
    # The second dilation causal convolution layer
    y = keras.layers.Conv1D(filters=filters,
                            kernel_size=kernel_size,
                            padding='causal',
                            dilation_rate=dilation_rate)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation(activation='relu')(y)
    y = keras.layers.Dropout(rate=dropout_rate)(y)
    # Residual connection
    if x.shape[-1] == y.shape[-1]:
        # Identity map
        residual = x
    else:
        # Conv1D map
        residual = keras.layers.Conv1D(filters=int(y.shape[-1]),
                                       kernel_size=1,
                                       padding='same')(x)
    y = keras.layers.Add()([y, residual])
    return y


def TCN(input_shape=None, n_outputs=None, n_blocks=None,
        filters=None, kernel_size=None, dropout_rate=None,
        dense_units=None, name='TCN', optimizer='adam', loss='mse'):
    """
    TCN model
    Note: This version of TCN uses a dense layer as output layer instead of
    fully convolution network (FCN) structure.
    ----------------------------------------------------------------------------------
    Args:
        input_shape (tuple):   Input shape of data, (n_steps, n_features)
        n_outputs (int):       Length of output sequence
        n_blocks (int):        Number of TCN residual blocks
        filters (list):        A list containing number of filters for each TCN residual blocks
        kernel_size (list):    A list containing size of convolution window for each TCN residual blocks
        dropout_rate (list):   A list containing dropout rate for each TCN residual blocks
        dense_units (list):    A list containing number of units for each dense layers (output layer excepted)
        name (str):            Name of model
        loss:
        optimizer:
    Returns:
        model (keras model)    TCN model
    """

    # Input layer
    _input = keras.Input(shape=input_shape, name='input_layer')
    # TCN residual blocks
    _output = None
    for k in range(n_blocks):
        dilation_rate = 2**k
        if k == 0:
            _output = TCN_Residual_Block(_input,
                                         filters=filters[k],
                                         kernel_size=kernel_size[k],
                                         dilation_rate=dilation_rate,
                                         dropout_rate=dropout_rate[k])
        else:
            _output = TCN_Residual_Block(_output,
                                         filters=filters[k],
                                         kernel_size=kernel_size[k],
                                         dilation_rate=dilation_rate,
                                         dropout_rate=dropout_rate[k])
    # Flatten layer
    _output = keras.layers.Flatten()(_output)
    # Dense layers (optional)
    if dense_units is not None:
        n_dense = len(dense_units)
        for k in range(n_dense):
            _output = keras.layers.Dense(units=dense_units[k],
                                         activation='relu')(_output)
    # Output layer
    _output = keras.layers.Dense(units=n_outputs, activation='relu')(_output)
    # Build model
    model = keras.Model(inputs=_input, outputs=_output, name=name)
    # Compile model
    model.compile(optimizer=optimizer, loss=loss)
    return model
