"""
Created on Tue Sept 22 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Conv3D, Activation, Concatenate, BatchNormalization, Input, Dropout, \
    MaxPooling3D, UpSampling3D, AveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from internals.metrics import matlab_style_gauss2D, L1L2loss
from internals.metrics import jaccard_multiple_output


def jaccard_sparse3D(y_true, y_pred, skip_background=True):
    """Define Jaccard index (multi-class) for tensors with 4 Dimensions (z, height, width, channels).
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
            skip_background (bool, optional): skip background label.
       Return:
            jac (tensor): Jaccard index value
    """
    # number of classes (last dimension of predictions)
    num_classes = tf.shape(y_pred)[-1]

    # one_hot representation of predicted segmentation
    y_pred_ = tf.cast(y_pred, dtype=tf.int32)
    y_pred_ = tf.one_hot(tf.math.argmax(y_pred_, axis=-1), num_classes, axis=-1)

    # one_hot representation of ground truth segmentation
    y_true_ = tf.cast(y_true[..., 0], dtype=tf.int32)
    y_true_ = tf.one_hot(y_true_, num_classes, axis=-1)
    if skip_background:
        y_pred_ = y_pred_[..., 1:]
        y_true_ = y_true_[..., 1:]

    TP = tf.math.count_nonzero(y_pred_ * y_true_)
    FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
    FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                  lambda: tf.cast(0.000, dtype='float64'))

    return jac


def conv3dblock(input, n_filters=16
                , kernel_size=(1,3, 3)
                , dropout=0.1
                , activation='relu'
                , **kwargs):
    x = Conv3D(n_filters, kernel_size, **kwargs)(input)
    if activation is not None:
        # x = BatchNormalization()(x)
        x = Activation(activation)(x)
    if dropout is not None and dropout > 0:
        x = Dropout(dropout)(x)
    return x


def conv2dlstm_block(input, n_filters=16, kernel_size=(3, 3)
                     , dropout=0.1, activation='relu'
                     , name='LSTMConv2D', **kwargs):
    x = ConvLSTM2D(filters=n_filters, kernel_size=kernel_size
                   , data_format='channels_last'
                   , recurrent_activation='hard_sigmoid'
                   , activation=None
                   , return_sequences=True
                   , name=name + '_convlstm'
                   , **kwargs)(input)
    # x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if dropout is not None and dropout > 0:
        x = Dropout(dropout)(x)
    x = conv3dblock(x, n_filters=n_filters, kernel_size=(1, kernel_size[0], kernel_size[1])
                    , dropout=dropout, activation=activation
                    , name=name + '_conv', **kwargs)
    return x


def UNetLSTM(input_size=(None, None), time_windows=5, kernel_size=(3, 3), pools=3
             , n_filters=16
             , activation='relu'
             , lr=0.001
             , dropout=0.0
             , kernel_initializer="he_uniform"
             , metrics='accuracy'
             , lossfunction='sparse_cross_entropy'
             , **kwargs):
    input_shape = (time_windows, input_size[0], input_size[1], 1)
    inputs = Input(shape=input_shape, name='trailer_input')
    skip_connections = []
    x = inputs
    # downsampling block
    for p in range(pools):
        x = conv2dlstm_block(x, n_filters=n_filters * (2 ** p)
                             , kernel_size=kernel_size
                             , dropout=dropout * p
                             , activation=activation
                             , name="encoder_layer_%s" % (p)
                             , **kwargs)
        skip_connections.append(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), padding='same'
                         , data_format='channels_last')(x)
    # bottelneck
    x = conv2dlstm_block(x, n_filters=n_filters * (2 ** pools)
                         , kernel_size=kernel_size
                         , dropout=dropout * p
                         , activation=activation
                         , name="bottleneck"
                         , **kwargs)
    # upsampling block
    for p in reversed(range(pools)):
        x = UpSampling3D(size=(1, 2, 2), data_format='channels_last')(x)
        x = Concatenate(axis=-1)([skip_connections[p], x])
        x = conv3dblock(x, n_filters=n_filters * (2 ** p)
                        , kernel_size=(1, kernel_size[0], kernel_size[1])
                        , dropout=dropout
                        , activation=activation
                        , name="decoder_layer_%s_conv_1" % (p)
                        , **kwargs)
        x = conv3dblock(x, n_filters=n_filters * (2 ** p)
                        , kernel_size=(1, kernel_size[0], kernel_size[1])
                        , dropout=dropout
                        , activation=activation
                        , name="decoder_layer_%s_conv_2" % (p)
                        , **kwargs)

    # Reduce time dimension to get the binary mask.
    x = AveragePooling3D(pool_size=(time_windows, 1, 1)
                         , padding='same'
                         , data_format='channels_last')(x)
    outputs = conv3dblock(x, n_filters=2
                          , kernel_size=(1, 1, 1)
                          , dropout=0
                          , activation=None
                          , name='last_convolution'
                          , **kwargs)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[jaccard_sparse3D, metrics])
    model.summary()
    print('U-Net with LSTM units was created with {} loss function'.format(lossfunction))
    return model
