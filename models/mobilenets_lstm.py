"""
Created on Wed February 8 2021

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, SeparableConv2D, ConvLSTM2D, \
    TimeDistributed, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
# from internals.metrics import jaccard_sparse
from models.lstm import jaccard_sparse3D
from models.mobilenets import MobileNetV2_MobileUNet_base


## Check out further explanation here:
# https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f
# https://github.com/keras-team/keras/issues/6449


def DepthwiseSeparableLSTM(inputs, n_filters, kernel=(3, 3), strides=(1, 1), dilation_rate=1, activation="relu",
                           return_sequences=True):
    x = TimeDistributed(
        SeparableConv2D(n_filters, kernel_size=kernel, strides=strides, dilation_rate=dilation_rate, padding="same",
                        activation=activation))(inputs)
    # no batch normalization as there will be no batch
    x = ConvLSTM2D(filters=n_filters, kernel_size=kernel
                   , padding='same'
                   , data_format='channels_last'
                   , dilation_rate=dilation_rate
                   , activation=activation
                   , return_sequences=return_sequences)(
        x)  # channels_last: (samples, timesteps, new_rows, new_cols, filters)
    return x


def build_mobilenet_encoder(alpha=0.35):
    encoder = MobileNetV2(input_shape=(None, None, 3), include_top=False, weights='imagenet', alpha=alpha, pools=5)
    ## To keep some layers frozen.
    # Keep 9 layers to train
    # trainable = 9
    # for layer in enocder.layers[:-trainable]:
    #     layer.trainable = False
    # for layer in enocder.layers[-trainable:]:
    #     layer.trainable = True
    # output = keras.GlobalMaxPool2D()
    # Use the activations of these layers
    skip_layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [encoder.get_layer(name).output for name in skip_layer_names[:pools]]
    return Model(inputs=encoder.input, outputs=layers)


def MobileNetV2_lstm_decoder(n_filters=32, activation='relu', alpha=0.35, dilation_rate=1, lr=0.001, pools=5):
    inputs = Input(shape=(None, None, None, 1), name="input_image")
    _inputs = Concatenate(axis=-1, name="rgb_input")([inputs, inputs, inputs])  ## MobileNet expects RGB images

    encoder = build_mobilenet_encoder(alpha=alpha, pools=pools)
    encoder_outputs = []
    for out in encoder.output:
        encoder_outputs.append(TimeDistributed(Model(encoder.input, out))(_inputs))

    f = [n_filters * (2 ** i) for i in range(6)]
    x = encoder_outputs[-1]
    x = DepthwiseSeparableLSTM(x, f[-1] * 2, kernel=(3, 3), strides=(1, 1), dilation_rate=dilation_rate,
                               activation=activation)

    for i in range(2, len(encoder_outputs) + 1, 1):
        # x_skip = encoder.get_layer(skip_layer_names[-i]).output
        x_skip = encoder_outputs[-i]
        x = TimeDistributed(UpSampling2D((2, 2)))(x)
        x = Concatenate()([x, x_skip])
        x = DepthwiseSeparableLSTM(x, f[-i], kernel=(3, 3), strides=(1, 1), dilation_rate=dilation_rate,
                                   activation=activation)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    # Now LSTM is different as it does not return the sequence and reduces the time dimenstion.
    x = DepthwiseSeparableLSTM(x, f[0], kernel=(3, 3), strides=(1, 1), dilation_rate=dilation_rate,
                               activation=activation, return_sequences=False)

    # Generate the last layers for the segmentation.
    # preserve the time dimension so we can reuse the rest of the code for lstm
    # x = tf.keras.layers.Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)
    x = tf.expand_dims(x, axis=1)
    seg = TimeDistributed(Conv2D(2, (1, 1), padding="same", name='slog'))(x)

    model = Model(inputs=inputs, outputs=seg)
    # E tensorflow/core/grappler/optimizers/meta_optimizer.cc:563] layout failed: Invalid argument: MutableGraphView::SortTopologically error: detected edge(s) creating cycle(s)
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

    model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[jaccard_sparse3D, "accuracy"])
    model.summary()
    print('U-Net with MobileNetV2 encoder and MobileDecoder with ConvLSTM2D for segmentation.')
    return model



def Recursive_MobileNetV2_MobileUnet_base(n_filters=32, activation='relu', dilation_rate=1, alpha=0.35, dropout=None,
                            pools=5, train_decoder_only=False):
    inputs = Input(shape=(None, None, None, 1), name="input_image")
    mobileunet = MobileNetV2_MobileUNet_base(n_filters=n_filters, activation=activation, dilation_rate=dilation_rate,
                                        alpha=alpha, dropout=dropout, pools=pools, train_decoder_only=train_decoder_only)
    # base_model = Model(mobileunet.input, mobileunet.get_layer('last_separable_2d').output)
    base_model = Model(mobileunet.input, mobileunet.get_layer(index=-2).output)
    x = TimeDistributed(base_model)(inputs)
    x = ConvLSTM2D(filters=n_filters, kernel_size=(3, 3)
                   , padding='same'
                   , data_format='channels_last'
                   , dilation_rate=dilation_rate
                   , activation=activation
                   , return_sequences=False)(x)  # channels_last: (samples, timesteps, new_rows, new_cols, filters)
    # seg = ConvLSTM2D(filters=2, kernel_size=(1, 1)
    #                , padding='same'
    #                , data_format='channels_last'
    #                , dilation_rate=dilation_rate
    #                , activation=None
    #                , return_sequences=False
    #                , name='slog')(x)  # channels_last: (samples, new_rows, new_cols, filters)
    x = tf.expand_dims(x, axis=1)
    seg = TimeDistributed(Conv2D(filters=2, kernel_size=(1, 1), padding="same", activation=None, name='slog'))(x)

    model = Model(inputs=inputs, outputs=seg)
    return model

def Recursive_MobileNetV2_MobileUnet_compile(model, lr=0.001):
    model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[jaccard_sparse3D, "accuracy"])
    model.summary()
    print('U-Net with MobileNetV2 encoder and MobileDecoder with ConvLSTM2D for segmentation.')
    return model

