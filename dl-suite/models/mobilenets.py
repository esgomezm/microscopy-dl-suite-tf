"""
Created on Wed January 22 2021

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Activation, Concatenate, SeparableConv2D, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from internals.metrics import matlab_style_gauss2D, L1L2loss
from internals.metrics import jaccard_multiple_output, jaccard_sparse


def DepthwiseSeparableBlock2D(n_filters=16,
                              kernel=(3, 3),
                              strides=(1, 1),
                              dilation_rate=1,
                              activation="relu",
                              last=False,
                              dropout=None,
                              n_layers=2):
    """Convolutional layers
    Depthwise separable 2D Convolutions => BatchNormalization => Activation => Dropout (optional)
    Args:
        n_filters: number of filters
        kernel: filter size
        strides: strides for downsampling
        dilation_rate: dilated convolutions for increasing the receptive field.
        activation: activation function of each convolutional block
        dropout: If True, adds the dropout layer
        last: If true, the last convolutional block is named as last separable to identify it.
        n_layers: number of convolutional layers in block
    Returns:
    Convolutional block
    """
    result = tf.keras.Sequential()
    for i in range(n_layers):
        if last and i == n_layers:
            result.add(SeparableConv2D(n_filters, kernel, strides=strides, dilation_rate=dilation_rate, padding="same",
                                       name='last_separable_2d'))
        else:
            result.add(SeparableConv2D(n_filters, kernel, strides=strides, dilation_rate=dilation_rate, padding="same"))
        result.add(BatchNormalization())
        result.add(Activation(activation))
        if dropout is not None and dropout > 0:
            result.add(Dropout(dropout))

    return result


def upsample(**kwargs):
    """Upsamples an input.
    Depthwise separable block (2 layers) => Depthwise separable block (2 layers) => UpSampling
    Returns:
    Upsample Sequential Model
    """
    result = DepthwiseSeparableBlock2D(**kwargs)
    result.add(UpSampling2D((2, 2)))
    return result


def mobile_encoder(alpha, input_shape=(None, None, 3), pools=5):
    inputs = Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False,
                                                   alpha=alpha)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names[:pools]]
    # Create the feature extraction model
    encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)
    return encoder

def mobile_decoder(depth, n_filters, **kwargs):
    decoder = []
    for i in reversed(range(depth)):
        if i == depth-1:
            decoder.append(upsample(n_filters=n_filters * (2 ** i), last=True, **kwargs))
        else:
            decoder.append(upsample(n_filters=n_filters * (2 ** i), **kwargs))
    return decoder

def MobileNetV2_MobileUNet_base(n_filters=32, activation='relu', dilation_rate=1, alpha=0.35, dropout=None, pools=5,
                           train_decoder_only=False):
    # Input to the encoder
    inputs = Input(shape=(None, None, 1), name="input_image")
    _inputs = Concatenate(axis=-1, name="rgb_input")([inputs, inputs, inputs])  ## MobileNet expects RGB images

    # Load the encoder model
    encoder = mobile_encoder(alpha, pools=pools)
    if train_decoder_only:
        encoder.trainable = False
    else:
        encoder.trainable = True

    # Downsampling through the model and skip connections
    skips = encoder(_inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Load model decoder
    up_stack = mobile_decoder(len(encoder.output), n_filters, dilation_rate=dilation_rate,
                               activation=activation, dropout=dropout)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])
    x = UpSampling2D((2, 2), name='last_upsampling')(x)

    # Generate the last layers for the segmentation.
    last_layer = DepthwiseSeparableBlock2D(n_filters, kernel=(3, 3), dilation_rate=dilation_rate,
                                           activation=activation, last=True)
    seg = last_layer(x)
    seg = Conv2D(2, (1, 1), padding="same", name='slog')(seg)

    model = Model(inputs=inputs, outputs=seg)
    return model

def MobileNetV2_MobileUNet_compile(model, lr=0.001):
    model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[jaccard_sparse, "accuracy"])
    model.summary()
    print('U-Net with MobileNetV2 encoder and MobileDecoder for segmentation was created')
    return model


