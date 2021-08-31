"""
Created on Wed October 15 2020

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Activation, Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

from internals.metrics import matlab_style_gauss2D, L1L2loss
from internals.metrics import jaccard_multiple_output, jaccard_sparse
from models.unet import convolution_block

def unet4tips(output_ch=2, kernel_size=(3, 3), pools=3, n_filters=16, activation='relu', lr=0.001
                    , last_activation='sigmoid', category_weights=[1, 1], dropout=0.0, input_shape=(512, 512, 1),
                    kernel_initializer="he_uniform", padding='same',
                    metrics="accuracy", lossfunction='sparse_cross_entropy', loss_tips='mse', **kwargs):
    """
    2D U-Net with multiple output: segmentation + localization
    """
    inputs = Input((None, None, 1), name='inputs')
    skip_connections = []
    x = inputs
    # downsampling block
    for p in range(pools):
        x = convolution_block(x, n_filters=n_filters * (2 ** p)
                              , kernel_size=kernel_size
                              , dropout=dropout * (p+1)
                              , activation=activation
                              , kernel_initializer=kernel_initializer
                              , name="encoder_layer_%s" % (p)
                              , padding=padding
                              , **kwargs)

        skip_connections.append(x)

        x = MaxPooling2D(pool_size=(2, 2), padding=padding)(x)
    # bottelneck
    x = convolution_block(x, n_filters=n_filters * (2 ** pools)
                          , kernel_size=kernel_size
                          , dropout=dropout * (p+1)
                          , activation=activation
                          , kernel_initializer=kernel_initializer
                          , name="bottleneck"
                          , padding=padding
                          , **kwargs)
    # upsampling block
    for p in reversed(range(pools)):
        x = UpSampling2D((2, 2), name="up_%s" % (p))(x)
        x = Concatenate([skip_connections[p], x], axis=-1)

        x = convolution_block(x, n_filters=n_filters * (2 ** p)
                              , kernel_size=kernel_size
                              , dropout=dropout * (p+1)
                              , activation=activation
                              , kernel_initializer=kernel_initializer
                              , name="decoder_layer_%s" % (p)
                              , padding=padding
                              , **kwargs)
    seg = Conv2D(3, (3, 3), padding=padding, kernel_initializer=kernel_initializer, name='seg_last')(x)
    output_seg = Conv2D(output_ch, (1, 1), padding=padding, kernel_initializer=kernel_initializer, name='slog')(seg)

    tips = Conv2D(3, (3, 3), padding=padding, kernel_initializer=kernel_initializer, name='tips_last')(x)
    tips = Conv2D(1, (1, 1), padding=padding, kernel_initializer=kernel_initializer, name='tips_logits')(tips)
    output_tips = Activation(last_activation, name='tips')(tips)

    model = Model(inputs=inputs, outputs=[output_seg, output_tips])

    if loss_tips.__contains__('L1L2'):
        psf_heatmap = matlab_style_gauss2D(shape=(15, 15), sigma=4)
        gfilter = tf.reshape(psf_heatmap, [15, 15, 1, 1])
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                loss=[SparseCategoricalCrossentropy(from_logits=True), L1L2loss(input_shape, gfilter)],
                loss_weights=category_weights,
                metrics=[[jaccard_sparse, metrics], [jaccard_multiple_output, metrics, "mse"]])
    else:
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                      loss=[SparseCategoricalCrossentropy(from_logits=True), MeanSquaredError()],
                      loss_weights=category_weights,
                      metrics=[[jaccard_sparse, metrics], [jaccard_multiple_output, metrics]])
    model.summary()
    print('U-Net model was created')
    return model
