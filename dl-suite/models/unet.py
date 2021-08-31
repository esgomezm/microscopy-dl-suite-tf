"""
Created on Tue March 31 2020-

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Conv2DTranspose, MaxPooling2D, Dropout, Lambda, UpSampling2D, Activation, Concatenate, multiply
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from internals.metrics import weighted_bce_dice_loss, bce_loss, identity_loss
from internals.metrics import jaccard_cce, jaccard_multiple_output, jaccard_sparse

def convolution_block(inputs, n_filters=16 , kernel_size=(3, 3), name='convolution_block', dropout=None, batch_norm=False, **kwargs):
    x = Conv2D(n_filters, kernel_size, name = name+'_1', **kwargs)(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    
    x = Conv2D(n_filters, kernel_size, name = name+'_2',  **kwargs)(x)
    if batch_norm:
        x = BatchNormalization()(x)
        
    if dropout is not None and dropout > 0:
        x = Dropout(dropout)(x)
    return x

def dilated_block(inputs, n_filters=16, kernel_size=(3, 3), name='dilated', dropout = None, batch_norm=False, dilation_rate=2, **kwargs):
    
    x = Conv2D(n_filters, kernel_size
               , name = name+'_1'
               , dilation_rate=dilation_rate
               , **kwargs)(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    
    x = Conv2D(n_filters, kernel_size, name = name+'_2'
               , dilation_rate=2*dilation_rate
               , **kwargs)(x)
    if batch_norm:
        x = BatchNormalization()(x)
        
    if dropout is not None and dropout > 0:
        x = Dropout(dropout)(x)
    return x

def categorical_unet_transpose(output_ch=2, kernel_size=(3, 3), pools=3, n_filters=16, activation='relu', lr=0.001
             , last_activation = 'sigmoid', category_weights = [1,1], dropout=0.0, kernel_initializer="he_uniform"
             , metrics='accuracy', lossfunction='sparse_cross_entropy', padding='same', **kwargs):
    """
    2D U-Net with transposed convolutions in the expanding/decoding path.
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
        x = Conv2DTranspose(n_filters * (2 ** (p+1)), (2, 2)
                            , strides=(2, 2)
                            , padding='same'
                            , kernel_initializer=kernel_initializer
                            , name="trasnpose_%s" % (p)
                            , **kwargs)(x)
        
        x = Concatenate([skip_connections[p], x], axis=-1)
        
        x = convolution_block(x, n_filters=n_filters * (2 ** p)
                        , kernel_size=kernel_size
                        , dropout=dropout * (p+1)
                        , activation=activation
                        , kernel_initializer=kernel_initializer
                        , name="decoder_layer_%s" % (p)
                        , padding=padding
                        , **kwargs)
    if lossfunction == "multiple_output":
        output_bg = Conv2D(1, (1, 1), padding=padding, kernel_initializer=kernel_initializer, name='bg')(x)
        output_fg = Conv2D(1, (1, 1), padding=padding, kernel_initializer=kernel_initializer, name='cell')(x)
        model = Model(inputs=inputs, outputs=[output_bg, output_fg])
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
              loss=[BinaryCrossentropy(from_logits=True), BinaryCrossentropy(from_logits=True)],
              loss_weights=category_weights,
              metrics=[[jaccard_multiple_output,"accuracy"], [jaccard_multiple_output,"accuracy", "mse"]]) 
    else:
        # no activation to feed the loss with the logits
        outputs = Conv2D(output_ch, (1, 1), padding=padding, kernel_initializer=kernel_initializer,name='output_logits')(x)

        if lossfunction == "weighted_bce_dice":
            # Activate the logits for the defined loss function
            outputs = Activation(last_activation,name='activated_output')(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=lr, name='adam'), 
                          loss=weighted_bce_dice_loss, metrics=[jaccard_cce, metrics])
        else:
            model = Model(inputs=inputs, outputs=outputs)
            if lossfunction == "sparse_cce":
                model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                      loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[jaccard_sparse, metrics])
            else:
                model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                              loss=CategoricalCrossentropy(from_logits=True),
                              metrics=[jaccard_cce, metrics])
    model.summary()
    print('U-Net model was created')
    return model

def categorical_unet_fc(output_ch=2, kernel_size=(3, 3), pools=3, n_filters=16, activation='relu', padding='same', lr=0.001
             , last_activation = 'sigmoid', category_weights=[1, 1], dropout=0.0, kernel_initializer="he_uniform"
             , metrics='accuracy', lossfunction='sparse_cross_entropy', **kwargs):
    """
    2D U-Net with trainable convolutions for the encoding and decoding paths (down/up-sampling).
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
                             , padding=padding
                             , kernel_initializer=kernel_initializer
                             , name="encoder_layer_%s" % (p)
                             , **kwargs)
        
        skip_connections.append(x)
        
        #substitude the max pooling with a convolution
        x = Conv2D(n_filters * (2 ** p), kernel_size
                   , strides=(2,2)
                   , activation=activation
                   , padding=padding
                   , kernel_initializer=kernel_initializer
                   , name="downsampling_conv_%s" % p)(x)
    # bottelneck
    x = convolution_block(x, n_filters=n_filters * (2 ** pools)
                         , kernel_size=kernel_size
                         , dropout=dropout * (p+1)
                         , activation=activation
                         , padding=padding
                         , kernel_initializer=kernel_initializer
                         , name="bottleneck"
                         , **kwargs)
    # upsampling block
    for p in reversed(range(pools)):
        x = Conv2DTranspose(n_filters * (2 ** (p+1)), (2, 2)
                            , strides=(2, 2)
                            , padding=padding
                            , kernel_initializer=kernel_initializer
                            , name="trasnpose_%s" % (p)
                            , **kwargs)(x)
        
        x = Concatenate([skip_connections[p], x], axis=-1)
        
        x = convolution_block(x, n_filters=n_filters * (2 ** p)
                        , kernel_size=kernel_size
                        , dropout=dropout * (p+1)
                        , activation=activation
                        , padding=padding
                        , kernel_initializer=kernel_initializer
                        , name="decoder_layer_%s" % (p)
                        , **kwargs)
    
    if lossfunction == "multiple_output":
        output_bg = Conv2D(1, (1, 1), name='bg')(x)
        output_fg = Conv2D(1, (1, 1), name='cell')(x)
        model = Model(inputs=inputs, outputs=[output_bg, output_fg])
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
              loss=[BinaryCrossentropy(from_logits=True), BinaryCrossentropy(from_logits=True)],
              loss_weights=category_weights,
              metrics=[[jaccard_multiple_output,"accuracy"], [jaccard_multiple_output,"accuracy", "mse"]]) 
    else:
        # no activation to feed the loss with the logits
        outputs = Conv2D(output_ch, (1, 1),name='output_logits')(x)

        if lossfunction == "weighted_bce_dice":
            # Activate the logits for the defined loss function
            outputs = Activation(last_activation,name='activated_output')(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=lr, name='adam'), 
                          loss=weighted_bce_dice_loss, metrics=[jaccard_cce, metrics])
        else:
            model = Model(inputs=inputs, outputs=outputs)
            if lossfunction == "sparse_cce":
                model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                      loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[jaccard_sparse, metrics])
            else:
                model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                              loss=CategoricalCrossentropy(from_logits=True),
                              metrics=[jaccard_cce, metrics])
    model.summary()
    print('U-Net model was created')
    return model

def categorical_unet_fc_dil(output_ch=2, kernel_size=(3, 3), pools=3, n_filters=16, activation='relu', padding='same', lr=0.001
             , last_activation = 'sigmoid', category_weights = [1,1], dropout=0.0, kernel_initializer="he_uniform"
             , metrics='accuracy', lossfunction='sparse_cross_entropy', dilation_rate=2, batch_norm=False, **kwargs):
    """
    2D U-Net with trainable convolutions for the encoding and decoding paths (down/up-sampling) and dilated convolutions
    in the encoding path.
    """
    inputs = Input((None, None, 1), name='inputs')
    skip_connections = []
    x = Conv2D(n_filters, kernel_size
                   , activation=activation
                   , padding=padding
                   , kernel_initializer=kernel_initializer
                   , name="first_conv")(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    # downsampling block
    for p in range(pools):
        x = dilated_block(x, n_filters=n_filters * (2 ** p)
                             , kernel_size=kernel_size
                             , dropout=dropout * (p+1)
                             , activation=activation
                             , padding=padding
                             , dilation_rate=dilation_rate
                             , kernel_initializer=kernel_initializer
                             , batch_norm=batch_norm
                             , name="dilated_%s" % (p)
                             , **kwargs)
        
        skip_connections.append(x)
        
        #substitude the max pooling with a convolution
        x = Conv2D(n_filters * (2 ** p), kernel_size
                   , strides=(2,2)
                   , activation=activation
                   , padding=padding
                   , kernel_initializer=kernel_initializer
                   , name="downsampling_conv_%s" % p)(x)
        
    # bottelneck
    x = convolution_block(x, n_filters=n_filters * (2 ** pools)
                         , kernel_size=(3,3)
                         , dropout=dropout * (p+1)
                         , activation=activation
                         , padding=padding
                         , kernel_initializer=kernel_initializer
                         , name="bottleneck")
    # upsampling block
    for p in reversed(range(pools)):
        x = Conv2DTranspose(n_filters * (2 ** (p+1)), (2, 2)
                            , strides=(2, 2)
                            , padding=padding
                            , kernel_initializer=kernel_initializer
                            , name="trasnpose_%s" % (p))(x)
        
        x = Concatenate([skip_connections[p], x], axis=-1)
        
        x = convolution_block(x, n_filters=n_filters * (2 ** p)
                        , kernel_size=(3, 3)
                        , dropout=dropout * (p+1)
                        , activation=activation
                        , padding=padding
                        , kernel_initializer=kernel_initializer
                        , name="decoder_layer_%s" % (p))
    
    if lossfunction == "multiple_output":
        output_bg = Conv2D(1, (1, 1), name='bg')(x)
        output_fg = Conv2D(1, (1, 1), name='cell')(x)
        model = Model(inputs=inputs, outputs=[output_bg, output_fg])
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
              loss=[BinaryCrossentropy(from_logits=True), BinaryCrossentropy(from_logits=True)],
              loss_weights=category_weights,
              metrics=[[jaccard_multiple_output,"accuracy"], [jaccard_multiple_output,"accuracy", "mse"]]) 
    else:
        # no activation to feed the loss with the logits
        outputs = Conv2D(output_ch, (1, 1),name='output_logits')(x)

        if lossfunction == "weighted_bce_dice":
            # Activate the logits for the defined loss function
            outputs = Activation(last_activation,name='activated_output')(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=lr, name='adam'), 
                          loss=weighted_bce_dice_loss, metrics=[jaccard_cce, metrics])
        else:
            model = Model(inputs=inputs, outputs=outputs)
            if lossfunction == "sparse_cce":
                model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                      loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[jaccard_sparse, metrics])
            else:
                model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                              loss=CategoricalCrossentropy(from_logits=True),
                              metrics=[jaccard_cce, metrics])
    model.summary()
    print('U-Net model was created')
    return model


def unet(n_filters=16, activation='relu', lr=0.0001,
         padding='same',
         dropout=0.0,
         last_activation='sigmoid',  # None
         kernel_initializer="he_uniform",  # "he_normal", "glorot_uniform"
         metrics="accuracy",
         lossfunction="binary_crossentropy"):
    """
  Weighted U-Net architecture.

  The tuple 'input_size' corresponds to the size of the input images and labels.
  Default value set to (512, 512, 1) (input images size is 512x512).
  """
    inputs = Input((None, None, 1), name='inputs')
    # Get weights.
    # weights = Input((mi, ni, 1))
    # # Get true masks
    # true_masks = Input((mi, ni, 1))

    c1 = Conv2D(n_filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(inputs)
    # c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)
    c1 = Conv2D(n_filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c1)
    # c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(n_filters * 2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(p1)
    # c2 = BatchNormalization()(c2)
    c2 = Dropout(dropout)(c2)
    c2 = Conv2D(n_filters * 2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c2)
    # c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(n_filters * (2 ** 2), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        p2)
    # c3 = BatchNormalization()(c3)
    c3 = Dropout(2 * dropout)(c3)
    c3 = Conv2D(n_filters * (2 ** 2), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c3)
    # c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(n_filters * (2 ** 3), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        p3)
    # c4 = BatchNormalization()(c4)
    c4 = Dropout(2 * dropout)(c4)
    c4 = Conv2D(n_filters * (2 ** 3), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c4)
    # c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(n_filters * (2 ** 4), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        p4)
    # c5 = BatchNormalization()(c5)
    c5 = Dropout(3 * dropout)(c5)
    c5 = Conv2D(n_filters * (2 ** 4), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c5)
    # c5 = BatchNormalization()(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate([u6, c4])
    c6 = Conv2D(n_filters * (2 ** 3), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        u6)
    # c6 = BatchNormalization()(c6)
    c6 = Dropout(2 * dropout)(c6)
    c6 = Conv2D(n_filters * (2 ** 3), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c6)
    # c6 = BatchNormalization()(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate([u7, c3])
    c7 = Conv2D(n_filters * (2 ** 2), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        u7)
    # c7 = BatchNormalization()(c7)
    c7 = Dropout(2 * dropout)(c7)
    c7 = Conv2D(n_filters * (2 ** 2), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c7)
    # c7 = BatchNormalization()(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate([u8, c2])
    c8 = Conv2D(n_filters * 2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(u8)
    # c8 = BatchNormalization()(c8)
    c8 = Dropout(dropout)(c8)
    c8 = Conv2D(n_filters * 2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c8)
    # c8 = BatchNormalization()(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate([u9, c1], axis=3)
    c9 = Conv2D(n_filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(u9)
    # c9 = BatchNormalization()(c9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv2D(n_filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c9)
    # c9 = BatchNormalization()(c9)

    outputs = Conv2D(1, (1, 1), name="output_logits")(c9)
    # loss = partial(loss_function, weights)

    if lossfunction == "weighted_bce_dice":
        # run activation of the output
        outputs = Activation(last_activation, name='activated_output')(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'), loss=weighted_bce_dice_loss, metrics=[jaccard_multiple_output, metrics])
    elif lossfunction == "weighted_bce":
        # run activation of the output
        outputs = Activation(last_activation, name='activated_output')(outputs)
        # Specify input (image + weights) and output.
        weights = Input((None, None, 1), name='weights')
        # Get true masks
        true_masks = Input((None, None, 1), name='masks')
        # Use a lambda layer to calculate the weighted loss
        # loss = Lambda(weighted_bce_loss)([true_masks, outputs, weights]) #output_shape=(256, 256, 1)
        bce_output = Lambda(bce_loss)([true_masks, outputs])  # output_shape=(256, 256, 1)
        wloss = multiply([bce_output, weights])
        model = Model(inputs=[inputs, weights, true_masks], outputs=wloss)
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                      loss=identity_loss)  # it makes no sense to calculate accuracy
    else:
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                      loss=BinaryCrossentropy(from_logits=True),
                      metrics=[jaccard_multiple_output, metrics])

    # Use a lambda layer to calculate the weighted loss
    # loss = Lambda(weighted_binary_loss, output_shape=(mi, ni, 1))([outputs, weights, true_masks])
    # model = Model(inputs=[inputs, weights, true_masks], outputs=loss)
    # model.compile(optimizer=Adam(learning_rate=lr, name='adam'), loss=identity_loss, metrics=['accuracy'])

    # Specify input (image + weights) and output.
    # model = Model(inputs=[inputs,weights], outputs=outputs)
    # model.compile(optimizer=Adam(learning_rate=lr,name='adam'), loss=binary_crossentropy_weighted(weights), metrics=['accuracy'])
    # model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print('U-Net model was created')
    return model

def categorical_unet(output_ch=2, n_filters=16, activation='relu', lr=0.0001,
         padding='same',
         dropout=0.0,
         last_activation='sigmoid',  # None
         kernel_initializer="he_uniform",  # "he_normal", "glorot_uniform"
         metrics="accuracy",
         lossfunction="categorical_crossentropy",
         category_weights=[1,1]):
    """
  Weighted U-Net architecture.

  The tuple 'input_size' corresponds to the size of the input images and labels.
  Default value set to (512, 512, 1) (input images size is 512x512).
  """
    inputs = Input((None, None, 1), name='inputs')
    
    c1 = Conv2D(n_filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(inputs)
    # c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)
    c1 = Conv2D(n_filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c1)
    # c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(n_filters * 2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(p1)
    # c2 = BatchNormalization()(c2)
    c2 = Dropout(dropout)(c2)
    c2 = Conv2D(n_filters * 2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c2)
    # c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(n_filters * (2 ** 2), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        p2)
    # c3 = BatchNormalization()(c3)
    c3 = Dropout(2 * dropout)(c3)
    c3 = Conv2D(n_filters * (2 ** 2), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c3)
    # c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(n_filters * (2 ** 3), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        p3)
    # c4 = BatchNormalization()(c4)
    c4 = Dropout(2 * dropout)(c4)
    c4 = Conv2D(n_filters * (2 ** 3), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c4)
    # c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(n_filters * (2 ** 4), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        p4)
    # c5 = BatchNormalization()(c5)
    c5 = Dropout(3 * dropout)(c5)
    c5 = Conv2D(n_filters * (2 ** 4), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c5)
    # c5 = BatchNormalization()(c5)

    # u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=padding) (c5)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate([u6, c4])
    c6 = Conv2D(n_filters * (2 ** 3), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        u6)
    # c6 = BatchNormalization()(c6)
    c6 = Dropout(2 * dropout)(c6)
    c6 = Conv2D(n_filters * (2 ** 3), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c6)
    # c6 = BatchNormalization()(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate([u7, c3])
    c7 = Conv2D(n_filters * (2 ** 2), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        u7)
    # c7 = BatchNormalization()(c7)
    c7 = Dropout(2 * dropout)(c7)
    c7 = Conv2D(n_filters * (2 ** 2), (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        c7)
    # c7 = BatchNormalization()(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate([u8, c2])
    c8 = Conv2D(n_filters * 2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(u8)
    # c8 = BatchNormalization()(c8)
    c8 = Dropout(dropout)(c8)
    c8 = Conv2D(n_filters * 2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c8)
    # c8 = BatchNormalization()(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate([u9, c1], axis=3)
    c9 = Conv2D(n_filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(u9)
    # c9 = BatchNormalization()(c9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv2D(n_filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c9)
    # c9 = BatchNormalization()(c9)
    # c9 = Conv2D(2, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c9) ## try to add this new line if it does not work properly
   
    # outputs = Conv2D(2, (1, 1), activation=last_activation)(c9)
    # model = Model(inputs=inputs, outputs=outputs)
    if lossfunction == "multiple_output":
        output_bg = Conv2D(1, (1, 1), name='bg')(c9)
        output_fg = Conv2D(1, (1, 1), name='cell')(c9)
        model = Model(inputs=inputs, outputs=[output_bg, output_fg])
        model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
              loss=[BinaryCrossentropy(from_logits=True), BinaryCrossentropy(from_logits=True)],
              loss_weights=category_weights,
              metrics=[[jaccard_multiple_output,"accuracy"], [jaccard_multiple_output,"accuracy", "mse"]]) 
    else:
        # no activation to feed the loss with the logits
        outputs = Conv2D(output_ch, (1, 1),name='output_logits')(c9)

        if lossfunction == "weighted_bce_dice":
            # Activate the logits for the defined loss function
            outputs = Activation(last_activation,name='activated_output')(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=lr, name='adam'), 
                          loss=weighted_bce_dice_loss, metrics=[jaccard_cce, metrics])
        else:
            model = Model(inputs=inputs, outputs=outputs)
            if lossfunction == "sparse_cce":
                model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                      loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[jaccard_sparse, metrics])
            else:
                model.compile(optimizer=Adam(learning_rate=lr, name='adam'),
                              loss=CategoricalCrossentropy(from_logits=True),
                              metrics=[jaccard_cce, metrics])
    model.summary()
    print('U-Net model was created')
    return model

