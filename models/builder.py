"""
Created on Wed January 22 2020

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""
from models.lstm import UNetLSTM
from models.unet import categorical_unet_transpose, categorical_unet_fc_dil, categorical_unet_fc, categorical_unet_fc, \
    categorical_unet, unet
from models.mobilenets import MobileNetV2_MobileUNet_base, MobileNetV2_MobileUNet_compile
from models.mobilenets_lstm import MobileNetV2_lstm_decoder, Recursive_MobileNetV2_MobileUnet_compile, \
    Recursive_MobileNetV2_MobileUnet_base


def build_model(config):
    # extract model parameters
    model_kwargs = {k[len('model_'):]: v for (k, v) in vars(config).items() if k.startswith('model_')}
    print(model_kwargs)

    # if config.cnn_name.__contains__('mobilenet_mobileunet_lstm_tips'):
    if config.cnn_name.__contains__('mobilenet_mobileunet_lstm'):
        keras_model = build_mobilenet_mobileunet_lstm(config, model_kwargs)
        # Convert to None to use it later
        config.train_pretrained_weights = "None"

    elif config.cnn_name.__contains__('mobilenet_mobileunet'):
        keras_model = build_mobilenet_mobileunet(config, model_kwargs)
        # Convert to None to use it later
        config.train_pretrained_weights = "None"

    elif config.cnn_name.__contains__('categorical_unet_transpose'):
        keras_model = categorical_unet_transpose(**model_kwargs)

    elif config.cnn_name.__contains__('categorical_unet_fc_dil'):
        keras_model = categorical_unet_fc_dil(**model_kwargs)

    elif config.cnn_name.__contains__('categorical_unet_fc'):
        keras_model = categorical_unet_fc(**model_kwargs)

    elif config.cnn_name.__contains__('categorical_unet'):
        keras_model = categorical_unet(**model_kwargs)
    else:
        keras_model = unet(**model_kwargs)

    if config.train_pretrained_weights != "None":
        print('Loading weights from {}'.format(config.train_pretrained_weights))
        keras_model.load_weights(config.train_pretrained_weights)
    return keras_model


def build_mobilenet_mobileunet(config, model_kwargs):
    if config.train_pretrained_weights != "None":
        if config.train_load_from_frozen == True and model_kwargs['train_decoder_only'] == False:
            keras_model = MobileNetV2_MobileUNet_base(n_filters=model_kwargs['n_filters'],
                                                      activation=model_kwargs['activation'],
                                                      dilation_rate=model_kwargs['dilation_rate'],
                                                      alpha=model_kwargs['mobile_alpha'],
                                                      dropout=model_kwargs['dropout'],
                                                      pools=model_kwargs['pools'],
                                                      train_decoder_only=1)  # load as non-trainable and then it can be trained.
            print('Loading weights from {}'.format(config.train_pretrained_weights))
            keras_model.load_weights(config.train_pretrained_weights)
            keras_model.trainable = True
            keras_model = MobileNetV2_MobileUNet_compile(keras_model, lr=model_kwargs['lr'])
        else:
            keras_model = MobileNetV2_MobileUNet_base(n_filters=model_kwargs['n_filters'],
                                                      activation=model_kwargs['activation'],
                                                      dilation_rate=model_kwargs['dilation_rate'],
                                                      alpha=model_kwargs['mobile_alpha'],
                                                      dropout=model_kwargs['dropout'],
                                                      pools=model_kwargs['pools'],
                                                      train_decoder_only=model_kwargs['train_decoder_only'])
            keras_model = MobileNetV2_MobileUNet_compile(keras_model, lr=model_kwargs['lr'])
            print('Loading weights from {}'.format(config.train_pretrained_weights))
            keras_model.load_weights(config.train_pretrained_weights)
    else:
        keras_model = MobileNetV2_MobileUNet_base(n_filters=model_kwargs['n_filters'],
                                                  activation=model_kwargs['activation'],
                                                  dilation_rate=model_kwargs['dilation_rate'],
                                                  alpha=model_kwargs['mobile_alpha'],
                                                  dropout=model_kwargs['dropout'],
                                                  pools=model_kwargs['pools'],
                                                  train_decoder_only=model_kwargs['train_decoder_only'])
        keras_model = MobileNetV2_MobileUNet_compile(keras_model, lr=model_kwargs['lr'])
    return keras_model


def build_mobilenet_mobileunet_lstm(config, model_kwargs):
    if config.train_pretrained_weights != "None":
        if config.train_load_from_frozen == True and model_kwargs['train_decoder_only'] == False:
            keras_model = Recursive_MobileNetV2_MobileUnet_base(n_filters=model_kwargs['n_filters'],
                                                                activation=model_kwargs['activation'],
                                                                dilation_rate=model_kwargs['dilation_rate'],
                                                                alpha=model_kwargs['mobile_alpha'],
                                                                dropout=model_kwargs['dropout'],
                                                                pools=model_kwargs['pools'],
                                                                train_decoder_only=1)  # load as non-trainable and then it can be trained.
            print('Loading weights from {}'.format(config.train_pretrained_weights))
            keras_model.load_weights(config.train_pretrained_weights)
            keras_model.trainable = True
            keras_model = Recursive_MobileNetV2_MobileUnet_compile(keras_model, lr=model_kwargs['lr'])
        else:
            keras_model = Recursive_MobileNetV2_MobileUnet_base(n_filters=model_kwargs['n_filters'],
                                                                activation=model_kwargs['activation'],
                                                                dilation_rate=model_kwargs['dilation_rate'],
                                                                alpha=model_kwargs['mobile_alpha'],
                                                                dropout=model_kwargs['dropout'],
                                                                pools=model_kwargs['pools'],
                                                                train_decoder_only=model_kwargs[
                                                                    'train_decoder_only'])
            keras_model = Recursive_MobileNetV2_MobileUnet_compile(keras_model, lr=model_kwargs['lr'])
            print('Loading weights from {}'.format(config.train_pretrained_weights))
            keras_model.load_weights(config.train_pretrained_weights)
    else:
        keras_model = Recursive_MobileNetV2_MobileUnet_base(n_filters=model_kwargs['n_filters'],
                                                            activation=model_kwargs['activation'],
                                                            dilation_rate=model_kwargs['dilation_rate'],
                                                            alpha=model_kwargs['mobile_alpha'],
                                                            dropout=model_kwargs['dropout'],
                                                            pools=model_kwargs['pools'],
                                                            train_decoder_only=model_kwargs['train_decoder_only'])
        keras_model = Recursive_MobileNetV2_MobileUnet_compile(keras_model, lr=model_kwargs['lr'])

    return keras_model
