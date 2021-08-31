"""
Created on Tue Apr 7 2020

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""

import os
import sys
from internals.callbacks import initiate_callbacks, ImagesTensorboardCallback
from utils.read_config import Dict2Obj
from internals.build_processed_videos import build_videos_CTC
from internals.tiling_strategy import model_prediction, model_prediction_lstm
from models.builder import build_model
from data_generators.build_data import generate_data
from internals.build_processed_videos import build_videos

# Read the configuration file with all the metadata and information about the training.
PATH2CONFIG = sys.argv[1]
# PATH2CONFIG = 'trained_config/config_docker_local.json'

config = Dict2Obj(PATH2CONFIG)
keras_model = build_model(config)
training_generator, validation_generator = generate_data(config)





# Define callbacks and load pretrained weights
# ----------------------------------------------
# Create some shots for Tensorboard
B = validation_generator.batch_size
validation_generator.batch_size = 10
val_x, val_y = validation_generator.__getitem__(0)
validation_generator.batch_size = B

Itb = ImagesTensorboardCallback(val_x, val_y, os.path.join(config.OUTPUTPATH, 'logs/tmp/'), n_images=10, step=20)
del val_x, val_y

if config.train_pretrained_weights != "None":
    last_epoch = config.train_pretrained_weights
    if last_epoch.__contains__('/'):
        last_epoch = last_epoch.split('/')[-1]
    last_epoch = last_epoch.split('.')[0]
    callbacks = initiate_callbacks(config, keras_model, last_epoch=last_epoch)
else:
    callbacks = initiate_callbacks(config, keras_model)
callbacks.append(Itb)




# Train the model
# ----------------------------------------------
keras_model.fit(training_generator,validation_data=validation_generator, # validation_data=(val_x, val_y),
                epochs=config.train_max_epochs,
                validation_batch_size=config.datagen_batch_size,#validation_steps=config.datagen_batch_size,
                callbacks=callbacks)
keras_model.save_weights(os.path.join(config.OUTPUTPATH, 'checkpoints', config.cnn_name + 'last.hdf5'))





# Process test images
# ----------------------------------------------
output_dir = os.path.join(config.OUTPUTPATH, "test_output")
# PATH2VIDEOS = os.path.join(config.TESTPATH, "videos2im_relation.csv")

if config.cnn_name.__contains__('lstm'):
    model_prediction_lstm(config.TESTPATH, output_dir, config.PATH2VIDEOS, keras_model,
                          config.model_time_windows, dim_input=[960, 960], padding=config.newinfer_padding, normalization=config.normalization)
else:
    model_prediction(config.TESTPATH, output_dir, keras_model, dim_input=[960, 960], padding=config.newinfer_padding, normalization=config.normalization)




# Reconstruct videos
# ----------------------------------------------
OUTPUTPATH = os.path.join(output_dir, 'reconstructed_videos')
build_videos(output_dir, OUTPUTPATH, config.PATH2VIDEOS)
# Save the results as in the CTC
OUTPUTPATH = os.path.join(output_dir, 'CTC_evaluation')
build_videos_CTC(output_dir, OUTPUTPATH, config.PATH2VIDEOS, threshold=0.5, tips=False)
