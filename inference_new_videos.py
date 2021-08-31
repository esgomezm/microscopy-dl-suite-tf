from utils.read_config import Dict2Obj
from models.builder import build_model
import numpy as np
import sys
import os
import SimpleITK as sitk
from internals.process_full_videos import process_video

PATH2CONFIG = sys.argv[1]
config = Dict2Obj(PATH2CONFIG)

# Load the model
# ----------------------------------------------
# Choose the model to process the data
# model_name = config.cnn_name + "{epoch:0>5}.hdf5".format(epoch=config.newinfer_epoch_process_test)
model_name = config.newinfer_epoch_process_test
print(model_name)
# Obtaining network architecture and load the weights of the trained model
config.train_pretrained_weights = os.path.join(config.OUTPUTPATH, 'checkpoints/', model_name)
keras_model = build_model(config)

# Set up paths
# ----------------------------------------------
if hasattr(config, 'newinfer_output_folder_name'):
    output_dir = os.path.join(config.OUTPUTPATH, config.newinfer_output_folder_name)
else:
    output_dir = os.path.join(config.OUTPUTPATH, "newinfer_output")

if hasattr(config, 'newinfer_data'):
    TESTPATH = config.newinfer_data
else:
    TESTPATH = config.TESTPATH

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the names of the videos to analyze
names = os.listdir(TESTPATH)
names.sort()

# Run inference
# ----------------------------------------------
for i, video_name in enumerate(names):
    input_path = os.path.join(TESTPATH, video_name)
    if type(keras_model.output_shape) is list:
        MASK, TIPS = process_video(input_path, keras_model,
                                   halo=config.newinfer_padding,
                                   step=[2**config.model_pools, 2**config.model_pools],
                                   batch_size=config.datagen_batch_size,
                                   normalization=config.normalization,
                                   time_window=config.model_time_windows)
        sitk.WriteImage(sitk.GetImageFromArray(MASK), os.path.join(output_dir, video_name.split('.')[0] + '.tif'))
        sitk.WriteImage(sitk.GetImageFromArray(TIPS), os.path.join(output_dir, video_name.split('.')[0] + '_tips.tif'))
        del MASK, TIPS
    else:
        MASK = process_video(input_path, keras_model,
                                   halo=config.newinfer_padding,
                                   step=[2 ** config.model_pools, 2 ** config.model_pools],
                                   batch_size=config.datagen_batch_size,
                                   normalization=config.normalization,
                                   time_window=config.model_time_windows)
        sitk.WriteImage(sitk.GetImageFromArray(MASK), os.path.join(output_dir, video_name.split('.')[0] + '.tif'))
