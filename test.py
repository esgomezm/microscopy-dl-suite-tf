from utils.read_config import Dict2Obj
from utils.utils import stack2im
from internals.tiling_strategy import model_prediction, model_prediction_lstm
from internals.build_processed_videos import build_videos, build_videos_CTC
from models.builder import build_model

# import numpy as np
import sys
import os

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

if not config.cnn_name.__contains__('lstm'):
    # Convert test videos into single frames
    if not os.path.exists(config.newinfer_data):
        stack2im(config.newinfer_data[:-len("stack2im")])
        # stack2im('../data/test/')

## It is assumed that on each layer of the U-net there are 2 convolutions of 3x3. Hence:
# new_size = config.dim_size[0] # it is assumed to be a squared frame
# # number of maxpooling in the encoding part
# new_size=512
# downsamplings = 3
# # number opsampling in the decoding part
# upsamplings = 3
# # first convolution
# new_size = new_size - 4
# # Encoding
# for i in range(downsamplings):
#     aux = (new_size/2)-4
#     new_size = aux - np.mod(aux,2)
#     print(new_size)
# # Decoding
# for i in range(upsamplings):
#     new_size = new_size*2-4
#     print(new_size)
# # last convolution with 1 filter for the sigmoid activation
# new_size = new_size -2
# padding = (512-new_size)/2
# print(padding)
# padding = [(config.dim_size[0]-new_size)/2, (config.dim_size[0]-new_size)/2]

# Run on the test dataset

if hasattr(config, 'newinfer_output_folder_name'):
    output_dir = os.path.join(config.OUTPUTPATH, config.newinfer_output_folder_name)
else:
    output_dir = os.path.join(config.OUTPUTPATH, "newinfer_output")

if hasattr(config, 'newinfer_data'):
    TESTPATH = config.newinfer_data
else:
    TESTPATH = config.TESTPATH
# PATH2VIDEOS = os.path.join(config.PATH2VIDEOS, "videos2im_relation.csv")

# Process test images
# ----------------------------------------------
if config.cnn_name.__contains__('lstm'):
    model_prediction_lstm(TESTPATH, output_dir, config.PATH2VIDEOS, keras_model,
                          config.model_time_windows, dim_input=[960, 960], padding=config.newinfer_padding, normalization=config.normalization)
else:
    model_prediction(TESTPATH, output_dir, keras_model, dim_input=[960, 960], padding=config.newinfer_padding, normalization=config.normalization)

# Reconstruct videos
OUTPUTPATH = os.path.join(output_dir, 'reconstructed_videos')

build_videos(output_dir, OUTPUTPATH, config.PATH2VIDEOS)

# Save the results as in the CTC
OUTPUTPATH = os.path.join(output_dir, 'CTC_evaluation')
build_videos_CTC(output_dir, OUTPUTPATH, config.PATH2VIDEOS, threshold=0.5, tips=False)
