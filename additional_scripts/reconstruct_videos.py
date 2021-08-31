from utils.read_config import Dict2Obj
from utils.utils import stack2im
from internals.build_processed_videos import build_videos, get_accuracy_measures_videos
# import numpy as np
import sys
import os

PATH2CONFIG = sys.argv[1]
config = Dict2Obj(PATH2CONFIG)

# Convert test videos into single frames
if not os.path.exists(config.newinfer_data):
    stack2im(config.newinfer_data[:-len("stack2im")])
    # stack2im('../data/test/')

# Run on the test dataset
# output_dir = 'binary_bc_dice_weighted_001/'
if hasattr(config, 'newinfer_output_folder_name'):
    output_dir = os.path.join(config.OUTPUTPATH, config.newinfer_output_folder_name)
else:
    output_dir = os.path.join(config.OUTPUTPATH, "newinfer_output")

OUTPUTPATH = os.path.join(output_dir, 'reconstructed_videos')
PATH2VIDEOS = config.PATH2VIDEOS
PATH2GT = os.path.join(config.newinfer_data, 'labels')
PATH2STORE = output_dir
EXPERIMENT = config.OUTPUTPATH
# Reconstruct videos
build_videos(output_dir, OUTPUTPATH, PATH2VIDEOS)
# Get accuracy measures
get_accuracy_measures_videos(output_dir, PATH2GT, PATH2STORE, PATH2VIDEOS, EXPERIMENT)

