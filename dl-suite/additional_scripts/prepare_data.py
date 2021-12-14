"""
Created on Tue Apr 7 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm

Script that converts the videos into single frames to train segmentation.
Creates a new folder in the given directory as follows:

Directory structre:
stack2im
    weights
        instance_ids_001_weight.tif
        instance_ids_002_weight.tif
        ...
    labels
        instance_ids_001.tif
        instance_ids_002.tif
        ...
    inputs
        raw_001.tif
        raw_002.tif
        ...

Usage:
    preprocess_data.py <data_dir>
"""
import sys
from utils.utils import stack2im, do_save_wm, do_save_marks
PATH = sys.argv[1]
stack2im(PATH, END=None, keypoints=True, COUNT=729)

## Generate weights
# PATH2WEIGHTS = PATH + '/stack2im/'
# do_save_wm(PATH2WEIGHTS, blur_kernel_size=9, w0=10, only_contours = False)

## Generate marks and store them
# do_save_marks(PATH)

## Run it from the editor:
# from MU_Lux_CZ.data.data_handling import stack2im, do_save_wm
# PATH = "/home/esgomezm/Documents/3D-PROTUCEL/data/train/"
# stack2im(PATH)
# # Generate weights
# PATH2WEIGHTS = PATH + '/stack2im/'
# do_save_wm(PATH2WEIGHTS)

