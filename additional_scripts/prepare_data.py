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
    preprocess_data.py --data-dir=<data_dir>
"""
# import sys
from utils.utils import stack2im, do_save_wm, do_save_marks

# PATH = sys.argv[1]

# Convert videos to images
# PATH = "/home/esgomezm/Documents/3D-PROTUCEL/data/test/"
# stack2im(PATH)
# PATH = "/home/esgomezm/Documents/3D-PROTUCEL/data/train/"
# stack2im(PATH)
# PATH = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/test/"
PATH = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/praful/"
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


## Change the name of the data to the CTC format.
import os
import cv2
import numpy as np

from data.data_handling import read_instances
##
##

import os
import SimpleITK as sitk
import sys
import numpy as np

INPUTPATH_L = '/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/test/stack2im/labels/'
# INPUTPATH_C = '/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/test/stack2im/contours/'
PATH2VIDEOS = '/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/test/stack2im/videos2im_relation.csv'
# OUTPUTPATH = '/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/test/stack2im/CTCbin_contours/'
OUTPUTPATH = '/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/test/stack2im/CTCbin/'

files = [x for x in open(PATH2VIDEOS, "r")]
files = files[1:]  # First row contains labels
file_relation = [[x.split(';')[0], x.split(';')[1][:-1]] for x in files]

if not os.path.exists(OUTPUTPATH):
    os.makedirs(OUTPUTPATH)
COUNT = 1
SEQ = 1
while COUNT <= len(file_relation):
    # Get the name of the original videos and the number of frames that it contains
    file_name = file_relation[COUNT][0]  # video name
    # Calculate how many frames you need to process (it is said in the name of the video)
    start_time, end_time = file_name.split('_')[-1].split('-')
    start_time = np.int(start_time)
    end_time = np.int(end_time)
    sys.stdout.write("\rProcessing video {0}:\n".format(file_name))
    seqName = os.path.join(OUTPUTPATH,'{0:0>2}_GT'.format(SEQ), 'SEG')
    # Create the directory to store the results
    if not os.path.exists(seqName):
        os.makedirs(seqName)
    # Process the frames of this video
    seqCount = 0
    for i in range(end_time - start_time + 1):
        # Load the image prediction
        frame_name = os.path.join(INPUTPATH_L, 'instance_ids_{0:0>3}.tif'.format(int(COUNT + i)))
        frame = sitk.ReadImage(frame_name)
        frame = sitk.GetArrayFromImage(frame)
        frame = frame > 0
        frame = frame.astype(np.uint16)

        # frame_name = os.path.join(INPUTPATH_C, 'instance_ids_{0:0>3}.tif'.format(int(COUNT + i)))
        # boundary = sitk.ReadImage(frame_name)
        # boundary = sitk.GetArrayFromImage(boundary)
        # boundary = boundary > 0
        # frame[boundary] = 2

        # Save the binary mask in the new sequence folder
        sitk.WriteImage(sitk.GetImageFromArray(frame),
                        os.path.join(seqName, 'man_seg{0:0>3}.tif'.format(seqCount)))
        # Update the counter
        seqCount += 1
        progress = (i + 1) / (end_time - start_time + 1)
        text = "\r[{0}] {1}%".format("-" * (i + 1) + " " * (end_time - start_time - i), progress * 100)
        sys.stdout.write(text)
        sys.stdout.flush()
    # Update counters
    COUNT = COUNT + i + 1
    SEQ += 1
    sys.stdout.write("\n")  # this ends the progress bar


# # Remove contours from labels
# import SimpleITK as sitk
# import os
# import numpy as np
# # PATH = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/DATA/data_corrected/test/stack2im"
# PATH = "/home/esgomezm/Documents/3D-PROTUCEL/data/Usiigaci/data2convine"
# labels = os.path.join(PATH, 'labels_original')
# contours = os.path.join(PATH, 'contours')
# OUT = os.path.join(PATH, 'labels')
# if not os.path.isdir(OUT):
#     os.mkdir(OUT)
# for i in os.listdir(labels):
#     L = sitk.ReadImage(os.path.join(labels, i))
#     L = sitk.GetArrayFromImage(L)
#     L[L>0] = 1
#     C = sitk.ReadImage(os.path.join(contours, i))
#     C = sitk.GetArrayFromImage(C)
#     C[C>0] = 1
#     L[C==1] = 0
#     sitk.WriteImage(sitk.GetImageFromArray(L.astype(np.uint8)),
#                     os.path.join(OUT,i))