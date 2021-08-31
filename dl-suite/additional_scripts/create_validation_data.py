"""
Created on Wed March 10 2021

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""
import os
import sys
sys.path.append('.')
import SimpleITK as sitk
import numpy as np
from utils.create_patches import random_crop, random_crop_complex

# Read the configuration file with all the metadata and information about the training.
VAL_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
video_length = 5

def create_time_window(video_file, labels_file, tips_name, video_length=5):
    """
    Args:
        x: path to the video file
        y: path to its labels

    Returns:
        sub_x: sub-satck from [t-self.time_window, t]
        sub_y: label of frame t.
    """
    x = sitk.ReadImage(video_file)
    x = sitk.GetArrayFromImage(x)
    y = sitk.ReadImage(labels_file)
    y = sitk.GetArrayFromImage(y)
    tips = sitk.ReadImage(tips_name)
    tips = sitk.GetArrayFromImage(tips)

    LENGTH = y.shape[0]
    t = np.random.randint(0, LENGTH - 1)
    sub_y = y[t]
    sub_tips = tips[t]
    del tips, y
    if t < video_length - 1:
        sub_x = np.zeros((video_length, x.shape[1], x.shape[2]), dtype=x.dtype)
        extra_frames = video_length - (t + 1)
        sub_x[extra_frames:] = x[:t + 1]
        for f in range(extra_frames):
            sub_x[f] = x[np.mod(extra_frames - f, t + 1)]
    else:
        sub_x = x[t - (video_length - 1):t + 1]
    sub_x = np.transpose(sub_x, [1, 2, 0])
    del x
    return sub_x, sub_y, sub_tips

# Create filders
if not os.path.exists(os.path.join(OUTPUT_PATH, 'inputs')):
    os.mkdir(os.path.join(OUTPUT_PATH, 'inputs'))
if not os.path.exists(os.path.join(OUTPUT_PATH, 'labels')):
    os.mkdir(os.path.join(OUTPUT_PATH, 'labels'))
if not os.path.exists(os.path.join(OUTPUT_PATH, 'keypoints')):
    os.mkdir(os.path.join(OUTPUT_PATH, 'keypoints'))
COUNT = 0
videos = os.listdir(os.path.join(VAL_PATH, 'inputs'))
videos.sort()
for v in videos:
    print(v)
    video_file = os.path.join(VAL_PATH, 'inputs', v)
    # labels_file = v.split('.')[0].split('stackreg_')[0] + v.split('.')[0].split('stackreg_')[-1] + "_Segmentationim-label.tif"
    # labels_file = os.path.join(VAL_PATH, 'labels', labels_file)
    # tips_name = os.path.join(VAL_PATH, 'keypoints', v)
    # # Load small videos
    # x, y, t = create_time_window(video_file, labels_file, tips_name, video_length=video_length)
    # x = x.astype(np.uint16)
    # # Get random crops
    # crop_size = (512, 512)
    # x, y, t, aux = random_crop_complex(x, y, t, t, crop_size, crop_size, pdf=500000)
    # del aux
    # x = np.transpose(x, [2, 0, 1])
    # # x = np.squeeze(x)
    # y = np.vstack([[y]] * video_length)
    # t = np.vstack([[t]] * video_length)

    labels_file = 'instance_ids_' + v.split('_')[-1]
    labels_file = os.path.join(VAL_PATH, 'labels', labels_file)
    tips_name = os.path.join(VAL_PATH, 'keypoints', labels_file)
    # Create frames from videos
    x = sitk.ReadImage(video_file)
    x = sitk.GetArrayFromImage(x)
    x = x[-1]
    y = sitk.ReadImage(labels_file)
    y = sitk.GetArrayFromImage(y)
    y = y[-1]
    t = sitk.ReadImage(tips_name)
    t = sitk.GetArrayFromImage(t)
    t = t[-1]

    print(x.shape)
    sitk.WriteImage(sitk.GetImageFromArray(x), os.path.join(OUTPUT_PATH, 'inputs', 'raw_{0:0>3}.tif'.format(int(COUNT))))
    sitk.WriteImage(sitk.GetImageFromArray(y), os.path.join(OUTPUT_PATH, 'labels', 'instance_ids_{0:0>3}.tif'.format(int(COUNT))))
    sitk.WriteImage(sitk.GetImageFromArray(t), os.path.join(OUTPUT_PATH, 'keypoints', 'instance_ids_{0:0>3}.tif'.format(int(COUNT))))
    # sitk.WriteImage(sitk.GetImageFromArray(x), os.path.join(OUTPUT_PATH, 'inputs', 'raw_{0:0>3}.tif'.format(int(COUNT))))
    # sitk.WriteImage(sitk.GetImageFromArray(y), os.path.join(OUTPUT_PATH, 'labels', 'instance_ids_{0:0>3}.tif'.format(int(COUNT))))
    # sitk.WriteImage(sitk.GetImageFromArray(t), os.path.join(OUTPUT_PATH, 'keypoints', 'instance_ids_{0:0>3}.tif'.format(int(COUNT))))
    print('Stored')
    COUNT += 1




