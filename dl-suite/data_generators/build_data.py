"""
Created on Wed January 22 2020

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""

from data_generators.initializers import training_videos, test_videos
from data_generators.initializers import training_contours, test_shots_contours
from data_generators.initializers import training_weightedmap, test_shots_weightedmap
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from utils.utils import read_input_image, read_input_videos, one_hot_it, read_instances


def load_val_data(config):
    files4test = os.listdir(config.VALPATH + '/inputs/')
    files4test.short()

    val_x = []
    val_y = []
    for f in files4test[:]:
        input_file = os.path.join(config.VALPATH, f)
        if config.cnn_name.__contains__('lstm'):
            x = read_input_videos(input_file, normalization=config.normalization)
        else:
            x = read_input_image(input_file, normalization=config.normalization)
        val_x.append(x)
        del x

        labels_file = 'instance_ids_' + f.split('_')[-1]
        y = sitk.ReadImage(os.path.join(config.VALPATH, 'labels', labels_file))
        y = sitk.GetArrayFromImage(y)
        y = (y > 0).astype(np.uint8)
        val_y.append(y)
        del y
    val_x = np.array(val_x)
    val_x = np.expand_dims(val_x, axis=-1)
    val_y = np.array(val_y)
    val_y = np.expand_dims(val_y, axis=-1)
    return val_x, val_y


def generate_data(config):
    if config.cnn_name.__contains__('lstm'):
    	training_generator = training_videos(config)
    	validation_generator = test_videos(config)

    else:
        if config.datagen_type.__contains__('contours'):
            training_generator = training_contours(config)
            validation_generator = test_shots_contours(config)
        else:
            training_generator = training_weightedmap(config)
            validation_generator = test_shots_weightedmap(config)
    return training_generator, validation_generator
