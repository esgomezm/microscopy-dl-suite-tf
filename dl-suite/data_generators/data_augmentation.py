"""
Created on Tue March 31 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.create_patches import random_crop, random_crop_complex

def data_augmentation(x, y, self):
    data_gen_args = dict(rotation_range=self.rotation_range,
                         width_shift_range=self.width_shift_range,
                         height_shift_range=self.height_shift_range,
                         shear_range=self.shear_range,
                         zoom_range=self.zoom_range,
                         horizontal_flip=self.horizontal_flip,
                         fill_mode=self.fill_mode)

    # Initialize Keras data augmentaiton
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(dtype='uint8', **data_gen_args)

    # Generate the same seed:
    seed = np.random.randint(10000000)

    x = x.reshape((1, x.shape[0], x.shape[1], 1))
    for batch in image_datagen.flow(x, batch_size=1, seed=seed):
        augmented_x = batch[0, :, :, 0]
        break
    del x
    # Apply the same data augmentation to the ground truth.
    y = y.reshape((1, y.shape[0], y.shape[1], 1))
    for batch in mask_datagen.flow(y, batch_size=1, seed=seed):
        augmented_y = batch[0, :, :, 0]
        break
    del y
    random_crop_size_input = (self.dim_input[0], self.dim_input[1])

    input_im, output_labels = random_crop(augmented_x, augmented_y,
                                          random_crop_size_input,
                                          pdf=self.pdf)

    return input_im, output_labels

def data_augmentation_weightedmaps(x, y, y_marks, y_weights, self):
      
    data_gen_args = dict(rotation_range=self.rotation_range,
                        width_shift_range=self.width_shift_range,
                        height_shift_range=self.height_shift_range,
                        shear_range=self.shear_range,
                        zoom_range=self.zoom_range,
                        horizontal_flip=self.horizontal_flip,
                        fill_mode=self.fill_mode) 

    # Initialize Keras data augmentaiton
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(dtype='uint8',**data_gen_args)
    weights_datagen = ImageDataGenerator(**data_gen_args)

    # Generate the same seed:
    seed = np.random.randint(10000000)       
   
    x = x.reshape((1, x.shape[0], x.shape[1], 1))
    for batch in image_datagen.flow(x,  batch_size=1, seed = seed):
        augmented_x = batch[0,:,:,0]
        break
    del x
    # Apply the same data augmentation to the ground truth.

    y = y.reshape((1, y.shape[0], y.shape[1], 1))
    for batch in mask_datagen.flow(y,  batch_size=1, seed = seed):
        augmented_y = batch[0,:,:,0]
        break
    del y

    y_marks = y_marks.reshape((1, y_marks.shape[0], y_marks.shape[1], 1))
    for batch in mask_datagen.flow(y_marks, batch_size=1, seed = seed):
        augmented_y_marks = batch[0,:,:,0]
        break
    del y_marks

    y_weights = y_weights.reshape((1, y_weights.shape[0], y_weights.shape[1], 1))
    for batch in weights_datagen.flow(y_weights,  batch_size=1, seed = seed):
        augmented_weights = batch[0,:,:,0]
        break
    del y_weights

    augmented_y = (augmented_y > 0).astype(np.uint8)
    augmented_y_marks = (augmented_y_marks > 0).astype(np.uint8)
    
    random_crop_size_input = (self.dim_input[0],  self.dim_input[1])
    random_crop_size_output = (self.dim_output[0],  self.dim_output[1])

    input_im,output_labels,output_marks,weights = random_crop_complex(augmented_x,augmented_y,augmented_y_marks,
                                                                      augmented_weights, random_crop_size_input,
                                                                      random_crop_size_output, pdf=self.pdf)

    return input_im, output_labels, output_marks, weights

def data_augmentation_time(x, y, self):
    """

    Args:
        x: 3D array (video) for which all the frames should suffer the same transformation (height, width, t).
        y: 2D array that should be also augmented.
        self: from augmentation class. It defined the augmentation parameters.

    Returns:
        A patch of the transformed video and the mask with size self.dim_inputs.

    """
    data_gen_args = dict(rotation_range=self.rotation_range,
                         width_shift_range=self.width_shift_range,
                         height_shift_range=self.height_shift_range,
                         shear_range=self.shear_range,
                         zoom_range=self.zoom_range,
                         horizontal_flip=self.horizontal_flip,
                         fill_mode=self.fill_mode)

    # Initialize Keras data augmentaiton
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Generate a random seed for the transformation of the same tuple:
    seed = np.random.randint(10000000)
    # Apply the same data augmentation to all the frames in the video
    augmented_x = np.zeros(x.shape)
    for t in range(x.shape[-1]):
        x_t = x[:,:,t]
        x_t = x_t.reshape((1, x_t.shape[0], x_t.shape[1], 1))
        for batch in image_datagen.flow(x_t, batch_size=1, seed=seed):
            augmented_x[:,:,t] = batch[0, :, :, 0]
            break
    del x, x_t
    # Apply the same data augmentation to the ground truth.
    # GROUND TRUTH IS ASSUMED TO BE 2D
    aux_y = np.copy(y)
    aux_y = aux_y.reshape((1, y.shape[0], y.shape[1], 1))
    for batch in mask_datagen.flow(aux_y, batch_size=1, seed=seed):
        augmented_y = batch[0, :, :, 0]
        break
    del y
    augmented_y[augmented_y > 0] = 1
    crop_size = (self.dim_input[0], self.dim_input[1])
    input_im, output_labels = random_crop(augmented_x, augmented_y, crop_size, pdf=self.pdf)

    return input_im, output_labels

def data_augmentation_contours(x, y, c, self):
    data_gen_args = dict(rotation_range=self.rotation_range,
                         width_shift_range=self.width_shift_range,
                         height_shift_range=self.height_shift_range,
                         shear_range=self.shear_range,
                         zoom_range=self.zoom_range,
                         horizontal_flip=self.horizontal_flip,
                         fill_mode=self.fill_mode)

    # Initialize Keras data augmentaiton
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(dtype='uint8', **data_gen_args)

    # Generate the same seed:
    seed = np.random.randint(10000000)

    x = x.reshape((1, x.shape[0], x.shape[1], 1))
    for batch in image_datagen.flow(x, batch_size=1, seed=seed):
        augmented_x = batch[0, :, :, 0]
        break
    del x
    # Apply the same data augmentation to the masks and contours
    y = y.reshape((1, y.shape[0], y.shape[1], 1))
    for batch in mask_datagen.flow(y, batch_size=1, seed=seed):
        augmented_y = batch[0, :, :, 0]
        break
    del y
    c = c.reshape((1, c.shape[0], c.shape[1], 1))
    for batch in mask_datagen.flow(c, batch_size=1, seed=seed):
        augmented_c = batch[0, :, :, 0]
        break
    del c
    # Create a single mask with the contours and segmentations
    augmented_c = (augmented_c > 0)
    augmented_y = (augmented_y > 0).astype(np.uint8)
    augmented_y[augmented_c] = 2

    random_crop_size_input = (self.dim_input[0], self.dim_input[1])

    input_im, output_labels = random_crop(augmented_x, augmented_y,
                                          random_crop_size_input,
                                          pdf=self.pdf)

    return input_im, output_labels
