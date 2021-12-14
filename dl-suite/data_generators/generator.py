"""
Created on Mon Oct 5 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import numpy as np
import cv2
# from skimage.segmentation import find_boundaries
import SimpleITK as sitk
import tensorflow.keras
from utils.utils import read_input_image, read_input_videos, one_hot_it, read_instances
from utils.create_patches import random_crop, random_crop_complex
from data_generators.data_augmentation import data_augmentation, data_augmentation_time, \
    data_augmentation_weightedmaps, data_augmentation_contours
from scipy.ndimage import gaussian_filter


## Define DataGenerator class
class BaseGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, list_IDs, dataset_path, rotation_range,
                 width_shift_range, height_shift_range, shear_range,
                 zoom_range, horizontal_flip, fill_mode, dim_input, normalization,
                 pdf=1, module="train", patch_batch=1, batch_size=5, shuffle=True):
        """
        Suffle is used to take everytime a different
        sample from the list in a random way so the
        training order differs. We create two instances
        with the same arguments.
        """
        self.list_IDs = list_IDs
        self.dataset_path = dataset_path
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.fill_mode = fill_mode
        self.pdf = pdf
        self.module = module  # "train" or "test"
        # self.weights_loss = weights_loss # weighted_bce_dice, weighted_bce or a keras loss function("binary_crossentropy").
        self.dim_input = dim_input
        self.patch_batch = patch_batch
        self.batch_size = batch_size
        self.shuffle = shuffle  # #
        self.on_epoch_end()
        self.normalization = normalization

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class DataGenerator(BaseGenerator):
    """
    The ground truth is created with one channel so use mse, bce or sparse-cce
    """

    def __init__(self, list_IDs, dataset_path, rotation_range,
                 width_shift_range, height_shift_range, shear_range,
                 zoom_range, horizontal_flip, fill_mode, normalization, dim_input=None,
                 pdf=1, module="train", patch_batch=1, batch_size=5, shuffle=True):
        BaseGenerator.__init__(self, list_IDs, dataset_path, rotation_range,
                               width_shift_range, height_shift_range, shear_range,
                               zoom_range, horizontal_flip, fill_mode, dim_input, normalization,
                               pdf=pdf, module=module, patch_batch=patch_batch, batch_size=batch_size, shuffle=shuffle)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size * self.patch_batch, self.dim_input[0], self.dim_input[1], 1))
        y = np.empty((self.batch_size * self.patch_batch, self.dim_input[0], self.dim_input[1], 1))
        # Generate data
        j = 0
        if self.module == 'train':
            for patch in range(self.patch_batch):
                for i, ID in enumerate(list_IDs_temp):
                    ID = ID.split('_')[-1].split('.')[0]
                    aux_x = read_input_image("{0}/inputs/raw_{1}.tif".format(self.dataset_path, ID),
                                             normalization=self.normalization)
                    aux_y = cv2.imread("{0}/labels/instance_ids_{1}.tif".format(self.dataset_path, ID)
                                       , cv2.IMREAD_ANYDEPTH)
                    # Transform the data
                    augmented_x, augmented_y = data_augmentation(aux_x, aux_y, self)
                    x[j, :, :, 0] = augmented_x
                    # Store ground truth
                    y[j, :, :, 0] = augmented_y
                    j += 1

        elif self.module == 'test':
            # print("Creating validation data...")
            for patch in range(self.patch_batch):
                for i, ID in enumerate(list_IDs_temp):
                    ID = ID.split('_')[-1].split('.')[0]
                    aux_x = read_input_image("{0}/inputs/raw_{1}.tif".format(self.dataset_path, ID),
                                             normalization=self.normalization)
                    aux_y = cv2.imread("{0}/labels/instance_ids_{1}.tif".format(self.dataset_path, ID)
                                       , cv2.IMREAD_ANYDEPTH)
                    # Get a random crop without transformation
                    aux_x, aux_y = random_crop(aux_x, aux_y,
                                               (self.dim_input[0],
                                                self.dim_input[1]),
                                               pdf=self.pdf)
                    x[j, :, :, 0] = aux_x
                    y[j, :, :, 0] = aux_y
                    j += 1
        x = x.astype(np.float32)
        y = y.astype(np.uint8)
        return x, y


class DataGeneratorLSTM(BaseGenerator):
    def __init__(self, list_IDs, dataset_path, rotation_range,
                 width_shift_range, height_shift_range, shear_range,
                 zoom_range, horizontal_flip, fill_mode, normalization, dim_input=None,
                 pdf=1, module="train", patch_batch=1, batch_size=5, shuffle=True, video_length=None,
                 weights_loss="binary_crossentropy"):
        BaseGenerator.__init__(self, list_IDs, dataset_path, rotation_range,
                               width_shift_range, height_shift_range, shear_range,
                               zoom_range, horizontal_flip, fill_mode, dim_input, normalization,
                               pdf=pdf, module=module, patch_batch=patch_batch,
                               batch_size=batch_size, shuffle=shuffle)
        if video_length == None:
            video_length = 5
        self.video_length = video_length
        self.weights_loss = weights_loss

    def create_time_window(self, video_file, labels_file):
        """
        Args:
            x: path to the video file
            y: path to its labels

        Returns:
            sub_x: sub-satck from [t-self.time_window, t]
            sub_y: label of frame t.
        """
        x = read_input_videos(video_file, normalization=self.normalization)
        y = sitk.ReadImage(labels_file)
        y = sitk.GetArrayFromImage(y)
        LENGTH = y.shape[0]
        t = np.random.randint(0, LENGTH - 1)
        t=1
        sub_y = y[t]
        if t < self.video_length - 1:
            sub_x = np.zeros((self.video_length, x.shape[1], x.shape[2]))
            extra_frames = self.video_length - (t + 1)
            sub_x[extra_frames:] = x[:t + 1]
            for f in range(extra_frames):
                sub_x[f] = x[np.mod(extra_frames - f, t + 1)]
        else:
            sub_x = x[t - (self.video_length - 1):t + 1]
        sub_x = np.transpose(sub_x, [1, 2, 0])
        return sub_x, sub_y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def __data_generation(self, list_IDs_temp):
        x = np.empty((self.batch_size * self.patch_batch,
                      self.video_length, self.dim_input[0],
                      self.dim_input[1], 1))
        if self.weights_loss == 'categorical_crossentropy':
            y = np.empty((self.batch_size * self.patch_batch, 1, self.dim_input[0], self.dim_input[1], 2))
        else:
            y = np.empty((self.batch_size * self.patch_batch, 1, self.dim_input[0], self.dim_input[1], 1))
        # Generate data
        j = 0
        if self.module == 'train':
            for patch in range(self.patch_batch):
                for i, ID in enumerate(list_IDs_temp):
                    video_name = ID.split('.')[0]
                    labels_name = video_name
                    video_name = "{0}/inputs/{1}.tif".format(self.dataset_path, video_name)
                    labels_name = "{0}/labels/{1}.tif".format(self.dataset_path, labels_name)
                    # Create a substack
                    aux_x, aux_y = self.create_time_window(video_name, labels_name)
                    augmented_x, augmented_y = data_augmentation_time(aux_x, aux_y, self)
                    augmented_x = np.transpose(augmented_x, [2, 0, 1])
                    x[j, ..., 0] = augmented_x
                    # Store ground truth
                    augmented_y[augmented_y > 0] = 1
                    # Check if one hot representation is needed
                    if self.weights_loss == 'categorical_crossentropy':
                        augmented_y = one_hot_it(augmented_y, [0, 1])
                    # augmented_y_marks[augmented_y_marks>0] = 1
                    if self.weights_loss == 'categorical_crossentropy':
                        y[j, 0] = augmented_y
                    else:
                        y[j, 0, ..., 0] = augmented_y
                    j += 1
        elif self.module == 'test':
            # print("Creating validation data...")
            for patch in range(self.patch_batch):
                for i, ID in enumerate(list_IDs_temp):
                    ID = ID.split('_')[-1].split('.')[0]
                    aux_x = read_input_videos("{0}/inputs/raw_{1}.tif".format(self.dataset_path, ID),
                                              normalization=self.normalization)

                    aux_y = sitk.ReadImage("{0}/labels/instance_ids_{1}.tif".format(self.dataset_path, ID))
                    aux_y = sitk.GetArrayFromImage(aux_y)
                    aux_y = (aux_y[-1] > 0).astype(np.uint8)

                    if self.dim_input[0] < aux_x.shape[1]:
                        # Place the time dimension at the end (axis=-1).
                        aux_x = np.transpose(aux_x, [1, 2, 0])
                        aux_x, aux_y = random_crop(aux_x, aux_y, (self.dim_input[0], self.dim_input[1]), pdf=self.pdf)
                        # Place the time dimension at the beginning (axis=0).
                        aux_x = np.transpose(aux_x, [2, 0, 1])
                    x[j, ..., 0] = aux_x
                    # Store ground truth
                    aux_y = (aux_y > 0).astype(np.uint8)
                    if self.weights_loss == 'categorical_crossentropy':
                        aux_y = one_hot_it(aux_y, [0, 1])
                        y[j, 0] = aux_y
                    else:
                        y[j, 0, ..., 0] = aux_y
                    del aux_y, aux_x
                    j += 1

        x = x.astype(np.float32)
        y = y.astype(np.uint8)
        return x, y


class DataGeneratorWeights(BaseGenerator):

    def __init__(self, list_IDs, dataset_path, rotation_range,
                 width_shift_range, height_shift_range, shear_range,
                 zoom_range, horizontal_flip, fill_mode, normalization, dim_input=None,
                 pdf=1, module="train", instance_radious=5, patch_batch=1,
                 batch_size=5, shuffle=True, weights_loss="binary_crossentropy",
                 dim_output=None):
        BaseGenerator.__init__(self, list_IDs, dataset_path, rotation_range,
                               width_shift_range, height_shift_range, shear_range,
                               zoom_range, horizontal_flip, fill_mode, dim_input, normalization,
                               pdf=pdf, module=module, patch_batch=patch_batch,
                               batch_size=batch_size, shuffle=shuffle)
        self.instance_radious = instance_radious
        self.weights_loss = weights_loss  # weighted_bce_dice, weighted_bce or a keras loss function("binary_crossentropy").
        self.dim_output = dim_output

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def __data_generation(self, list_IDs_temp):
        x = np.empty((self.batch_size * self.patch_batch, self.dim_input[0], self.dim_input[1], 1))
        if self.weights_loss == 'weighted_bce':
            w = np.empty((self.batch_size * self.patch_batch, self.dim_input[0], self.dim_input[1], 1))
        if self.weights_loss == 'categorical_crossentropy':
            y = np.empty((self.batch_size * self.patch_batch, self.dim_output[0], self.dim_output[1], 2))
        else:
            y = np.empty((self.batch_size * self.patch_batch, self.dim_output[0], self.dim_output[1], 1))
        # Generate data
        j = 0
        if self.module == 'train':
            for patch in range(self.patch_batch):
                for i, ID in enumerate(list_IDs_temp):
                    ID = ID.split('_')[-1].split('.')[0]
                    aux_x = read_input_image("{0}/inputs/raw_{1}.tif".format(self.dataset_path, ID),
                                             normalization=self.normalization)
                    aux_y, aux_y_marks = read_instances("{0}/labels/instance_ids_{1}.tif".format(self.dataset_path, ID),
                                                        radious=self.instance_radious)
                    aux_y_weights = cv2.imread("{0}/weights/instance_ids_{1}_weight.tif".format(self.dataset_path, ID),
                                               cv2.IMREAD_ANYDEPTH)
                    augmented_x, augmented_y, augmented_y_marks, augmented_weights = data_augmentation_weightedmaps(
                        aux_x, aux_y, aux_y_marks, aux_y_weights, self)
                    x[j, :, :, 0] = augmented_x
                    if self.weights_loss == 'weighted_bce':
                        w[j, :, :, 0] = augmented_weights
                    # Store ground truth
                    augmented_y[augmented_y > 0] = 1
                    # Check if one hot representation is needed
                    if self.weights_loss == 'categorical_crossentropy':
                        augmented_y = one_hot_it(augmented_y, [0, 1])
                    # augmented_y_marks[augmented_y_marks>0] = 1
                    if self.weights_loss == 'categorical_crossentropy':
                        y[j] = augmented_y
                    else:
                        y[j, :, :, 0] = augmented_y
                    j += 1

        elif self.module == 'test':
            # print("Creating validation data...")
            for patch in range(self.patch_batch):
                for i, ID in enumerate(list_IDs_temp):
                    ID = ID.split('_')[-1].split('.')[0]
                    aux_x = read_input_image("{0}/inputs/raw_{1}.tif".format(self.dataset_path, ID),
                                             normalization=self.normalization)
                    aux_y, aux_y_marks = read_instances("{0}/labels/instance_ids_{1}.tif".format(self.dataset_path, ID),
                                                        radious=self.instance_radious)
                    # Get random crops
                    input_im, output_labels, output_marks, weights = random_crop_complex(aux_x, aux_y, aux_y_marks,
                                                                                         aux_y, (self.dim_input[0],
                                                                                                 self.dim_input[1]), (
                                                                                             self.dim_output[0],
                                                                                             self.dim_output[1]),
                                                                                         pdf=self.pdf)
                    x[j, :, :, 0] = input_im
                    if self.weights_loss == 'weighted_bce':
                        w[j, :, :, 0] = weights
                    # Store ground truth
                    output_labels[output_labels > 0] = 1
                    if self.weights_loss == 'categorical_crossentropy':
                        output_labels = one_hot_it(output_labels, [0, 1])
                    output_marks[output_marks > 0] = 1
                    if self.weights_loss == 'categorical_crossentropy':
                        y[j] = output_labels
                    else:
                        y[j, :, :, 0] = output_labels
                    j += 1

        x = x.astype(np.float32)
        y = y.astype(np.uint8)
        if self.weights_loss == 'weighted_bce':
            w = w.astype(np.float32)

        if self.weights_loss == 'weighted_bce':
            # Convert it to float to concatenate it with weights
            y = y.astype(np.float32)
            return [x, w, y], y
        elif self.weights_loss == "multiple_output":
            return x, [1 - y, y]
        else:
            return x, y


class DataGeneratorContours(BaseGenerator):

    def __init__(self, list_IDs, dataset_path, rotation_range,
                 width_shift_range, height_shift_range, shear_range,
                 zoom_range, horizontal_flip, fill_mode, normalization, dim_input=None,
                 pdf=1, module="train", patch_batch=1, batch_size=5, shuffle=True):
        BaseGenerator.__init__(self, list_IDs, dataset_path, rotation_range,
                               width_shift_range, height_shift_range, shear_range,
                               zoom_range, horizontal_flip, fill_mode, dim_input, normalization,
                               pdf=pdf, module=module, patch_batch=patch_batch,
                               batch_size=batch_size, shuffle=shuffle)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size * self.patch_batch, self.dim_input[0], self.dim_input[1], 1))
        y = np.empty((self.batch_size * self.patch_batch, self.dim_input[0], self.dim_input[1], 1), dtype=np.uint8)
        # Generate data
        j = 0
        if self.module == 'train':
            for patch in range(self.patch_batch):
                for i, ID in enumerate(list_IDs_temp):
                    ID = ID.split('_')[-1].split('.')[0]
                    aux_x = read_input_image("{0}/inputs/raw_{1}.tif".format(self.dataset_path, ID),
                                             normalization=self.normalization)
                    aux_y = cv2.imread("{0}/labels/instance_ids_{1}.tif".format(self.dataset_path, ID)
                                       , cv2.IMREAD_ANYDEPTH)
                    aux_c = cv2.imread("{0}/contours/instance_ids_{1}.tif".format(self.dataset_path, ID)
                                       , cv2.IMREAD_ANYDEPTH)
                    augmented_x, augmented_y = data_augmentation_contours(aux_x, aux_y, aux_c, self)
                    x[j, :, :, 0] = augmented_x
                    y[j, :, :, 0] = augmented_y
                    j += 1

        elif self.module == 'test':
            # print("Creating validation data...")
            for patch in range(self.patch_batch):
                for i, ID in enumerate(list_IDs_temp):
                    ID = ID.split('_')[-1].split('.')[0]
                    aux_x = read_input_image("{0}/inputs/raw_{1}.tif".format(self.dataset_path, ID),
                                             normalization=self.normalization)
                    aux_y = cv2.imread("{0}/labels/instance_ids_{1}.tif".format(self.dataset_path, ID)
                                       , cv2.IMREAD_ANYDEPTH)
                    aux_c = cv2.imread("{0}/contours/instance_ids_{1}.tif".format(self.dataset_path, ID)
                                       , cv2.IMREAD_ANYDEPTH)
                    aux_y = (aux_y > 0).astype(np.uint8)
                    aux_c = (aux_c > 0)
                    aux_y[aux_c] = 2
                    # Get a random crop without transformation
                    if self.dim_input[0] < aux_x.shape[0]:
                        aux_x, aux_y = random_crop(aux_x, aux_y,
                                                              (self.dim_input[0], self.dim_input[1]),
                                                              pdf=self.pdf)
                    x[j, :, :, 0] = aux_x
                    y[j, :, :, 0] = aux_y
                    j += 1
        x = x.astype(np.float32)
        return x, y
# ------------------------------------------------------------
## Test your data generator with the following code:

# import matplotlib.pyplot as plt
# import glob
# DATASET_PATH = "/content/drive/My Drive/Projectos/3D-PROTUCEL/Code/data/train/stack2im"
# params = {'dataset_path': DATASET_PATH,
#           'rotation_range' : 30,
#           'width_shift_range': 0.2,
#           'height_shift_range': 0.2,
#           'shear_range': 0.2,
#           'zoom_range': 0.2,
#           'horizontal_flip': True,
#           'pdf': 5000,
#           'fill_mode': 'reflect',
#           'patch_batch': 2,
#           'batch_size': 5,
#           'module': 'train'}
# files4training = os.listdir(DATASET_PATH + '/inputs/')
# files4training.sort()
# partition ={'train': files4training}
# self = DataGenerator(partition['train'],**params)
# data = self.__getitem__(1)
# for i in range(data[1].shape[0]):
#   plt.figure(figsize=(10,10))
#   plt.subplot(2,2,1)
#   plt.imshow(data[0][0][i,:,:,0])
#   plt.subplot(2,2,2)
#   plt.imshow(data[0][1][i,:,:,0])
#   plt.subplot(2,2,3)
#   plt.imshow(data[1][i,:,:,0])
#   plt.subplot(2,2,4)
#   plt.imshow(data[1][i,:,:,0])
#   plt.show()
