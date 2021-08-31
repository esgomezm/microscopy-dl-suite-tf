from data_generators.generator import DataGenerator, DataGeneratorLSTM, DataGeneratorWeights, DataGeneratorContours
import matplotlib.pyplot as plt
import os

mi = 512
ni = 512
TRAINPATH = '/home/esgomezm/Documents/3D-PROTUCEL/data/test/stack2im/'
files4training = os.listdir(os.path.join(TRAINPATH + 'inputs'))
files4training.sort()
partition = {'train': files4training}
params = {'dataset_path': TRAINPATH,
          'rotation_range': 30,
          'width_shift_range': 0.2,
          'height_shift_range': 0.2,
          'shear_range': 0.2,
          'zoom_range': 0.2,
          'horizontal_flip': True,
          'pdf': 5000,
          'fill_mode': 'reflect',
          'dim_input': (mi, ni),
          'patch_batch': 1,
          'batch_size': 10,
          'module': 'train'
          }
# Generator
training_generator = DataGeneratorContours(partition['train'], **params)
x, y = training_generator.__getitem__(0)

for i in range(x.shape[0]):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x[i,...,0])
    plt.subplot(1, 2, 2)
    plt.imshow(y[i, ..., 0], vmin=0, vmax=2)
    plt.show()
#
# ID='032'
#
# aux_x = read_input_image("{0}/inputs/raw_{1}.tif".format(TRAINPATH, ID))
# aux_y = cv2.imread("{0}/labels/instance_ids_{1}.tif".format(TRAINPATH, ID)
#                    , cv2.IMREAD_ANYDEPTH)
# aux_c = cv2.imread("{0}/contours/instance_ids_{1}.tif".format(TRAINPATH, ID)
#                    , cv2.IMREAD_ANYDEPTH)
#
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(aux_y)
# plt.subplot(1, 2, 2)
# plt.imshow(aux_c)
# plt.show()
#
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
#
# data_gen_args = dict(rotation_range=params['rotation_range'],
#                     width_shift_range=params['width_shift_range'],
#                     height_shift_range=params['height_shift_range'],
#                     shear_range=params['shear_range'],
#                     zoom_range=params['zoom_range'],
#                     horizontal_flip=params['horizontal_flip'],
#                     fill_mode=params['fill_mode'])
# # Initialize Keras data augmentaiton
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(dtype='uint8', **data_gen_args)
# contour_datagen = ImageDataGenerator(dtype='uint8', **data_gen_args)
#
# # Generate the same seed:
# seed = np.random.randint(10000000)
#
# x = aux_x.reshape((1, aux_x.shape[0], aux_x.shape[1], 1))
# for batch in image_datagen.flow(x, batch_size=1, seed=seed):
#     augmented_x = batch[0, :, :, 0]
#     break
#
# # Apply the same data augmentation to the masks and contours
# y = aux_y.reshape((1, aux_y.shape[0], aux_y.shape[1], 1))
# for batch in mask_datagen.flow(y, batch_size=1, seed=seed):
#     augmented_y = batch[0, :, :, 0]
#     break
#
# c = aux_c.reshape((1, aux_c.shape[0], aux_c.shape[1], 1))
# for batch in contour_datagen.flow(c, batch_size=1, seed=seed):
#     augmented_c = batch[0, :, :, 0]
#     break
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(augmented_y)
# plt.subplot(1, 2, 2)
# plt.imshow(augmented_c)
# plt.show()
#
# # Create a single mask with the contours and segmentations
# augmented_c = (augmented_c > 0).astype(np.uint8)
# augmented_y = (augmented_y > 0).astype(np.uint8)
# augmented_y[augmented_c] = 2
#
# input_im, output_labels = random_crop(augmented_x, augmented_y,
#                                           (512,512),
#                                           pdf=50000)
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(input_im)
# plt.subplot(1, 2, 2)
# plt.imshow(output_labels)
# plt.show()
