"""
Created on Tue Apr 14 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
from data_generators.generator import DataGenerator, DataGeneratorLSTM, DataGeneratorWeights, DataGeneratorContours
import os

def training(config):
    mi = config.datagen_dim_size[0]
    ni = config.datagen_dim_size[1]
    files4training = os.listdir(config.TRAINPATH + '/inputs/')
    files4training.sort()
    partition = {'train': files4training}
    params = {'dataset_path': config.TRAINPATH,
              'rotation_range': 30,
              'width_shift_range': 0.2,
              'height_shift_range': 0.2,
              'shear_range': 0.2,
              'zoom_range': 0.2,
              'horizontal_flip': True,
              'pdf': config.datagen_sampling_pdf,
              'fill_mode': 'reflect',
              'dim_input': (mi, ni),
              'patch_batch': config.datagen_patch_batch,
              'batch_size': config.datagen_batch_size,
              'module': 'train',
              'normalization': config.normalization
              }
    # Generator
    training_generator = DataGenerator(partition['train'], **params)
    return training_generator

def test_shots(config):
    # import SimpleITK as sitk
    mi = config.datagen_dim_size[0]
    ni = config.datagen_dim_size[1]
    files4test = os.listdir(config.VALPATH + '/inputs/')
    files4test.sort()
    partition = {'test': files4test}
    params_test = {'dataset_path': config.VALPATH,
                   'rotation_range': 0,
                   'width_shift_range': 0,
                   'height_shift_range': 0,
                   'shear_range': 0,
                   'zoom_range': 0,
                   'horizontal_flip': False,
                   'pdf': 5000,
                   'fill_mode': 'reflect',
                   'dim_input': (mi, ni),
                   'patch_batch': 1,
                   'batch_size': config.datagen_batch_size, #50, #len(files4test),
                   'module': 'test',
                   'normalization': config.normalization
                   }
    # Generators
    test_generator = DataGenerator(partition['test'], **params_test)
    # test_input, test_output = test_generator.__getitem__(0)
    return test_generator

def training_contours(config):
    mi = config.datagen_dim_size[0]
    ni = config.datagen_dim_size[1]
    files4training = os.listdir(config.TRAINPATH + '/inputs/')
    files4training.sort()
    partition = {'train': files4training}
    params = {'dataset_path': config.TRAINPATH,
              'rotation_range': 30,
              'width_shift_range': 0.2,
              'height_shift_range': 0.2,
              'shear_range': 0.2,
              'zoom_range': 0.2,
              'horizontal_flip': True,
              'pdf': config.datagen_sampling_pdf,
              'fill_mode': 'reflect',
              'dim_input': (mi, ni),
              'patch_batch': config.datagen_patch_batch,
              'batch_size': config.datagen_batch_size,
              'module': 'train',
              'normalization': config.normalization
              }
    # Generator
    training_generator = DataGeneratorContours(partition['train'], **params)
    return training_generator

def test_shots_contours(config):
    # import SimpleITK as sitk
    mi = config.datagen_dim_size[0]
    ni = config.datagen_dim_size[1]
    files4test = os.listdir(config.VALPATH + '/inputs/')
    files4test.sort()
    partition = {'test': files4test}
    params_test = {'dataset_path': config.VALPATH,
                   'rotation_range': 0,
                   'width_shift_range': 0,
                   'height_shift_range': 0,
                   'shear_range': 0,
                   'zoom_range': 0,
                   'horizontal_flip': False,
                   'pdf': 5000,
                   'fill_mode': 'reflect',
                   'dim_input': (mi, ni),
                   'patch_batch': 1,
                   'batch_size': config.datagen_batch_size, #50,# len(files4test),
                   'module': 'test',
                   'normalization': config.normalization
                   }
    # Generators
    test_generator = DataGeneratorContours(partition['test'], **params_test)
    # test_input, test_output = test_generator.__getitem__(0)
    return test_generator

def training_weightedmap(config):
    mi = config.datagen_dim_size[0]
    ni = config.datagen_dim_size[1]
    params = {'dataset_path': config.TRAINPATH,
              'rotation_range': 30,
              'width_shift_range': 0.2,
              'height_shift_range': 0.2,
              'shear_range': 0.2,
              'zoom_range': 0.2,
              'horizontal_flip': True,
              'pdf': config.datagen_sampling_pdf,
              'fill_mode': 'reflect',
              'dim_input': (mi, ni),
              'dim_output': (mi, ni),
              'patch_batch': config.datagen_patch_batch,
              'batch_size': config.datagen_batch_size,
              'module': 'train',
              'weights_loss': config.model_lossfunction,
              'normalization': config.normalization
              }
    files4training = os.listdir(config.TRAINPATH + '/inputs/')
    files4training.sort()
    partition = {'train': files4training}
    # Generator
    training_generator = DataGeneratorWeights(partition['train'], **params)
    return training_generator

def test_shots_weightedmap(config):
    # import SimpleITK as sitk
    mi = config.datagen_dim_size[0]
    ni = config.datagen_dim_size[1]
    files4test = os.listdir(config.VALPATH + '/inputs/')
    files4test.sort()
    partition = {'test': files4test}
    params_test = {'dataset_path': config.VALPATH,
                   'rotation_range': 0,
                   'width_shift_range': 0,
                   'height_shift_range': 0,
                   'shear_range': 0,
                   'zoom_range': 0,
                   'horizontal_flip': False,
                   'pdf': 5000,
                   'fill_mode': 'reflect',
                   'dim_input': (mi, ni),
                   'dim_output': (mi, ni),
                   'patch_batch': 1,
                   'batch_size': config.datagen_batch_size, # 50, #len(files4test),
                   'module': 'test',
                   'weights_loss': config.model_lossfunction,
                   'normalization': config.normalization
                   }
    # Generators
    test_generator = DataGeneratorWeights(partition['test'], **params_test)
    # test_input, test_output = test_generator.__getitem__(0)
    return test_generator

def training_videos(config):
    mi = config.datagen_dim_size[0]
    ni = config.datagen_dim_size[1]
    params = {'dataset_path': config.TRAINPATH,
              'rotation_range': 30,
              'width_shift_range': 0.2,
              'height_shift_range': 0.2,
              'shear_range': 0.2,
              'zoom_range': 0.2,
              'horizontal_flip': True,
              'pdf': config.datagen_sampling_pdf,
              'fill_mode': 'reflect',
              'dim_input': (mi, ni),
              'video_length': config.model_time_windows,
              'patch_batch': config.datagen_patch_batch,
              'batch_size': config.datagen_batch_size,
              'module': 'train',
              'weights_loss': config.model_lossfunction,
              'normalization': config.normalization
              }
    files4training = os.listdir(config.TRAINPATH + '/inputs/')
    files4training.sort()
    partition = {'train': files4training}
    # Generator
    training_generator = DataGeneratorLSTM(partition['train'], **params)
    return training_generator

def test_videos(config):
    mi = config.datagen_dim_size[0]
    ni = config.datagen_dim_size[1]
    files4test = os.listdir(config.VALPATH + '/inputs/')
    files4test.sort()
    partition = {'test': files4test}
    params_test = {'dataset_path': config.VALPATH,
                   'rotation_range': 0,
                   'width_shift_range': 0,
                   'height_shift_range': 0,
                   'shear_range': 0,
                   'zoom_range': 0,
                   'horizontal_flip': False,
                   'pdf': 5000,
                   'fill_mode': 'reflect',
                   'dim_input': (mi, ni),
                   'video_length': config.model_time_windows,
                   'patch_batch': 1,
                   'batch_size': config.datagen_batch_size, #len(files4test), #16,
                   'module': 'test',
                   'weights_loss': config.model_lossfunction,
                   'normalization': config.normalization
                   }
    # Generators
    test_generator = DataGeneratorLSTM(partition['test'], **params_test)
    # test_input, test_output = test_generator.__getitem__(0)
    return test_generator

# # Store a couple of snapshots to see the progress of the network.
# if not os.path.exists(config.OUTPUTPATH + '/test_shots/'):
#     os.makedirs(config.OUTPUTPATH + '/test_shots/')

# for j in range(len(test_data[1])):
#     sitk.WriteImage(sitk.GetImageFromArray(test_data[1][j, :, :, 0].astype(np.uint8)),
#                     config.OUTPUTPATH + "/test_shots/gt_{0}.tif".format(np.str(j)))
#     if config.model_loss_function == "weighted_bce":
#         sitk.WriteImage(sitk.GetImageFromArray(test_data[0][0][j, :, :, 0]),
#                         config.OUTPUTPATH + "/test_shots/input_{0}.tif".format(np.str(j)))
#     else:
#         sitk.WriteImage(sitk.GetImageFromArray(test_data[0][j, :, :, 0]),
#                         config.OUTPUTPATH + "/test_shots/input_{0}.tif".format(np.str(j)))
