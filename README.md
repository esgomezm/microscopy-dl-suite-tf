[![minimal Python version](https://img.shields.io/badge/Python-3.6-6666ff)](https://www.anaconda.com/distribution/)
![PyPI](https://img.shields.io/pypi/v/TensorFlow?color=orange&label=TensorFlow&logo=TensorFlow)
[![License](https://img.shields.io/badge/License-BSD%203--Clause--Clear-yellowlima.svg)](https://spdx.org/licenses/BSD-3-Clause-Clear.html)

# Deep Learning segmentation suite dessigned for 2D microscopy image segmentation
This repository provides researchers with a code to try different encoder-decoder configurations for the binary segmentation of 2D images in a video. It offers regular 2D U-Net variants and recursive approaches by combining ConvLSTM on top of the encoder-decoder.

## Programmed loss-functions
### When the output of the network has just one channel: foreground prediction
- Weighted binary cross-entropy: define a map of weights for the segmentation and feed the network with three images: input, weights and output.
  - Inspired by: 
    - https://stackoverflow.com/questions/48555820/keras-binary-segmentation-add-weight-to-loss-function/48577360#48577360
    - https://stackoverflow.com/questions/55213599/u-net-with-pixel-wise-weighted-cross-entropy-input-dimension-errors
  - Open post: https://stackoverflow.com/questions/61225857/pixel-wise-weighted-loss-function-in-keras-tensorflow-2-0

- Weighted binary cross-entropy + DICE coefficient: Fix a weight value for foreground and combine it with DICE coefficient to create a loss function focused on false negatives (i.e. the tendency is to avoid them). 
  - Taken from https://www.kaggle.com/lyakaap/weighing-boundary-pixels-loss-script-by-keras2
  - Check description here: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
  - Alternative loss functions: https://github.com/kohrah/DSBowl2018/blob/master/src/zoo_losses_K.py
- Binary cross-entropy: keras classical binary cross-entropy

### When the output of the network has two channels: background and foreground prediction

- (Weighted) categorical cross-entropy: keras classical categorical cross-entropy
- Sparse categorical cross-entropy: same as the categorical cross-entropy but it allows the user to enter labelled grounf truth with a single channel and as many labels as classes, rather tha in a one-hote encoding fashion.
  
# Parameter configuration in the configuration.json
| argument                  | description                                                                   | default value |
| ------------------------- | ----------------------------------------------------------------------------- | ------------- |
| **model parameters**           
| cnn_name                  | Name used to store the model. It determines the type of network that will be trained. If it contains 'categorical_unet' : different labels ({0}, {1}, ...). Other chances: 'categorical_unet_transpose', 'categorical_unet_fc_dil', 'categorical_unet_fc'. If it contains 'lstm': trains a 2D U-Net with LSTM units in the contracting path. | Normal U-Net with signle channel output and binary cross-entropy as loss function |
| OUTPUTPATH                | Directory where the trained model, logs and results are stored                | "externaldata_cce_weighted_001" |
| TRAINPATH                 | in pixels, mandatory if 'full' MARKER_ANNOTATIONS                             | "/data/train/stack2im" |
| TESTPATH                  | parameter **c**, mandatory if 'weak' MARKER_ANNOTATIONS                       | "/data/test/stack2im" |
| model_n_filters           | source of reference annotations, in CTC corresponds to gold and silver truth  | 32  |
| model_pools               | depth of the U-Net                                                            | 3  |
| model_kernel_size         | size of the kernel for the convolutions inside the U-Net. It's 2D as the network is thought to be for 2D data segmentation.                    | [3, 3] |
| model_lr                  | directory with full annotations  (full annotations)                           | 0.01 |
| model_dropout             | directory with reference markers (weak annotations)                           | 0.2 |
| model_activation          | number of digits to indexing images                                           | "relu"|
| model_last_activation     |                                                                               |"sigmoid"|
| model_padding             |                                                                               | "same"|
| model_kernel_initializer  |                                                                               |            |
| model_lossfunction        | Categorical-unets: "sparse_cce", "multiple_output", "categorical_cce", "weighted_bce_dice". Binary-unets: "binary_cce", "weighted_bce_dice" or "weighted_bce"                     | "sparse_cce" |
| model_metrics             |                                                                               |            |
| model_category_weights    | Weights for multioutput networks (tips prediction)                                          |            |
| **training**  
| train_max_epochs          |                                                                             |            |
| train_pretrained_weights  | The pretrained weights are thought to be inside the `checkpoints` folder that is created in `OUTPUTPATH`. In case you want to make a new experiment, we suggest changing the name of the network `cnn_name`. This is thought to keep track of the weights used for the pretraining.                                          |  None or 'lstm_unet00004.hdf5'  |
| callbacks_save_freq       | Use a quite large saving frequency to store networks every 100 epochs for example. If the frequency is smaller than the number of inputs processed on each epoch, a set of trained weights will be stored in each epoch. Note that this increases significantly the size of the checkpoints folder.                                  |    50, 2000, ...   |
| callbacks_patience        | Number of analyzed inputs for which the improvement is analyzed before reducing the learning rate.                         |    100  |
| callbacks_tb_update_freq  | Tensorboard updating frequency                                              | 10 |
| datagen_patch_batch       | Number of patches to crop from each image entering the generator            |   1   |
| datagen_batch_size        | Number of images that will compose the batch on each iteration of the training. `Final batch size = datagen_sigma * datagen_patch_batch`. Total number of `iterations = np.floor((total images)/datagen_sigma * datagen_patch_batch)`         |     5   |
| datagen_dim_size          | 2D size of the data patches that will enter the network.                    |  [512, 512] |
| datagen_sampling_pdf      | Sampling probability distribution function to deal with data unbalance (few objects in the image).   |  500000 |
| datagen_type              | If it contains 'contours', the data generator will create a ground truth with the segmentation and the contours of those segmentations. If it contains 'tips', it will generate outputs of type list for multioutput models: the segmentation ground truth and the keypoints of the tips. | By default it will generate a ground truth with two channels (background and foreground) |
| datagen_tips_factor       | factor by which the tips annotations are multiplied. We recommend to use 1 to keep a better control of the loss function and weighted losses. | 1 |
| datagen_sigma             | sigma value used to smooth the tips annotations. This sigma is different from the one used in the L1L2 loss function.  | 2 |
| **inference**
| newinfer_normalization        |     "MEAN", "MEDIAN"                            |            |
| newinfer_uneven_illumination  |      "False"                                                                     |            |
| newinfer_epoch_process_test   |    File with the trained network at the specified epoch, with the name specified in `cnn_name` and stored at `OUTPUTPATH/checkpoints`.    |  20 |
| newinfer_padding              |  Halo, padding, half receptive field of a pixel.              |  [95, 95]  |
| newinfer_data              |    Data to process.                     | "data/test/" |
| newinfer_output_folder_name |  Name of the forlder in which all the processed images will be stored. | "test_output" |
| PATH2VIDEOS              |    csv file with the relation between the single 2D frames and the videos from where they come.                       | "data/test/stack2im/videos2im_relation.csv"     |



# Notes about the code reused from different sources or the CNN architecture definitions
## U-Net for binary segmentation
U-Net architecture for TensorFlow 2 based on the example given in https://www.kaggle.com/advaitsave/tensorflow-2-nuclei-segmentation-unet
