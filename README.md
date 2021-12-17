[![minimal Python version](https://img.shields.io/badge/Python-3.6-6666ff)](https://www.anaconda.com/distribution/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellowlima.svg)](https://opensource.org/licenses/MIT)


# Deep Learning segmentation suite dessigned for 2D microscopy image segmentation
This repository provides researchers with a code to try different encoder-decoder configurations for the binary segmentation of 2D images in a video. It offers regular 2D U-Net variants and recursive approaches by combining ConvLSTM on top of the encoder-decoder.
# Citation
If you found this code useful for your research, please, cite the corresponding preprint:

[Estibaliz Gómez-de-Mariscal, Hasini Jayatilaka, Özgün Çiçek, Thomas Brox, Denis Wirtz, Arrate Muñoz-Barrutia, *Search for temporal cell segmentation robustness in phase-contrast microscopy videos*, arXiv 2021 (arXiv:2112.08817).](https://arxiv.org/abs/2112.08817)

```
@misc{gómezdemariscal2021search,
      title={Search for temporal cell segmentation robustness in phase-contrast microscopy videos}, 
      author={Estibaliz Gómez-de-Mariscal and Hasini Jayatilaka and Özgün Çiçek and Thomas Brox and Denis Wirtz and Arrate Muñoz-Barrutia},
      year={2021},
      eprint={2112.08817},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
# Quick guide
## Installation
Clone this repository and create all the required libraries
```
git clone https://github.com/esgomezm/microscopy-dl-suite-tf
pip3 install -r microscopy-dl-suite-tf/dl-suite/requirements.txt
```

## Download or place your data in an accessible directory
Download the [example data from Zenodo](https://zenodo.org/record/5777994). Place the training, validation and test data in three independent folders. Each of them should contain an `input`and `labels` folder. **For 2D images**, the name of the images should be `raw_000.tif` and `instance_ids_000.tif` for the input and ground truth images respectively. If **the ground truth is given as videos**, then the inputs and labels should have the same name.

## Create a configuration .json file with all the information for the model architecture and training. 

Check out some [examples](https://github.com/esgomezm/microscopy-dl-suite-tf/tree/main/examples/config) of configuration files. You will need to update the paths to the training, validation and test datasets. All the details for this file is given [here](https://github.com/esgomezm/microscopy-dl-suite-tf#parameter-configuration-in-the-configurationjson).

## Run model training
Run the file `train.py` indicating the path to the configuration `JSON` that contains all the information. This script will also test the model with the images provided in the `"TESPATH"` field of the configuration file.
```
python microscopy-dl-suite-tf/dl-suite/train.py 'microscopy-dl-suite-tf/examples/config/config_mobilenet_lstm_5.json' 
```

## Run model testing
If you only want to run the test step, it is also possible with the `test.py`:
```
python microscopy-dl-suite-tf/dl-suite/test.py 'microscopy-dl-suite-tf/examples/config/config_mobilenet_lstm_5.json' 
```
## Cell tracking from the instance segmentations
Videos with instance segmentations can be easily tracked with [TrackMate](https://imagej.net/plugins/trackmate/). TrackMate is compatible with cell splitting, merging, and gap filling, making it suitable for the task.

The cells in our 2D videos exit and enter the focus plane, so we fill part of the gaps caused by these irregularities. We apply a Gaussian filter along the time axis on the segmented output images. The filtered result is merged with the output masks of the model as follows: all binary masks are preserved, and the positive values of the filtered image are included as additional masks. Those objects smaller than 100 pixels are discarded. 

This processing is contained in the file `tracking.py`, in the section called `Process the binary images and create instance segmentations`. TrackMate outputs a new video with the information of the tracks given as uniquely labelled cells. Then, such information can be merged witht he original segmentation (without the temporal interpolation), using the code section `Merge tracks and segmentations`.

# Technicalities
## Available model architectures

- `'mobilenet_mobileunet_lstm'`: A pretrained mobilenet in the encoder with skip connections to the decoder of a mobileunet and a ConvLSTM layer at the end that will make the entire architecture recursive.
- `'mobilenet_mobileunet'`: A pretrained mobilenet in the encoder with skip connections to the decoder of a mobileunet (2D).
- `'unet_lstm'`: 2D U-Net with ConvLSTM units in the contracting path.
- `'categorical_unet_transpose'`: 2D U-Net for different labels ({0}, {1}, ...) with transpose convolutions instead of upsampling.
- `'categorical_unet_fc_dil'`: 2D U-Net for different labels ({0}, {1}, ...) with fully connected dilated convolutions.
- `'categorical_unet_fc'`: 2D U-Net for different labels ({0}, {1}, ...) with fully connected convolutions.
- `'categorical_unet'`: 2D U-Net for different labels ({0}, {1}, ...).
- `'unet'` or `"None"`: 2D U-Net with a single output.

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

## Prepare the data

- If you want to create a set of ground truth data with the format specified in the Cell Tracking Challenge, you can use the script [`prepare_videos_ctc.py`](https://github.com/esgomezm/microscopy-dl-suite-tf/blob/main/dl-suite/additional_scripts/prepare_videos_ctc.py).
- If you want to create 2D images from the videos, you can use the script [`prepare_data.py`](https://github.com/esgomezm/microscopy-dl-suite-tf/blob/main/dl-suite/additional_scripts/prepare_data.py).
- In the folder [`additional_scripts`](https://github.com/esgomezm/microscopy-dl-suite-tf/tree/main/dl-suite/additional_scripts) you will find ImageJ macros or python code to keep processing the data to generate borders around the segmented cells for example.

## Parameter configuration in the configuration.json
| argument                  | description                                                                   | example value |
| ------------------------- | ----------------------------------------------------------------------------- | ------------- |
| **model parameters**           
| cnn_name                  | Model architecture. Options available [here](https://github.com/esgomezm/microscopy-dl-suite-tf#available-model-architectures)    | "mobilenet_mobileunet_lstm_tips" |
| OUTPUTPATH                | Directory where the trained model, logs and results are stored                | "externaldata_cce_weighted_001" |
| TRAINPATH                 | Directory with the source of reference annotations that will be used for training the network. It should contain two folders (`inputs` and `labels`). The name of the images should be `raw_000.tif` and `instance_ids_000.tif` for the input and ground truth images respectively.| "/data/train/stack2im" |
| VALPATH                  | Directory with the source of reference annotations that will be used for validation of the network. It should contain two folders (`inputs` and `labels`). The name of the images should be `raw_000.tif` and `instance_ids_000.tif` for the input and ground truth images respectively. If you are running different configurations of a network or different instances, it might be recommended to keep always the same forlder for this.| "/data/val/stack2im" |
| TESTPATH                  | Directory with the source of reference annotations that will be used to test the network. It should contain two folders (`inputs` and `labels`). The name of the images should be `raw_000.tif` and `instance_ids_000.tif` for the input and ground truth images respectively. | "/data/test/stack2im" |
| model_n_filters           | source of reference annotations, in CTC corresponds to gold and silver truth  | 32  |
| model_pools               | Depth of the U-Net                                                            | 3  |
| model_kernel_size         | size of the kernel for the convolutions inside the U-Net. It's 2D as the network is thought to be for 2D data segmentation.| [3, 3] |
| model_lr                  | Model learning rate                                                     | 0.01 |
| model_mobile_alpha        | Width percentage of the MobileNetV2 used as a pretrained encoder. The values are limited by the TensorFlow model zoo to 0.35, 0.5, 1| 0.35 |
| model_time_windows        | Length in frames of the input video when training recurrent networks (ConvLSTM layers) | 5 |
| model_dilation_rate       | Dilation rate for dilated convolutions. If set to 1, it will be like a normal convolution. | 1 |
| model_dropout             | Dropout ration. It will increase with the depth of the encoder decoder. | 0.2 |
| model_activation          | Same as in Keras & TensorFlow libraries. "relu", "elu" are the most common ones.| "relu"|
| model_last_activation     | Same as in Keras & TensorFlow libraries. "sigmoid", "tanh" are the most common ones. |"sigmoid"|
| model_padding             | Same as in Keras & TensorFlow libraries. "same" is strongly recommended. | "same"|
| model_kernel_initializer  | Model weights initializer method. Same name as the one in Keras & TensorFlow library. | "glorot_uniform" |
| model_lossfunction        | Categorical-unets: "sparse_cce", "multiple_output", "categorical_cce", "weighted_bce_dice". Binary-unets: "binary_cce", "weighted_bce_dice" or "weighted_bce"                     | "sparse_cce" |
| model_metrics             |  Accuracy metric to compute during model training.  | "accuracy" |
| model_category_weights    | Weights for multioutput networks (tips prediction)                                          |            |
| **training**  
| train_max_epochs          | Number of training epochs                            |  1000 |
| train_pretrained_weights  | The pretrained weights are thought to be inside the `checkpoints` folder that is created in `OUTPUTPATH`. In case you want to make a new experiment, we suggest changing the name of the network `cnn_name`. This is thought to keep track of the weights used for the pretraining.                                          |  None or `lstm_unet00004.hdf5`  |
| callbacks_save_freq       | Use a quite large saving frequency to store networks every 100 epochs for example. If the frequency is smaller than the number of inputs processed on each epoch, a set of trained weights will be stored in each epoch. Note that this increases significantly the size of the checkpoints folder.                                  |    50, 2000, ...   |
| callbacks_patience        | Number of analyzed inputs for which the improvement is analyzed before reducing the learning rate.                         |    100  |
| callbacks_tb_update_freq  | Tensorboard updating frequency                                              | 10 |
| datagen_patch_batch       | Number of patches to crop from each image entering the generator            |   1   |
| datagen_batch_size        | Number of images that will compose the batch on each iteration of the training. `Final batch size = datagen_sigma * datagen_patch_batch`. Total number of `iterations = np.floor((total images)/datagen_sigma * datagen_patch_batch)`         |     5   |
| datagen_dim_size          | 2D size of the data patches that will enter the network.                    |  [512, 512] |
| datagen_sampling_pdf      | Sampling probability distribution function to deal with data unbalance (few objects in the image).   |  500000 |
| datagen_type              | If it contains `contours`, the data generator will create a ground truth with the segmentation and the contours of those segmentations. | By default it will generate a ground truth with two channels (background and foreground) |
| **inference**
| newinfer_normalization        | Intensity normalization procedure. It will calculate the mean, median or percentile of each input image before augmentation and cropping. | "MEAN", "MEDIAN", "PERCENTILE" |
| newinfer_uneven_illumination  |  To correct or not for uneven illumination the input images (before tiling) |   "False"    |
| newinfer_epoch_process_test   |    File with the trained network at the specified epoch, with the name specified in `cnn_name` and stored at `OUTPUTPATH/checkpoints`.       |  20 |
| newinfer_padding              |  Halo, padding, half receptive field of a pixel.              |  [95, 95]  |
| newinfer_data              |    Data to process.                     | "data/test/" |
| newinfer_output_folder_name |  Name of the forlder in which all the processed images will be stored. | "test_output" |
| PATH2VIDEOS              |    csv file with the relation between the single 2D frames and the videos from where they come.                       | "data/test/stack2im/videos2im_relation.csv"     |



# Notes about the code reused from different sources or the CNN architecture definitions
**U-Net for binary segmentation**
U-Net architecture for TensorFlow 2 based on the example given in https://www.kaggle.com/advaitsave/tensorflow-2-nuclei-segmentation-unet
