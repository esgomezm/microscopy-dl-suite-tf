"""
Created on Thu Sept 24 2020-

@author: E. Gomez de Mariscal
GitHub username: esgomezm
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
from tensorflow.keras import losses

# --------------------------------
# ## Unet with tf 2.0.0
# https://www.kaggle.com/advaitsave/tensorflow-2-nuclei-segmentation-unet

# ## binary weighted loss example

# https://www.kaggle.com/lyakaap/weighing-boundary-pixels-loss-script-by-keras2

# https://stackoverflow.com/questions/48555820/keras-binary-segmentation-add-weight-to-loss-function/48577360

# https://stackoverflow.com/questions/55213599/u-net-with-pixel-wise-weighted-cross-entropy-input-dimension-errors

# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

# https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred
# --------------------------------

# weight: weighted tensor(same s☺hape as mask image)
def weighted_bce(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
           (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
        y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce(y_true, y_pred, weight) + weighted_dice(y_true, y_pred, weight)
    return loss


def bce_loss(X):
    # y_true, y_pred, weight = X
    y_true, y_pred = X
    loss = binary_crossentropy(y_true, y_pred)
    loss = tf.expand_dims(loss, 3)
    # loss = multiply([loss, weight])
    return loss


def identity_loss(y_true, y_pred):
    # return K.mean(y_pred, axis=-1)
    return y_pred

def jaccard_multiple_output(y_true, y_pred, from_logits = True):
    """Define Jaccard index for multiple labels.
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
       Return:
            jac (tensor): Jaccard index value
    """
    
    if from_logits:
        # run activation to evaluate the jaccard index
        y_pred_ = tf.sigmoid(y_pred)
    y_pred_ = y_pred_ > 0.5
    y_pred_ = tf.cast(y_pred_, dtype=tf.int8)
    y_true_ = tf.cast(y_true, dtype=tf.int8)

    TP = tf.math.count_nonzero(y_pred_ * y_true_)
    FP = tf.math.count_nonzero(y_pred_ * (1 - y_true_))
    FN = tf.math.count_nonzero((1 - y_pred_) * y_true_)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                  lambda: tf.cast(0.000, dtype='float64'))

    return jac

def jaccard_sparse(y_true, y_pred, skip_background=True):
    """Define Jaccard index (multi-class).
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
            skip_background (bool, optional): skip background label.
       Return:
            jac (tensor): Jaccard index value
    """
    # number of classes (last dimension of predictions)
    num_classes = tf.shape(y_pred)[-1]
    
    # one_hot representation of predicted segmentation
    y_pred_ = tf.cast(y_pred, dtype=tf.int32)
    y_pred_ = tf.one_hot(tf.math.argmax(y_pred_, axis=-1), num_classes, axis=-1)         

    # one_hot representation of ground truth segmentation
    y_true_ = tf.cast(y_true[...,0], dtype=tf.int32)
    y_true_ = tf.one_hot(y_true_, num_classes, axis=-1) 
    if skip_background:
        y_pred_ = y_pred_[...,1:]
        y_true_ = y_true_[...,1:]
    
    TP = tf.math.count_nonzero(y_pred_ * y_true_)
    FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
    FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                  lambda: tf.cast(0.000, dtype='float64'))

    return jac


def jaccard_cce(y_true, y_pred, skip_background=True):
    """Define Jaccard index for multiple labels.
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
            skip_background (bool, optional): skip 0-label from calculation
       Return:
            jac (tensor): Jaccard index value
    """
    # We read the number of classes from the last dimension of the true labels
    num_classes = tf.shape(y_true)[-1]
    # one_hot representation of predicted segmentation after argmax
    y_pred_ = tf.cast(y_pred, dtype=tf.float32)
    y_pred_ = tf.one_hot(tf.math.argmax(y_pred_, axis=-1), num_classes, axis=-1)
    
    # y_true is already one-hot encoded
    y_true_ = tf.cast(y_true, dtype=tf.float32)
    # skip background pixels from the Jaccard index calculation
    if skip_background:
      y_true_ = y_true_[...,1:]
      y_pred_ = y_pred_[...,1:]

    TP = tf.math.count_nonzero(y_pred_ * y_true_)
    FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
    FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                  lambda: tf.cast(0.000, dtype='float64'))

    return jac

## Code taken from DeepSTORM at ZeroCostDL4Mic. Please cite when using it
#  Define a matlab like gaussian 2D filter
def matlab_style_gauss2D(shape=(7,7),sigma=1):
    """
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h.astype(dtype=K.floatx())
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h*2.0
    h = h.astype('float32')
    return h


# Expand the filter dimensions
## We changed the kernel size from 7 to 10.
# psf_heatmap = matlab_style_gauss2D(shape=(14, 14), sigma=2)
# gfilter = tf.reshape(psf_heatmap, [14, 14, 1, 1])

# Combined MSE + L1 loss
def L1L2loss(input_shape, gfilter, strides=(1, 1)):
    """
    Args:
        input_shape: (512,512,1)

    Returns:
    """
    def bump_mse(heatmap_true, spikes_pred):

        # generate the heatmap corresponding to the predicted spikes
        if len(strides) == 2:
            heatmap_pred = K.conv2d(spikes_pred, gfilter, strides=strides, padding='same')

        elif len(strides) == 3:
            heatmap_pred = K.conv3d(spikes_pred, gfilter, strides=strides, padding='same')

        # heatmaps MSE
        loss_heatmaps = losses.mean_squared_error(heatmap_true,heatmap_pred)

        # l1 on the predicted spikes
        loss_spikes = losses.mean_absolute_error(spikes_pred,tf.zeros(input_shape))
        return loss_heatmaps + loss_spikes
    return bump_mse

