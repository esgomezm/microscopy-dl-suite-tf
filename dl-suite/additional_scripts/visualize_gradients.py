# from data.data_handling import read_input_image, read_instances
# from models.unet import unet, identity_loss
import tensorflow as tf
import numpy as np
import cv2
# from utils.gradients import obtain_gradients
# import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import concatenate, multiply
import matplotlib.pyplot as plt
def identity_loss(y_true, y_pred):
    return y_pred

## Load models
# model_name = '/media/esgomezm/bcss/egomez/3dprotucell_data/externaldata_bce_001/checkpoints/unet_weights00200.hdf5'
# model_name = '/media/esgomezm/bcss/egomez/3dprotucell_data/externaldata_bce_weighted_001/checkpoints/unet_weights00200.hdf5'
model = unet(n_filters=32, activation='relu',lr=0.0001,
                           padding='same',
                           dropout=0.0,
                           last_activation='sigmoid', # None
                           kernel_init = "he_uniform", # "he_normal""glorot_uniform"
                           metrics="accuracy",
                           loss_function="weighted_bce")
#
# model = unet(n_filters=32, activation='relu',lr=0.0001,
#                            padding='same',
#                            dropout=0.0,
#                            last_activation='sigmoid', # None
#                            kernel_init="glorot_uniform", # "he_normal"
#                            metrics="accuracy",
#                            loss_function="binary_crossentropy")
# model.load_weights(model_name)

## Read images
# input_im = read_input_image('/media/esgomezm/bcss/egomez/3dprotucell_data/data/train/stack2im/inputs/raw_026.tif')
# output_im = cv2.imread('/media/esgomezm/bcss/egomez/3dprotucell_data/data/train/stack2im/labels/instance_ids_026.tif', cv2.IMREAD_ANYDEPTH)
# weights = cv2.imread('/media/esgomezm/bcss/egomez/3dprotucell_data/data/train/stack2im/weights/instance_ids_026_weight.tif', cv2.IMREAD_ANYDEPTH)
input_im = read_input_image('/home/esgomezm/Documents/3D-PROTUCEL/data/SemmanticSeg/train/raw_026.tif')
output_im = cv2.imread('/home/esgomezm/Documents/3D-PROTUCEL/data/SemmanticSeg/train_labels/instance_ids_026.tif', cv2.IMREAD_ANYDEPTH)
output_im[output_im>0]=1
weights = cv2.imread('/home/esgomezm/Documents/3D-PROTUCEL/data/train/stack2im/weights/instance_ids_026_weight.tif', cv2.IMREAD_ANYDEPTH)

input_patch = input_im[250:506,600:856]
output_patch = output_im[250:506,600:856]
weight_patch = weights[250:506,600:856]

weight_patch = weight_patch.reshape(1,256,256,1)
input_patch = input_patch.reshape(1,256,256,1)
output_patch = output_patch.reshape(1,256,256,1)

### Weighted binary cross entropy evaluation
# Initial output of the network
my_layer = model.layers[-4]
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=my_layer.output)

intermediate_output = intermediate_layer_model.predict([input_patch,weight_patch,output_patch])
# Lambda layer
lambda_layer = model.layers[-1]
lambda_layer_model = Model(inputs=model.input,
                                 outputs=lambda_layer.output)

lambda_layer_output = lambda_layer_model.predict([input_patch,weight_patch,output_patch])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(intermediate_output[0, :, :, 0])
plt.colorbar()
plt.title('Intermediate output (output of the network, before bce)')
plt.subplot(1,2,2)
plt.imshow(lambda_layer_output[0, :, :, 0])
plt.colorbar()
plt.title('Lambda layer (Loss value)')

## Calculate the logits, loss and gradients
# logits, identity_value, grads = obtain_gradients(model, [input_patch,weight_patch,output_patch], output_patch, identity_loss)
logits, identity_value, gradsBCE = obtain_gradients(model, input_patch, output_patch, binary_crossentropy)

logits_intermediate = intermediate_layer_model.predict([input_patch,weight_patch,output_patch])
bce_value = np.array(binary_crossentropy(output_patch, logits_intermediate))
weighted_loss_values = np.array(tf.multiply(bce_value, weight_patch[:, :, :, 0]))

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(logits_intermediate[0, :, :, 0])
plt.colorbar()
plt.title('logits intermediate (Conv2D)')

plt.subplot(2, 3, 2)
plt.imshow(np.array(identity_value)[0,:,:,0])
plt.colorbar()
plt.title('weighted loss_values (identity loss)')

plt.subplot(2, 3, 3)
plt.imshow(np.array(logits)[0, :, :, 0])
plt.colorbar()
plt.title('output of the weighted network')

plt.subplot(2, 3, 4)
plt.imshow(bce_value[0])
plt.colorbar()
plt.title('bce value manually calculated')

plt.subplot(2, 3, 5)
plt.imshow(weighted_loss_values[0])
plt.colorbar()
plt.title('weighted bce manually calculated')

plt.subplot(2, 3, 6)
plt.imshow(weight_patch[0, :, :, 0])
plt.colorbar()
plt.title('weights for the network')
# Use the gradient tape to automatically retrieve
# the gradients of the trainable variables with respect to the loss.

## Gradients for a foreground image
meanG = []
for i in range(len(grads)):
    G = np.array(grads[i])
    meanG.append(np.mean(G))
## Gradients for a foreground image
meanBCEG = []
for i in range(len(gradsBCE)):
    G = np.array(gradsBCE[i])
    meanBCEG.append(np.mean(G))
## Gradients for a background image

# Background:
input_patch = input_im[250:506,200:456]
output_patch = output_im[250:506,200:456]
weight_patch = weights[250:506,200:456]

weight_patch = weight_patch.reshape(1,256,256,1)
input_patch = input_patch.reshape(1,256,256,1)
output_patch = output_patch.reshape(1,256,256,1)

# logits, identity_value, grads = obtain_gradients(model, [input_patch,weight_patch,output_patch], output_patch, identity_loss)
logits, identity_value, grads = obtain_gradients(model, input_patch, output_patch, identity_loss)

meanBCEGB = []
for i in range(len(grads)):
    G = np.array(grads[i])
    meanBCEGB.append(np.mean(G))

plt.figure()
plt.plot(np.abs(meanG), 'r--')
plt.plot(np.abs(meanGB), 'b--')
plt.plot(np.abs(meanBCEG), color='red')
plt.plot(np.abs(meanBCEGB), 'b')
plt.show()



# ## Normal binary cross entropy evaluation and multiplication
#
# # Open a GradientTape to record the operations run
# # during the forward pass, which enables autodifferentiation.
# with tf.GradientTape() as tape:
#
#   # Run the forward pass of the layer.
#   # The operations that the layer applies
#   # to its inputs are going to be recorded
#   # on the GradientTape.
#   logits = model(input_patch, training=True)  # Logits for this minibatch
#
#   # Compute the loss value for this minibatch.
#   loss_value = binary_crossentropy(output_patch, logits)
#   weighted_loss_values = tf.multiply(loss_value, weight_patch[:,:,:,0])
#
#
# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(np.array(logits)[0,:,:,0])
# plt.colorbar()
# plt.title('output')
#
# plt.subplot(1,3,2)
# plt.imshow(np.array(loss_value)[0])
# plt.colorbar()
# plt.title('loss_values')
#
# plt.subplot(1,3,3)
# plt.imshow(weighted_loss_values[0,:,:])
# plt.colorbar()
# plt.title('weighted_loss_values')
#
# # Use the gradient tape to automatically retrieve
# # the gradients of the trainable variables with respect to the loss.
# grads = tape.gradient(loss_value, model.trainable_weights)
