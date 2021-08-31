import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error


def identity_loss(y_true, y_pred):
    return y_pred


def obtain_gradients(model, input, output, loss_function):
    with tf.GradientTape() as tape:
      logits = model(input, training=True)
      loss_value = loss_function(output, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    return logits, loss_value, grads