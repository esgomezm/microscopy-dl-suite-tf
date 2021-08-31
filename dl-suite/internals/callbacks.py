from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback
import tensorflow as tf
import numpy as np
import os

class ImagesTensorboardCallback(Callback):
    def __init__(self, val_input, val_gt, logdir, n_images=5, step=20):
        """
        The callback displays a small set (n_images) of images, their prediction and corresponding ground truth.
        The callback needs to be initialized as:

        :param val_input: set of input images
        :param val_gt: set of ground truth for val_input
        :param logdir: output dir where the images will be stored. It is recommended './logs/tmp
        :param n_images:
        :param step: controls at which epoch the image is displayed
        """

        super(ImagesTensorboardCallback, self).__init__()
        self.logdir = logdir  # where the event files will be written
        self.val_input = val_input # validation data generator
        self.val_gt = val_gt # validation data generator
        self.n_images = n_images
        self.step = step
        self.seen = 0

    def on_epoch_end(self, epoch, logs=None):   # at the end of each epoch run this
        # n_images = int(np.ceil(np.divide(self.num_samples, self.batch_size)))
        n_images = self.n_images
        self.seen += 1
        if self.seen % self.step:
            x = self.val_input[:n_images]
            if len(x.shape) == 5:
                # lstm networks: shape: (samples, time, height, width, channels)
                # choose the last frame of the time sequence. It is the one segmented.
                x = x[:, -1]

            y_pred1 = self.model.predict(self.val_input[:n_images], batch_size=1) # get a reduced amount of images
            if len(self.val_gt) == 2:
                # segmentations + detections
                multioutput = True
                if len(self.val_gt[0].shape)==5:
                    # ground truth for lstm networks: shape: (samples, time, height, width, channels)
                    # surpress the time dimension
                    y1 = np.squeeze(self.val_gt[0][:n_images], axis=1)
                    y2 = np.squeeze(self.val_gt[1][:n_images], axis=1)
                    y_pred2 = np.squeeze(y_pred1[1], axis=1)
                    y_pred1 = np.squeeze(y_pred1[0], axis=1)
                else:
                    # ground truth for cnn networks: shape: (samples, height, width, channels)
                    y1 = self.val_gt[0][:n_images]
                    y2 = self.val_gt[1][:n_images]
                    y_pred2 = y_pred1[1]
                    y_pred1 = y_pred1[0]
                y_argmax = np.argmax(y_pred1, axis=-1)
            else:
                # segmentations
                multioutput = False
                if len(self.val_gt.shape) == 5:
                    # ground truth for lstm networks: shape: (samples, time, height, width, channels)
                    # eliminate the time dimension
                    y1 = np.squeeze(self.val_gt[:n_images], axis=1)
                    y_pred1 = np.squeeze(y_pred1, axis=1)
                else:
                    # ground truth for cnn networks: shape: (samples, height, width, channels)
                    y1 = self.val_gt[:n_images]
                y_argmax = np.argmax(y_pred1, axis=-1)

            y_argmax = 255*y_argmax.astype(np.uint8)
            y_argmax = np.expand_dims(y_argmax, axis=-1)
            y1 = 255*y1.astype(np.uint8)
            bc = np.expand_dims(y_pred1[..., 0], axis=-1)
            fr = np.expand_dims(y_pred1[..., 1], axis=-1)
            file_writer = tf.summary.create_file_writer(self.logdir) # creating the summary writer
            with file_writer.as_default():
                tf.summary.image("input/", x, step=epoch, max_outputs=n_images)
                tf.summary.image("mask/", y1, step=epoch, max_outputs=n_images)
                tf.summary.image("argmax/", y_argmax, step=epoch, max_outputs=n_images)
                tf.summary.image("0/", bc, step=epoch, max_outputs=n_images)
                tf.summary.image("1/", fr, step=epoch, max_outputs=n_images)
                if multioutput:
                    tf.summary.image("tips-gt/", y2, step=epoch, max_outputs=n_images)
                    tf.summary.image("tips/", y_pred2, step=epoch, max_outputs=n_images)

## Configure all the callbacks
# ------------------------------------------------------------------------
def initiate_callbacks(config, keras_model, last_epoch = None):
    callbacks = []

    if not os.path.exists(config.OUTPUTPATH + '/checkpoints/'):
        os.makedirs(config.OUTPUTPATH + '/checkpoints/')

    # accuracy meassure to monitor
    # change mode in reduce learning rate according to the monitored mesure.
    if config.datagen_type.__contains__('tips'):
        if config.cnn_name.__contains__("mobilenet_mobileunet_lstm"):
            # measureMonitor = 'time_distributed_1_jaccard_sparse3D'
            measureMonitor = 'val_time_distributed_1_jaccard_sparse3D'
        else:
            if config.cnn_name.__contains__("lstm"):
                # measureMonitor = 'slog_jaccard_sparse3D'
                measureMonitor = 'val_slog_jaccard_sparse3D'
            else:
                # measureMonitor = 'slog_jaccard_sparse'
                measureMonitor = 'val_slog_jaccard_sparse'
    else:
        if config.cnn_name.__contains__("lstm"):
            measureMonitor = 'val_jaccard_sparse3D' # 'jaccard_sparse3D'
        else:
            if config.model_lossfunction == 'sparse_cce':
                measureMonitor = 'val_jaccard_sparse' # 'jaccard_sparse' 

            elif config.model_lossfunction == 'multiple_output':
                measureMonitor = 'val_cell_jaccard_multiple_output' # 'cell_jaccard_multiple_output'

            elif config.model_lossfunction == 'weighted_bce':
                measureMonitor = 'loss'

            else:
                measureMonitor = 'val_jaccard_cce'  # 'jaccard_cce'
            # else:
            # measureMonitor = 'jaccard_multiple_output'  # 'val_jaccard_multiple_output'

    if config.callbacks_save_best_only == True:

        if last_epoch is not None:
            chpath = os.path.join(config.OUTPUTPATH, 'checkpoints',
                                  config.cnn_name + 'keep_from_{0}_best.hdf5'.format(last_epoch))
        else:
            chpath = os.path.join(config.OUTPUTPATH, 'checkpoints', 'weights_best.hdf5')

    elif last_epoch is not None:
        chpath = os.path.join(config.OUTPUTPATH, 'checkpoints', config.cnn_name + 'keep_from_{0}_'.format(last_epoch) + '{epoch:05d}.hdf5')
    else:
        chpath = os.path.join(config.OUTPUTPATH, 'checkpoints', config.cnn_name + '{epoch:05d}.hdf5')

    if config.callbacks_save_best_only == True:
        checkpoint = ModelCheckpoint(chpath,
                                     monitor=measureMonitor,
                                     save_weights_only=True,
                                     save_best_only=config.callbacks_save_best_only,
                                     mode='auto')
    else:
        checkpoint = ModelCheckpoint(chpath,
                                     monitor=measureMonitor,
                                     save_weights_only=True,
                                     mode='auto',
                                     save_freq=config.callbacks_save_freq)

    checkpoint.set_model(keras_model)
    callbacks.append(checkpoint)

    tensorboard = TensorBoard(log_dir=os.path.join(config.OUTPUTPATH,'logs'),
                              write_graph=True,
                              write_images=False,
                              histogram_freq=config.callbacks_tb_update_freq,
                              update_freq=config.callbacks_tb_update_freq)
    tensorboard.set_model(keras_model)
    callbacks.append(tensorboard)
    if config.callbacks_patience < config.train_max_epochs:
        reducelearning = ReduceLROnPlateau(monitor=measureMonitor,
                                           factor=0.5,
                                           patience=config.callbacks_patience,
                                           verbose=1,
                                           mode='auto', # 'min', 'max'
                                           min_delta=0.0001, # 0.05,
                                           cooldown=1,
                                           min_lr=0)
        reducelearning.set_model(keras_model)
        callbacks.append(reducelearning)
    return callbacks
