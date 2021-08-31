"""
Created on Wed Sept 23 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import numpy as np
import sys
from internals.tiling_strategy import mirror_border, optimal_input_size
from utils.utils import load_mat_files, get_normal_fce, read_input_videos
import SimpleITK as sitk
from internals.postprocessing import post_processing

def process_video(input_path, model, halo=[95, 95], step=[16, 16], batch_size=5, normalization='PERCENTILE',
                  time_window=5, reduce_tips=True, clean_masks=True):
    if input_path.__contains__('.mat'):
        video = load_mat_files(input_path)
        if video.dtype.name == 'uint16':
          video = video.astype(np.float32)
          video = video / 65535
        else:
            video = video.astype(np.float32)
            video = video / 255
        normalization_fce = get_normal_fce(normalization)
        video = normalization_fce(video)
    else:
        # video = sitk.ReadImage(input_path)
        # video = sitk.GetArrayFromImage(video)
        video = read_input_videos(input_path, normalization=normalization)
    time_length = video.shape[0]
    input_shape = video[0].shape
    optimal_size = optimal_input_size(input_shape, halo, step)
    sys.stdout.write('\rOptimal size for one frame: {0}\n'.format(optimal_size))
    sys.stdout.write("\rProcessing video {0}:\n".format(input_path))
    sys.stdout.write('\rRaw input video shape: {0}\n'.format(video.shape))
    for t in range(time_length):
        # output = normalization_fce(video[t])
        output = mirror_border(video[t], optimal_size[0], optimal_size[1])
        output = np.expand_dims(output, axis=0)
        if t == 0:
            output_video = np.copy(output)
        else:
            output_video = np.concatenate((output_video, output), axis=0)
    sys.stdout.write('\rShape of the video with proper input size: {0}\n'.format(output_video.shape))
    del video

    # Inference
    if len(model.input_shape) > 4:
        if type(model.output_shape) is list:
            MASK, TIPS = lstm_model_inference(output_video, model, time_window=time_window, batch_size=batch_size,
                                              reduce_tips=reduce_tips)
        else:
            MASK = lstm_model_inference(output_video, model, time_window=time_window, batch_size=batch_size,
                                        reduce_tips=reduce_tips)
    else:
        if type(model.output_shape) is list:
            MASK, TIPS = model_inference_2d(output_video, model, batch_size=batch_size)
        else:
            MASK = model_inference_2d(output_video, model, batch_size=batch_size)
    h = np.int(np.floor((optimal_size[0] - input_shape[0]) / 2))
    w = np.int(np.floor((optimal_size[1] - input_shape[1]) / 2))
    MASK = MASK[:, h: h + input_shape[0], w: w + input_shape[1]]
    if clean_masks == True:
        ## The postprocssing executes the ordered operations:
        # 1.- morphological clossing
        # 2.- holes filling
        # 3.- remove small particles
        # 4.- remove objects from boundary
        for m in range(MASK.shape[0]):
            MASK[m] = post_processing(MASK[m])
    if type(model.output_shape) is list:
        TIPS = TIPS[:, h: h + input_shape[0], w: w + input_shape[1]]
        return MASK, TIPS
    else:
        return MASK


def predict_batch_at_time_lstm(video, model, batch_size, t, time_window):
    # sys.stdout.write('\rBatch from {0} to {1} time-points\n'.format(t, np.min((video.shape[0], t + batch_size))))
    for b in range(batch_size):
        if b == 0 and t < video.shape[0]:
            BATCH = video[t:t + time_window]
            BATCH = np.expand_dims(BATCH, axis=0)
        elif t < video.shape[0]:
            BATCH = np.concatenate((BATCH, np.expand_dims(video[t:t + time_window], axis=0)), axis=0)
        t += 1
    BATCH = np.expand_dims(BATCH, axis=-1)
    return model.predict(BATCH, batch_size=batch_size)


def lstm_model_inference(video, model, time_window=5, batch_size=5, reduce_tips=True):
    sys.stdout.write('\rInference on a video of size: {0}\n'.format(video.shape))
    time_points = video.shape[0]
    # Create the a fake time window for the first frame of the video
    time_padding = np.ones((time_window - 1, video.shape[1], video.shape[2]))
    time_padding = video[0] * time_padding # [4, 985, 983]
    # Concatenate the first frames at the begining of the video to infer the first frame
    video = np.concatenate((time_padding, video), axis=0)
    del time_padding
    iterations = np.int(np.ceil(time_points / batch_size))
    sys.stdout.write('\rIterations to compute: {0}\n'.format(iterations))
    t = 0
    for i in range(iterations):
        sys.stdout.write('\rProcessing batch {0} of {1}\n'.format(i, iterations))
        model_output = predict_batch_at_time_lstm(video, model, batch_size, t, time_window)
        t += batch_size
        # Check if it is multioutput
        if type(model_output) is list:
            mask = np.copy(model_output[0])
            tips = np.copy(model_output[1])
            # This tensors has shape (batch, time, height, width, 1)
            tips = np.squeeze(tips, axis=-1)
            tips = np.squeeze(tips, axis=1)
            if reduce_tips == True:
                tips = 255 * tips
                tips = tips.astype(np.uint8)
        else:
            mask = model_output
        del model_output

        # Check if it has multiple channels and do softmax to get labels
        if mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1)
            mask = mask.astype(np.uint8)
        else:
            mask = np.squeeze(mask, axis=-1)

        # Reduce time dimension which is in the second dim. This tensors have shape (batch, time, height, width, channels).
        mask = np.squeeze(mask, axis=1)

        if i == 0:
            FINAL_MASK = np.copy(mask)
            if type(model.output_shape) is list:
                FINAL_TIPS = np.copy(tips)
            
        else:
            FINAL_MASK = np.concatenate((FINAL_MASK, mask), axis=0)
            if type(model.output_shape) is list:
                FINAL_TIPS = np.concatenate((FINAL_TIPS, tips), axis=0)
                del tips
        del mask

    if type(model.output_shape) is list:
        return FINAL_MASK, FINAL_TIPS
    else:
        return FINAL_MASK


def predict_batch_at_time_2d(video, model, batch_size, t):
    # sys.stdout.write('\rBatch from {0} to {1} time-points\n'.format(t, np.min((video.shape[0], t + batch_size))))
    BATCH = video[t:np.min((video.shape[0], t + batch_size))]
    BATCH = np.expand_dims(BATCH, axis=-1)
    if BATCH.ndim == 3:
        BATCH = np.expand_dims(BATCH, axis=0)
    return model.predict(BATCH, batch_size=batch_size)


def model_inference_2d(video, model, batch_size=5, reduce_tips=True):
    sys.stdout.write('\rInference on a video of size: {0}\n'.format(video.shape))
    time_points = video.shape[0]
    iterations = np.int(np.ceil(time_points / batch_size))
    sys.stdout.write('\rIterations to compute: {0}\n'.format(iterations))
    t = 0
    for i in range(iterations):
        sys.stdout.write('\rProcessing batch {0} of {1}\n'.format(i, iterations))
        model_output = predict_batch_at_time_2d(video, model, batch_size, t)
        t += batch_size
        # Check if it is multioutput
        if type(model_output) is list:
            mask = np.copy(model_output[0])
            tips = np.copy(model_output[1])
            tips = np.squeeze(tips, axis=-1)
            if reduce_tips == True:
                tips = 255 * tips
                tips = tips.astype(np.uint8)
        else:
            mask = np.copy(model_output)

        del model_output

        # Check if it has multiple channels and do softmax to get labels
        if mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1)
            mask = mask.astype(np.uint8)
        else:
            mask = np.squeeze(mask, axis=-1)

        # Build final video
        if i == 0:
            FINAL_MASK = np.copy(mask)
            if type(model.output_shape) is list:
                FINAL_TIPS = np.copy(tips)
        else:
            FINAL_MASK = np.concatenate((FINAL_MASK, mask), axis=0)
            if type(model.output_shape) is list:
                FINAL_TIPS = np.concatenate((FINAL_TIPS, tips), axis=0)
                del tips
        del mask

    if type(model.output_shape) is list:
        return FINAL_MASK, FINAL_TIPS
    else:
        return FINAL_MASK