"""
Created on Wed Apr 8 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import numpy as np
import os
import cv2
import SimpleITK as sitk
from utils.utils import read_input_image, read_input_videos
import sys

def optimal_input_size(input_shape, halo, step):
    optimal_size = []
    for d in range(len(halo)):
        receptive = input_shape[d] + 2 * halo[d]
        extra = np.mod(receptive, step[d])
        if extra > 0:
            extra = (step[d] - extra)
        optimal_size.append(receptive + extra)
    return optimal_size

def mirror_border(image, sizeH, sizeW):
    im_size_h = image.shape[0]  # height
    im_size_w = image.shape[1]  # width
    new_image = np.zeros((3 * im_size_h, 3 * im_size_w), dtype=np.float32 )

    # img = Image.fromarray(image.astype(np.float))
    # mirror_img = ImageOps.mirror(img)
    # mirror_img = np.array(mirror_img)
    mirror_img = cv2.flip(image, 1)

    new_image[im_size_h:2 * im_size_h, :im_size_w] = mirror_img
    new_image[im_size_h:2 * im_size_h, im_size_w:2 * im_size_w] = image
    new_image[im_size_h:2 * im_size_h, 2 * im_size_w:] = mirror_img

    # img = img.rotate(180)
    img = cv2.rotate(image, cv2.ROTATE_180)
    new_image[:im_size_h:, :im_size_w] = img
    new_image[:im_size_h, 2 * im_size_w:] = img
    new_image[2 * im_size_h:, :im_size_w] = img
    new_image[2 * im_size_h:, 2 * im_size_w:] = img

    # mirror_img = ImageOps.mirror(img)
    # mirror_img = np.array(mirror_img)
    mirror_img = cv2.flip(img, 1)
    new_image[:im_size_h, im_size_w:2 * im_size_w] = mirror_img
    new_image[2 * im_size_h:, im_size_w:2 * im_size_w] = mirror_img

    index_h = int(np.round((new_image.shape[0] - sizeH) / 2))
    index_w = int(np.round((new_image.shape[1] - sizeW) / 2))
    new_image = new_image[index_h:index_h + sizeH, index_w:index_w + sizeW]

    return new_image


def patch_processing(im, model, dim_input, padding):
    # dim.data = [980 978 3];
    # dim.input = [400 400 3];
    # dim.output = [368 368 1];
    # Number of tails to process and cover the entire image
    dim_input = np.array(dim_input)
    dim_input = dim_input.astype(np.int)
    padding = np.array(padding)
    padding = padding.astype(np.int)
    dim_output = dim_input - 2*padding
    dim_output = dim_output * np.array((1, 1))
    padding = padding * np.array((1, 1))
    dim_input = dim_input * np.array((1, 1))
    if len(im.shape) == 3:
        dim_tiles = np.ceil(im.shape[1:] / dim_output)
    else:
        dim_tiles = np.ceil(im.shape / dim_output)
    dim_tiles = dim_tiles.astype(np.int)

    # Calculate the padding or offset
    # dim_min_offset = np.ceil((dim_output - dim_input) / 2)
    dim_min_offset = -padding
    dim_min_offset = dim_min_offset.astype(np.int)

    sizeH = dim_output[0] * dim_tiles[0] - 2 * dim_min_offset[0]
    sizeW = dim_output[1] * dim_tiles[1] - 2 * dim_min_offset[1]
    if len(im.shape) == 2:
        input_image = mirror_border(im, sizeH, sizeW)
    else:
        # change channels to the end to make things easier
        im = np.transpose(im, [1, 2, 0])
        input_image = mirror_border_video(im, sizeH, sizeW)

    if type(model.output_shape) is list:
        multi_output = len(model.output_shape)
        out_channels = 0
        for ch in range(multi_output):
            out_channels += model.output_shape[ch][-1]
    else:
        out_channels = model.output_shape[-1]
    if out_channels > 1:
        pred = np.empty((np.int(dim_tiles[0] * dim_output[0]),
                         np.int(dim_tiles[1] * dim_output[1]),
                         out_channels), 'single')
    else:
        pred = np.empty((np.int(dim_tiles[0] * dim_output[0]),
                        np.int(dim_tiles[1] * dim_output[1])), 'single')

    for i in range(0, dim_tiles[0]):  # width
        for j in range(0, dim_tiles[1]):  # height
            # obtain patch
            coord = 2 * i * dim_min_offset[0] + i * dim_input[0]
            aux = input_image[coord:coord + dim_input[0]]
            coord = 2 * j * dim_min_offset[1] + j * dim_input[1]
            aux = aux[:, coord:coord + dim_input[1]]

            # predict the patch
            aux = aux.astype(np.float32)
            if len(aux.shape) == 3: # time dimension
                aux = np.transpose(aux, [2, 0, 1])
            aux = np.expand_dims(aux, axis=[0, -1])

            # This step is done for CPU processing. If a GPU is available, collect first all the patches and
            # use model.predict to process them in once. Then, reconstruct the image.
            out = model.predict(aux, verbose=0)

            if type(out) is list:
                if len(aux.shape) == 3:
                    out[0] = out[0][:,0]
                    out[1] = out[1][:,0]
                out = np.concatenate([out[0], out[1]], axis=-1)
            if len(out.shape)>4:
                out = out[0]

            if padding[0] > 0:
                out = out[0, padding[0]:-padding[0]]
            else:
                out = out[0]
            if padding[1] > 0:
                out = out[:, padding[1]:-padding[1]]

            # fill matrix of predictions
            if out_channels == 1:
                pred[i * dim_output[0]:dim_output[0] + i * dim_output[0],
                    j * dim_output[1]:dim_output[1] + j * dim_output[1]] = out[:, :, 0]
            else:

                pred[i * dim_output[0]:dim_output[0] + i * dim_output[0],
                j * dim_output[1]:dim_output[1] + j * dim_output[1]] = out

            print("column - " + np.str(j))
        print("row - " + np.str(i))

    if pred.shape[0] > im.shape[0] or pred.shape[1] > im.shape[1]:
        offsetH = int(np.floor((pred.shape[0] - im.shape[0]) / 2))
        offsetW = int(np.floor((pred.shape[1] - im.shape[1]) / 2))
        pred = pred[offsetH:offsetH + im.shape[0],
               offsetW:offsetW + im.shape[1]]
    return pred

def model_prediction(path2im, output_dir, model, dim_input, padding, normalization):
    """
    This function reads all the images in path2im, processes them with the model, and saves the output of the model in
    a new directory in output_dir.
    :param path2im:
    :param output_dir:
    :param model: keras model to process the data
    :param dim_input: 2D array indicating the input size of the model in rows and columns
    :param padding: 2D array indicating the offset in rows and columns. Receptive field = dim_input-2*padding
    """
    print('Processing images from {}'.format(os.path.join(path2im, 'inputs')))
    names = os.listdir(os.path.join(path2im, 'inputs'))
    names.sort()

    # Create a directory to save the results
    # new_dir = os.path.join(output_dir, 'test_output')
    new_dir = output_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    print(dim_input)
    print(padding)
    for i, name in enumerate(names):
        print('Processing image {}'.format(name))
        im = read_input_image(os.path.join(path2im, 'inputs', name), normalization=normalization, uneven_illumination=False)
        if im is None:
            print('image {} was not loaded'.format(name))

        recons_frame = patch_processing(im, model, dim_input, padding)
        if len(recons_frame.shape)>2:
            recons_frame = np.transpose(recons_frame,[2,0,1])
        sitk.WriteImage(sitk.GetImageFromArray(recons_frame.astype(np.float32)), os.path.join(new_dir, name))

    # markers, bin_image = GetMaskFromProbMap(recons_frame, threshold, min_size, split_clusters, min_hole_size,
    #                                         top_hat_size)
    # del (recons_frame)
    # markers = images_resize2original(path2im, markers, dataset_num)
    #
    # markers_tps = np.zeros(markers.shape)
    # for l in range(np.max(markers)):
    #     b_round = skimage.filters.gaussian(np.float16(markers == l + 1), sigma=sigma,
    #                                        mode='mirror', preserve_range=True)
    #     b_round = np.int8(b_round > smooth_t)
    #
    #     markers_tps[b_round == 1] = (l + 1)
    # bin_image = markers_tps > 0
    # markers_tps = np.int16(markers_tps)
    # return bin_image, markers_tps, im

def mirror_border_video(image, sizeH, sizeW):

    # time dimension is assumed to be at the end
    im_size_h = image.shape[0]  # height
    im_size_w = image.shape[1]  # width
    new_image = np.zeros((3 * im_size_h, 3 * im_size_w, image.shape[-1]), dtype=np.float32)

    # img = Image.fromarray(image.astype(np.float))
    # mirror_img = ImageOps.mirror(img)
    # mirror_img = np.array(mirror_img)
    index_h = int(np.round((new_image.shape[0] - sizeH) / 2))
    index_w = int(np.round((new_image.shape[1] - sizeW) / 2))

    for t in range(image.shape[-1]):

        mirror_img = cv2.flip(image[..., t], 1)

        new_image[im_size_h:2 * im_size_h, :im_size_w, t] = mirror_img
        new_image[im_size_h:2 * im_size_h, im_size_w:2 * im_size_w, t] = image[..., t]
        new_image[im_size_h:2 * im_size_h, 2 * im_size_w:, t] = mirror_img

        # img = img.rotate(180)
        img = cv2.rotate(image[..., t], cv2.ROTATE_180)
        new_image[:im_size_h:, :im_size_w, t] = img
        new_image[:im_size_h, 2 * im_size_w:, t] = img
        new_image[2 * im_size_h:, :im_size_w, t] = img
        new_image[2 * im_size_h:, 2 * im_size_w:, t] = img

        # mirror_img = ImageOps.mirror(img)
        # mirror_img = np.array(mirror_img)
        mirror_img = cv2.flip(img, 1)
        new_image[:im_size_h, im_size_w:2 * im_size_w, t] = mirror_img
        new_image[2 * im_size_h:, im_size_w:2 * im_size_w, t] = mirror_img

    new_image = new_image[index_h:index_h + sizeH, index_w:index_w + sizeW]
    return new_image


def model_prediction_lstm(path2im, output_dir, PATH2VIDEOS, model, time_window, dim_input, padding, normalization):
    """
    This function reads all the images in path2im, processes them with the model, and saves the output of the model in
    a new directory in output_dir.
    :param path2im:
    :param output_dir:
    :param PATH2VIDEOS:
    :param model: keras model to process the data
    :param time_window: length of the videos that enter the network
    :param dim_input: 2D array indicating the input size of the model in rows and columns
    :param padding: 2D array indicating the offset in rows and columns. Receptive field = dim_input-2*padding
    """
    # Create a directory to save the results
    # new_dir = os.path.join(output_dir, 'test_output')
    new_dir = output_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    print(dim_input)
    print(padding)
    print('Processing images from {}'.format(path2im))
    # Get the names of the videos to analyze
    names = os.listdir(os.path.join(path2im, 'inputs'))
    names.sort()

    ## We will store all videos following the same order as for normal 2D inference so we can easily evaluate them
    files = [x for x in open(PATH2VIDEOS, "r")]
    files = files[1:]  # First row contains labels
    # file_relation = [[x.split(';')[0], x.split(';')[1][:-1]] for x in files]

    for i, video_name in enumerate(names):
        sys.stdout.write("\rProcessing video {0}:\n".format(video_name))
        im = read_input_videos(os.path.join(path2im, 'inputs', video_name), normalization=normalization)

        video_name = video_name.split('_stackreg_')[0] + '_' + video_name.split('_stackreg_')[1] #remove weird things in the name
        video_name = video_name.split('.')[0] # remove file format (.tif)

        frame_names = [x.split(';')[1] for x in files if x.split(';')[0].__contains__(video_name)]
        frame_names.sort()
        frame_names = [x.split('.')[0] for x in frame_names]
        if im is None:
            print('image {} was not loaded'.format(video_name))
        else:
            for t in range(im.shape[0]):
                print('Processing image {}'.format(frame_names[t]))
                if t < time_window - 1:
                    sub_x = np.zeros((time_window, im.shape[1], im.shape[2]))
                    extra_frames = time_window - (t + 1)
                    sub_x[extra_frames:] = im[:t + 1]
                    for f in range(extra_frames):
                        sub_x[f] = im[0]
                else:
                    sub_x = im[t - (time_window - 1):t + 1]
                recons_frame = patch_processing(sub_x, model, dim_input, padding)
                if len(recons_frame.shape) > 2:
                    recons_frame = np.transpose(recons_frame, [2, 0, 1])
                recons_frame = sitk.GetImageFromArray(recons_frame)
                output_name = os.path.join(new_dir, np.str(frame_names[t]) + '.tif')
                sitk.WriteImage(recons_frame,np.str(output_name))