"""
Created on Fri March 5 2021

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import sys
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter
from internals.postprocessing import post_processing
from analysis.morphology import centroidCalculator, connect_connectedcomponents, geodesic_distance_transform
import PIL
from PIL import ImageDraw
import os
import cv2
import time
import pandas as pd
from plantcv import plantcv as pcv


def mean_axis_uint(A, B):
    A = A.astype(np.uint16)
    for t in range(A.shape[0]):
        A[t] += B[t].astype(np.uint16)

    A = (0.5 * A).astype(np.uint8)
    return A


def smooth_video(video, sigma=2):
    # Time is assumed to be in the first axis (axis=0)
    # Process (rows,time) kind of frames
    print('Smooth colum-wise')
    colwise = []
    for c in range(video.shape[-1]):
        colwise.append((gaussian_filter(video[..., c], sigma)))
    colwise = np.array(colwise)
    colwise = np.transpose(colwise, [1, 2, 0])
    print('Smooth row-wise')
    rowwise = []
    video = np.transpose(video, [1, 0, 2])
    for r in range(video.shape[0]):
        rowwise.append((gaussian_filter(video[r], sigma)))
    del video
    rowwise = np.array(rowwise)
    rowwise = np.transpose(rowwise, [1, 0, 2])

    smooth = mean_axis_uint(rowwise, colwise)

    return smooth


def detect_connected_components(im, r):
    """
    im is assumed to be binary uint8
    r: radious of the circle to draw
    """
    # Calculate the connected components

    label_im = sitk.GetImageFromArray(im)
    label_im = sitk.GetArrayFromImage(sitk.ConnectedComponent(label_im))
    labels = np.unique(label_im)
    labels = labels[-1]

    ### create a blank image to print only annotations onto
    PIL_im = PIL.Image.fromarray(label_im)
    # define color scale of image
    mode = 'L'  # for color image “L” (luminance)
    # create blank image, define dimentaions as equal to those of the original image
    annotation_im = PIL.Image.new(mode, PIL_im.size)
    # define it as surface to draw on
    draw = ImageDraw.Draw(annotation_im)

    # Draw a point on each connected component element    
    for i in range(1, labels):

        # Centroid coordinates of the object
        indexes = np.where(label_im == i)
        if len(indexes[0]) >= 100:
            y = int(np.sum(indexes[0]) / len(indexes[0]))  # xcoordinate of centroid
            x = int(np.sum(indexes[1]) / len(indexes[1]))  # ycoordinate of centroid
            r = 2
            # draw the point with the value of the index
            draw.ellipse((x - r, y - r, x + r, y + r), fill=i, outline=i)
    return np.array(annotation_im)


def detect_connected_components_cv2(im, min_size=100):
    """
    im is assumed to be binary uint8
    r: radious of the circle to draw
    min_size: Small connected components are avoided. Equivalent to 0.64*min_size microns^2
    """
    # Calculate the connected components
    # label connected components
    idx, res = cv2.connectedComponents(im)
    detections = np.zeros_like(im, dtype=np.uint8)
    # Draw a point on each connected component element    
    for i in range(1, idx):
        # Centroid coordinates of the object
        cell = (res == i).astype(np.uint8)
        if np.sum(cell) >= min_size:
            C = centroidCalculator(cell)
            y = C[0]
            x = C[1]
            # y = int(np.sum(indexes[0]) / len(indexes[0]))   # xcoordinate of centroid
            # x = int(np.sum(indexes[1]) / len(indexes[1]))   # ycoordinate of centroid       
            cv2.circle(detections, tuple((x, y)), 0, (1), -1)
    return detections


def cell_detection(video_path, video_name, output_path, th, r, sigma, min_size, STORE_SMOOTH, POSTPROCESS=False):
    sys.stdout.write("\rProcessing video:\n")
    binary_video = sitk.ReadImage(video_path)
    binary_video = sitk.GetArrayFromImage(binary_video)
    if POSTPROCESS:
        for t in range(binary_video.shape[0]):
            # We increase the minimum size of the binary segmentations as
            # small objects are usually non-focused or partly focused cells, 
            # or residuals of some protrusions that we want to avoid.
            binary_video[t] = post_processing(binary_video[t], min_size=150, remove_objects_boundary=False)
            text = "\rPostprocessing{0}".format(" " + "." * np.mod(t, 4))
            sys.stdout.write(text)
            sys.stdout.flush()
    # the video is assumed to have values 0-1. To keep working with 8 bits and
    # reduce some memory consumption, we multiply it by 255. 
    # Otherwise, when filtering, as there are no values between 0 and 1 it 
    # will not process correctly the images.

    S = smooth_video(255 * binary_video, sigma=sigma)
    del binary_video
    # Threshold the image to make it binary
    S = (S > th).astype(np.uint8)
    points = []
    for t in range(S.shape[0]):
        points.append(detect_connected_components_cv2(S[t], min_size=min_size))
        text = "\rCell detection{0}".format(" " + "." * np.mod(t, 4))
        sys.stdout.write(text)
        sys.stdout.flush()
    # video_name = video_path.split("/")[-1]
    sitk.WriteImage(sitk.GetImageFromArray(np.array(points)), os.path.join(output_path, 'detections_' + video_name))
    if STORE_SMOOTH:
        sitk.WriteImage(sitk.GetImageFromArray(np.array(S)), os.path.join(output_path, 'smooth_' + video_name))

    sys.stdout.write('Detections of video {0} have been stored at {1}\n'.format(video_path, output_path))


def instance_segmentation(video_path, video_name, output_path, th, sigma, min_size, POSTPROCESS=False):
    sys.stdout.write("\rProcessing video:\n")
    binary_video = sitk.ReadImage(video_path)
    binary_video = sitk.GetArrayFromImage(binary_video)
    if POSTPROCESS:
        for t in range(binary_video.shape[0]):
            # We increase the minimum size of the binary segmentations as
            # small objects are usually non-focused or partly focused cells,
            # or residuals of some protrusions that we want to avoid.
            binary_video[t] = post_processing(binary_video[t], min_size=150, remove_objects_boundary=False)
            text = "\rPostprocessing{0}".format(" " + "." * np.mod(t, 4))
            sys.stdout.write(text)
            sys.stdout.flush()
    # the video is assumed to have values 0-1. To keep working with 8 bits and
    # reduce some memory consumption, we multiply it by 255.
    # Otherwise, when filtering, as there are no values between 0 and 1 it
    # will not process correctly the images.
    S = smooth_video(255 * binary_video, sigma=sigma)
    del binary_video
    # Threshold the image to make it binary
    S = (S > th).astype(np.uint8)
    points = []
    for t in range(S.shape[0]):
        # Connected components
        idx, res = cv2.connectedComponents(S[t])
        for i in range(1, idx):
            # Centroid coordinates of the object
            cell = (res == i).astype(np.uint8)
            if np.sum(cell) < min_size:
                res[res == i] = 0
        points.append(res)
        text = "\rCell detection{0}".format(" " + "." * np.mod(t, 4))
        sys.stdout.write(text)
        sys.stdout.flush()
    sitk.WriteImage(sitk.GetImageFromArray(np.array(points).astype(np.uint16)),
                    os.path.join(output_path, 'instances_' + video_name))
    sys.stdout.write('Detections of video {0} have been stored at {1}\n'.format(video_path, output_path))


def process_videos(path2stacks, OUTPUTPATH, th=0, sigma=2, min_size=100, POSTPROCESS=True):
    """
    Full videos were not postprocessed by any closing or hole-filling operation
    
    STORE-SMOOTH: 
    will just store the smooth along the time so we can see why 
    there might be some errors. 
    
    Small objects will not be removed during the detection as those can be the 
    result of the smoothing and might help to track vibrating cells.
    """
    # This function is specific to the experiments conducted and their subfolders
    folders = os.listdir(path2stacks)
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    print(folders)
    for f in folders:
        print(f)
        if not os.path.exists(os.path.join(OUTPUTPATH, f)):
            os.makedirs(os.path.join(OUTPUTPATH, f))
        files = os.listdir(os.path.join(path2stacks, f))
        if not files[0].__contains__('.tif'):
            process_videos(os.path.join(path2stacks, f), os.path.join(OUTPUTPATH, f), th=th, sigma=sigma,
                           min_size=min_size, POSTPROCESS=POSTPROCESS)
        else:
            for video in files:
                video_path = os.path.join(path2stacks, f, video)
                print('Processing {}'.format(video))

                t0 = time.time()
                instance_segmentation(video_path, video, os.path.join(OUTPUTPATH, f), th, sigma, min_size,
                                      POSTPROCESS=POSTPROCESS)
                # cell_detection(video_path, video, os.path.join(OUTPUTPATH, f), th, r, sigma, min_size, STORE_SMOOTH, POSTPROCESS=POSTPROCESS)
                t1 = time.time() - t0
                print('{} already processed.'.format(video_path))
                print("Time elapsed: ", t1)  # CPU seconds elapsed (floating point)
