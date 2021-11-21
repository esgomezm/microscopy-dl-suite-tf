"""
Created on Wed April 15 2021

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import SimpleITK as sitk
import numpy as np
import time
import os
from analysis.morphology import centroidCalculator, connect_connectedcomponents, geodesic_distance_transform
from plantcv import plantcv as pcv
import pandas as pd
pd.options.mode.chained_assignment = None
import cv2
import warnings
warnings. filterwarnings("ignore")


def print_protrusions_localization(frame, cell, centroid, pixel_size, min_len):
    skeleton = pcv.morphology.skeletonize(mask=cell)
    tip_pts_mask = pcv.morphology.find_tips(skel_img=skeleton, mask=None, label="default")
    tip_pts_mask = tip_pts_mask.astype(bool)

    ## Geodesic distance transform for the length of the protrusions
    if cell.shape[0] != cell.shape[1]:
        aux = np.zeros((np.max(cell.shape), np.max(cell.shape)), dtype=np.uint8)
        aux[:cell.shape[0], :cell.shape[1]] = cell

    mask = ~aux.astype(bool)
    m = np.ma.masked_array(aux.astype(np.float32), mask)
    # centroid is the same as we add pixels at the end of the column and rows
    m[centroid[0], centroid[1]] = 0
    gdt = geodesic_distance_transform(m)
    # Trasnform the measures to microns
    gdt = pixel_size * gdt

    tips_dist = np.where(tip_pts_mask == True)
    for tip_y, tip_x in zip(tips_dist[0], tips_dist[1]):
        if gdt[tip_y, tip_x] < min_len:
            tip_pts_mask[tip_y, tip_x] = False
        else:
            cv2.circle(frame, tuple((tip_x, tip_y)), 0, (1), -1)
    return frame


def create_protrusions_localization(video_path, OUTPUTPATH, pixel_size=0.802, min_len=20):
        print('********************************')
        print('Processing {}'.format(video_path))
        start_time = time.time()
        track_video = sitk.ReadImage(video_path)
        track_video = sitk.GetArrayFromImage(track_video)
        labels = np.unique(track_video)
        if sum(labels) > 0:
            for t in range(len(track_video)):
                print(t)
                frame = track_video[t]
                frame_labels = np.unique(frame)
                frame_labels = frame_labels[1:]
                frame_labels = [l for l in frame_labels]
                frame_tips = np.zeros_like(frame)
                for l in frame_labels:
                    # We need to remove the mitosis to avoid problems with geodesic distance
                    print('Cell with ID {}'.format(l))
                    cell = (frame == l).astype(np.uint8)
                    idx, concomp = cv2.connectedComponents(cell)
                    if idx > 2:
                        cell = connect_connectedcomponents(concomp, thickness=2)
                    C = centroidCalculator(cell)
                    frame_tips = print_protrusions_localization(frame_tips, cell, C, pixel_size, min_len)
                track_video[t] = frame_tips

            # Store the video
            sitk.WriteImage(sitk.GetImageFromArray(track_video),OUTPUTPATH)
            print("--- %s seconds ---" % (time.time() - start_time))

def process_videos_prot_localization(path2stacks, OUTPUTPATH, min_len=20, pixel_size=0.802):
    # This function is specific to the experiments conducted and their subfolders
    folders = os.listdir(path2stacks)
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    print(folders)
    for f in folders:
        print(f)
        if not f.__contains__('.tif'):
            process_videos_prot_localization(os.path.join(path2stacks, f),
                                             os.path.join(OUTPUTPATH, f),
                                             min_len=min_len, pixel_size=pixel_size)
        else:
            print('Processing {}'.format(path2stacks))
            t0 = time.time()
            create_protrusions_localization(os.path.join(path2stacks, f),
                                             os.path.join(OUTPUTPATH, f),
                                            pixel_size=pixel_size,
                                            min_len=min_len)
            t1 = time.time() - t0
            print('{} already processed.'.format(path2stacks))
            print("Time elapsed: ", t1)  # CPU seconds elapsed (floating point)


def print_protrusions(frame, cell, centroid, pixel_size, min_len):
    skeleton = pcv.morphology.skeletonize(mask=cell)
    tip_pts_mask = pcv.morphology.find_tips(skel_img=skeleton, mask=None, label="default")
    tip_pts_mask = tip_pts_mask.astype(bool)

    ## Geodesic distance transform for the length of the protrusions
    if cell.shape[0] != cell.shape[1]:
        aux = np.zeros((np.max(cell.shape), np.max(cell.shape)), dtype=np.uint8)
        aux[:cell.shape[0], :cell.shape[1]] = cell

    mask = ~aux.astype(bool)
    m = np.ma.masked_array(aux.astype(np.float32), mask)
    # centroid is the same as we add pixels at the end of the column and rows
    m[centroid[0], centroid[1]] = 0
    gdt = geodesic_distance_transform(m)
    # Trasnform the measures to microns
    gdt = pixel_size * gdt

    tips_dist = np.where(tip_pts_mask == True)
    for tip_y, tip_x in zip(tips_dist[0], tips_dist[1]):
        if gdt[tip_y, tip_x] < min_len:
            tip_pts_mask[tip_y, tip_x] = False
        else:
            cv2.circle(frame, tuple((tip_x, tip_y)), 3, (100), -1)
    tip_num = np.sum(tip_pts_mask.astype(np.uint8))
    if tip_num > 0:
        # tips_diam, branch_diam = diameterCalculator(cell, skeleton, tip_pts_mask, tip_labels, pixel_size)
        branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=skeleton, mask=cell, label="default")
        B = np.where(branch_pts_mask > 0)
        for by, bx in zip(B[0], B[1]):
            cv2.circle(frame, tuple((bx, by)), 3, (200), -1)
    return frame

def print_keypoints(video_path, pixel_size=0.802, min_len=20):
    print('********************************')
    print('Processing {}'.format(video_path))
    track_video = sitk.ReadImage(video_path)
    track_video = sitk.GetArrayFromImage(track_video)
    if np.sum(track_video) > 0:
        for t in range(len(track_video)):
            print(t)
            frame = track_video[t]
            frame_labels = np.unique(frame)
            frame_labels = frame_labels[1:]
            for l in frame_labels:
                # We need to remove the mitosis to avoid problems with geodesic distance
                print('Cell with ID {}'.format(l))
                cell = (frame == l).astype(np.uint8)
                idx, concomp = cv2.connectedComponents(cell)
                if idx > 2:
                    cell = connect_connectedcomponents(concomp, thickness=2)
                S = np.sum(cell) * (pixel_size ** 2)
                if S > 0:
                    C = centroidCalculator(cell)
                    cv2.circle(frame, tuple((C[1], C[0])), 5, (50), -1)
                    frame = print_protrusions(frame, cell, C, pixel_size, min_len)
            track_video[t] = frame
    return track_video