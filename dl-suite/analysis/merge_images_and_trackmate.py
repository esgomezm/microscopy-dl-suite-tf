"""
Created on Fri March 26 2021

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import pandas as pd
import os
import SimpleITK as sitk
import numpy as np
import cv2
import time
import sys

def merge_seg_track(tracks_path, video_path, OUTPUTPATH, th=75, sigma=2):
    sys.stdout.write('Merging the tracks of video {}\n'.format(video_path))
    # Make sure that there is some tracking.
    # Tracking ID	Timepoint	Time (secs)	X pos	Y pos
    track = sitk.ReadImage(tracks_path)
    track = sitk.GetArrayFromImage(track)
    if np.sum(track) > 0:
        mask = sitk.ReadImage(video_path)
        mask = sitk.GetArrayFromImage(mask)
        mask = (mask > 0).astype(np.uint8)
        for t in range(mask.shape[0]):
            detections = np.zeros_like(mask[t], dtype=np.uint8)
            # Get connected components from the original binary masks
            idx, res = cv2.connectedComponents(mask[t])
            u = np.unique(track[t])
            if len(u) > 1:
                u = u[1:]
                for i in u:
                    cell_mask_id_S = res[track[t] == i]
                    if np.sum(cell_mask_id_S) > 0:
                        intersection = np.multiply(res, track[t] == i)
                        candidates = np.array(np.unique(intersection, return_counts=True))
                        candidates = candidates[:, 1:]
                        candidates = candidates[0, candidates[1] == np.max(candidates[1])][0]
                        indexes = np.where(res == candidates)
                        # give the track id to the mask of the cell
                        detections[indexes] = np.int(i)
                    else:
                        indexes = np.where(track[t] == i)
                        # give the track id to the mask of the cell
                        detections[indexes] = np.int(i)
            mask[t] = detections
        root = os.path.dirname(video_path)
        video_path = video_path.split(root)[-1]
        video_path = video_path[1:]
        sitk.WriteImage(sitk.GetImageFromArray(mask), os.path.join(OUTPUTPATH, video_path))

def process_track_dir(path2tracks, path2mask, OUTPUTPATH):
    """
    path2track: root folder with the fubfolders to process. It should have csv files with the tracks
    path2mask: root folde with the same structure as the tracks containing the binary segmentartions
    OUTPUTPATH: path to store the masks with unique labels for each track
    """
    # This function is specific to the experiments conducted and their subfolders
    folders = os.listdir(path2mask)
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    print(folders)
    for f in folders:
        print(f)
        if not os.path.exists(os.path.join(OUTPUTPATH, f)):
            os.makedirs(os.path.join(OUTPUTPATH, f))
        files = os.listdir(os.path.join(path2mask, f))

        _path2tracks = os.path.join(path2tracks, f)
        _path2mask = os.path.join(path2mask, f)
        _OUTPUTPATH = os.path.join(OUTPUTPATH, f)

        if not files[0].__contains__('.tif'):
            process_track_dir(_path2tracks, _path2mask, _OUTPUTPATH)
        else:
            for video in files:
                tracks_path = os.path.join(_path2tracks, "LblImg_instances_" + video)
                print('Tracks ' + tracks_path)
                print(' ')
                # Tracks preserve the detection video name and detections, the video of the mask.
                video_path = os.path.join(_path2mask, video)
                print('Processing {}'.format(video))
                print(' ')
                t0 = time.time()
                merge_seg_track(tracks_path, video_path, _OUTPUTPATH)
                t1 = time.time() - t0
                print('Finished. {} already processed.'.format(video_path))
                print(' ')
                print("Time elapsed: ", t1)  # CPU seconds elapsed (floating point)

def find_mitotic_cells(splits, non_splits, split_tracks):
    """
    splits: video in which cells following a mitotic event have the same label --> splits are tracked.
    non_splits: video in which ALL cells have a different label --> splits are not tracked.
    split_tracks: track information  from TrackMate with the number of mitosis per cell.
    """
    tracks = np.genfromtxt(split_tracks, delimiter=',', skip_header=1)
    if len(tracks.shape) == 1:
        TrackID = np.array([(tracks[0] + 1).astype(np.uint8)])
        TrackSPLITS = np.array([(tracks[3]).astype(np.uint8)])
    else:
        TrackID = (tracks[:, 0] + 1).astype(np.uint8)
        TrackSPLITS = (tracks[:, 3]).astype(np.uint8)
    if np.sum(TrackSPLITS) > 0:
        relation = []
        labels = TrackID[TrackSPLITS > 0]
        S = sitk.ReadImage(splits)
        S = sitk.GetArrayFromImage(S)
        NS = sitk.ReadImage(non_splits)
        NS = sitk.GetArrayFromImage(NS)
        for l in labels:
            merge = np.multiply(S == l, NS)
            merge = np.unique(merge)
            merge = merge[1:]
        # If it only merges with one cell, it will not be considered as mitosis
        if len(merge) > 1:
            for m in merge:
                relation.append([m, l])
    return relation

def add_split_info(path2dynamics, videos_splits, videos_no_splits, tracks, OUTPUTPATH):
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    folders = os.listdir(path2dynamics)
    print(folders)
    for f in folders:
        print(f)
        if not os.path.exists(os.path.join(OUTPUTPATH, f)):
            os.makedirs(os.path.join(OUTPUTPATH, f))
        _videos_splits = os.path.join(videos_splits, f)
        _videos_no_splits = os.path.join(videos_no_splits, f)
        _tracks = os.path.join(tracks, f)
        files = os.listdir(os.path.join(path2dynamics, f))
        if not files[0].__contains__('.xlsx'):
            add_split_info(os.path.join(path2dynamics, f), _videos_splits, _videos_no_splits,
                           _tracks, os.path.join(OUTPUTPATH, f))
        else:
            for excel in files:
                print('Processing excel {}'.format(excel))
                writer = pd.ExcelWriter(os.path.join(OUTPUTPATH, f, excel))
                xl = pd.ExcelFile(os.path.join(path2dynamics, f, excel))
                xl_sheet_names = xl.sheet_names
                for sheet in xl_sheet_names:
                    video_name = sheet + '.tif'
                    print('Processing video {}'.format(video_name))
                    splits = os.path.join(_videos_splits, video_name)
                    non_splits = os.path.join(_videos_no_splits, video_name)
                    split_tracks = os.path.join(_tracks, 'detections_' + video_name + 'tracks_properties.csv')
                    mitotic_relation = find_mitotic_cells(splits, non_splits, split_tracks)
                    df = pd.read_excel(path2dynamics, sheet)
                    df['Mitotic'] = 'No'
                    df['Split Track'] = 0
                    for r in mitotic_relation:
                        print('Found relation of cell {0} with split track {1}'.format(r[0], r[1]))
                        index = np.where(df['Cell'] == r[1])
                        df['Mitotic'].iloc[index] = 'Yes'
                        df['Split Track'].iloc[index] = r[1]
                    df.to_excel(writer, sheet_name=sheet)
            writer.close()     
