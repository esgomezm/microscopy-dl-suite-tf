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
from analysis.prepare_videos4track import smooth_video
from analysis.morphology import protrusion_instances_morphology
from analysis.extract_dynamics import protrusions_mechanics_from_tracks

def merge_seg_track(tracks_path, video_path, OUTPUTPATH, th=75, sigma=2):
    sys.stdout.write('Merging the tracks of video {}\n'.format(video_path))
    tracks = np.genfromtxt(tracks_path, delimiter=',')

    # Make sure that there is some tracking.
    if len(tracks.shape) != 1:
        # Tracking ID	Timepoint	Time (secs)	X pos	Y pos
        tracks = tracks[1:]
        TrackID = (tracks[:, 0] + 1).astype(np.uint8)
        TrackINFO = (tracks[:, [1, 3, 4]] / [1, 0.802, 0.802]).astype(np.uint16)  # convert microns to pixels

        mask = sitk.ReadImage(video_path)
        mask = sitk.GetArrayFromImage(mask)
        mask = (mask > 0).astype(np.uint8)

        S = smooth_video(255 * mask, sigma=sigma)
        # Threshold the image to make it binary    
        S_clean = (S > th).astype(np.uint8)
        S = (S > 0).astype(np.uint8)
        for t in range(mask.shape[0]):
            detections = np.zeros_like(mask[t], dtype=np.uint8)
            # Get connected components from the original binary masks
            idx, res = cv2.connectedComponents(mask[t])
            # Get connected components from the smoothed image (to complement the masks)
            idx_S, res_S = cv2.connectedComponents(S[t])
            S_clean[t] = np.multiply(S_clean[t], res_S)
            # idx_Sclean, res_Sclean = cv2.connectedComponents(S_clean[t])
            # Positions
            positions = TrackINFO[TrackINFO[:, 0] == t][:, 1:]
            ID = TrackID[TrackINFO[:, 0] == t]

            # print at each position (keypoint) its corresponding track label
            for coord, i in zip(positions, ID):
                cell_mask_id_S = res_S[coord[1], coord[0]]
                if cell_mask_id_S > 0:
                    intersection = np.multiply(res, res_S == cell_mask_id_S)
                    if np.sum(intersection) > 0:
                        candidates = np.array(np.unique(intersection, return_counts=True))
                        candidates = candidates[:, 1:]
                        candidates = candidates[0, candidates[1] == np.max(candidates[1])][0]
                        indexes = np.where(res == candidates)
                        # give the track id to the mask of the cell
                        detections[indexes] = np.int(i)
                    else:
                        indexes = np.where(S_clean[t] == cell_mask_id_S)
                        # give the track id to the mask of the cell
                        detections[indexes] = np.int(i)

                        # Update the mask time frame with the information from the tracks
            mask[t] = detections
            # sys.stdout.write("\rTime: t = {0} processed\n".format(t))
            # sys.stdout.flush()
        root = os.path.dirname(video_path)
        video_path = video_path.split(root)[-1]
        video_path = video_path[1:]
        # video_path = video_path.split('/')[-1]
        # video_path = video_path.split("'\'")[-1]
        # print(os.path.join(OUTPUTPATH, video_path))
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

        _path2tracks = path2tracks + '/' + f
        _path2mask = os.path.join(path2mask, f)
        _OUTPUTPATH = os.path.join(OUTPUTPATH, f)

        if not files[0].__contains__('.tif'):
            process_track_dir(_path2tracks, _path2mask, _OUTPUTPATH)
        else:
            for video in files:
                tracks = "detections_" + video + "spots_properties.csv"
                tracks_path = _path2tracks + '/' + tracks
                print('Tracks ' + tracks_path)
                print(' ')
                # Tracks preserve the detection video name and detections, the video of the mask.
                # video = tracks.split('detections_')[-1]
                # video = video.split('spots_properties')[0]
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

def combine_protrusion_tracks(prot_track_path, cell_tracks_video_path, cell_excels_path, OUTPUTPATH, pixel_size=0.802):
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    files = os.listdir(prot_track_path)
    if not files[0].__contains__('.csv'):
        for f in files:
            print(f)
            _prot_track_path = os.path.join(prot_track_path, f)
            _cell_tracks_video_path = os.path.join(cell_tracks_video_path, f)
            _cell_excels_path = os.path.join(cell_excels_path, f)
            _OUTPUTPATH = os.path.join(OUTPUTPATH, f)
            combine_protrusion_tracks(_prot_track_path, 
                                      _cell_tracks_video_path, 
                                      _cell_excels_path, _OUTPUTPATH,
                                      pixel_size = pixel_size)
    else:
        excel_name = os.path.join(OUTPUTPATH, os.path.basename(cell_excels_path) + '.xlsx')
        cell_excels_path = os.path.join(cell_excels_path,
                                        os.path.basename(cell_excels_path) + '.xlsx')
        writer = pd.ExcelWriter(excel_name)
        for f in files:
            if f.__contains__('spots_properties.csv'):
                print(f)
                t0 = time.time()
                video_name = f.split('spots_properties.csv')[0]
                
                prot_info = protrusions_mechanics_from_tracks(video_name, prot_track_path, 
                                                 cell_tracks_video_path, 
                                                 cell_excels_path, 
                                                 pixel_size=pixel_size)
                sheet = f.split('.tif')[0]            
                prot_info.to_excel(writer, sheet_name=sheet)
                t1 = time.time() - t0
                print("Time elapsed: ", t1)
        writer.close()    
    













