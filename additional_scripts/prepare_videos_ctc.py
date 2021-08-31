"""
Created on Tue May 11 2021

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import SimpleITK as sitk
import numpy as np
import os
import cv2
import sys
import pandas as pd
import xml.etree.ElementTree as ET


def export2CTC_format(INPUTPATH,OUTPUTPATH,PATH2VIDEOS):
    '''
    INPUTPATH: path where the single frames are placed.
    OUTPUTPATH: path where the sequences are stored.
    PATH2VIDEOS: path to a txt file in which each row contains the name of the video and the frame that belongs to that
                video:
        Labels row      Videos ; Frames
                        video 1; raw_001.tif\n
                        video 1; raw_002.tif\n
                        ...
                        video 1; raw_032.tif\n
                        video 2; raw_033.tif\n
    '''
    files = [x for x in open(PATH2VIDEOS, "r")]
    files = files[1:]  # First row contains labels
    files = [x.split(';')[0] for x in files]

    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)

    COUNT = 0
    SEQ = 1
    while COUNT < len(files):
        # Get the name of the original videos and the number of frames that it contains
        file_name = files[COUNT]  # video name
        # Calculate how many frames you need to process (it is said in the name of the video)
        start_time, end_time = file_name.split('_')[-1].split('-')
        start_time = np.int(start_time)
        end_time = np.int(end_time)
        sys.stdout.write("\rProcessing video {0}:\n".format(file_name))
        seqName = os.path.join(OUTPUTPATH,'{0:0>2}_RES'.format(SEQ))
        # Create the directory to store the results
        if not os.path.exists(seqName):
            os.makedirs(seqName)
        # Process the frames of this video
        video = sitk.ReadImage(os.path.join(INPUTPATH, file_name+'_Segmentationim-label.tif'))
        video = sitk.GetArrayFromImage(video)
        video = video.astype(np.uint16)
        labels = [l for l in np.unique(video) if l>0]
        track_info = [[l, np.min(np.where(video==l)[0]), np.max(np.where(video==l)[0]), 0] for l in labels]
        for i in range(len(video)):
            # Save the binary mask in the new sequence folder                
            sitk.WriteImage(sitk.GetImageFromArray(video[i].astype(np.uint16)),
                            os.path.join(seqName, 'mask{0:0>3}.tif'.format(i)))
            progress = (i + 1) / (len(video) + 1)
            text = "\r[{0}] {1}%".format("-" * (i + 1) + " " * (len(video) - i), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
        res_track = open(os.path.join(seqName,'res_track.txt'),'w')
        for line in track_info:
            res_track.write('{} {} {} {}'.format(line[0],line[1],line[2],line[3]))
            res_track.write("\n")
        res_track.close()
        # Update counters
        COUNT = COUNT + i + 1
        SEQ += 1
        sys.stdout.write("\n")  # this ends the progress bar
    print("Results organizing finished.")

def export2CTC_GT(INPUTPATH, OUTPUTPATH, PATH2VIDEOS):
    
    files = [x for x in open(PATH2VIDEOS, "r")]
    files = files[1:]  # First row contains labels
    files = [x.split(';')[0] for x in files]
    
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    
    COUNT = 0
    SEQ = 1
    while COUNT < len(files):
        # Get the name of the original videos and the number of frames that it contains
        file_name = files[COUNT]  # video name
        sys.stdout.write("\rProcessing video {0}:\n".format(file_name))
        video = sitk.ReadImage(os.path.join(INPUTPATH, file_name+'_Segmentationim-label.tif'))
        video = sitk.GetArrayFromImage(video)
        video = video.astype(np.uint16)        
        segName = os.path.join(OUTPUTPATH,'{0:0>2}_GT'.format(SEQ), 'SEG')
        traName = os.path.join(OUTPUTPATH,'{0:0>2}_GT'.format(SEQ), 'TRA')
        # Create the directory to store the results
        if not os.path.exists(segName):
            os.makedirs(segName)
        if not os.path.exists(traName):
            os.makedirs(traName)
        # Process the frames of this video
        for i in range(len(video)):   
            # Save the binary mask in the new sequence folder
            if SEQ==5 and i>=46:
                im = video[i]
                index = np.where(im==3)
                im[index]=0
                video[i]=im
            sitk.WriteImage(sitk.GetImageFromArray(video[i]),
                            os.path.join(segName, 'man_seg{0:0>3}.tif'.format(i)))
            # Save the binary mask in the new sequence folder
            sitk.WriteImage(sitk.GetImageFromArray(video[i]),
                            os.path.join(traName, 'man_track{0:0>3}.tif'.format(i)))
            # Update the counter
            progress = (i + 1) / (len(video)  + 1)
            text = "\r[{0}] {1}%".format("-" * (i + 1) + " " * (len(video) - i), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
        labels = [l for l in np.unique(video) if l>0]
        track_info = [[l, np.min(np.where(video==l)[0]), np.max(np.where(video==l)[0]), 0] for l in labels]
        man_track = open(os.path.join(traName,'man_track.txt'),'w')
        for line in track_info:
            man_track.write('{} {} {} {}'.format(line[0],line[1],line[2],line[3]))
            man_track.write("\n")
        man_track.close()
        # Update counters
        COUNT = COUNT + i + 1
        SEQ += 1
        sys.stdout.write("\n")  # this ends the progress bar



def export2CTC_tiptracks(INPUTPATH,OUTPUTPATH,PATH2VIDEOS, size = [983,985], pixel_size = 0.802):
    '''
    INPUTPATH: path to the excel files with the information about the tips
    OUTPUTPATH: path where the sequences are stored.
    PATH2VIDEOS: path to a txt file in which each row contains the name of the video and the frame that belongs to that
                video:
        Labels row      Videos ; Frames
                        video 1; raw_001.tif\n
                        video 1; raw_002.tif\n
                        ...
                        video 1; raw_032.tif\n
                        video 2; raw_033.tif\n
    The function reads each of the tip localization and creates a tracking file as the one in the cell tracking challenge 
    and the images with the uniquely labelled tips.
    '''
    files = [x for x in open(PATH2VIDEOS, "r")]
    files = files[1:]  # First row contains labels
    files = [x.split(';')[0] for x in files]

    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)

    COUNT = 0
    SEQ = 1
    while COUNT < len(files):
        # Get the name of the original videos and the number of frames that it contains
        file_name = files[COUNT]  # video name
        
        # Calculate how many frames you need to process (it is said in the name of the video)
        start_time, end_time = file_name.split('_')[-1].split('-')
        
        directories = file_name.split("_")
        if directories[0].__contains__("Video"):
            inner_dir = os.path.join("V", directories[1], directories[2])
            video_name = file_name.split(directories[2]+"_")[-1]
            video_name = video_name.split("_" + start_time + "-" + end_time)[0]
        else:
            inner_dir = os.path.join(directories[0], directories[1])
            video_name = file_name.split(directories[1]+"_")[-1]
            video_name = video_name.split("_" + start_time + "-" + end_time)[0]
        start_time = np.int(start_time) - 1 # Python counter starts at 0
        end_time = np.int(end_time) - 1 # Python counter starts at 0
        
        sys.stdout.write("\rProcessing video {0}:\n".format(file_name))
        seqName = os.path.join(OUTPUTPATH,'{0:0>2}_RES'.format(SEQ))
        # Create the directory to store the results
        if not os.path.exists(seqName):
            os.makedirs(seqName)
                 
        # video = sitk.ReadImage(os.path.join(INPUTPATH, file_name+'_Segmentationim-label.tif'))
        # video = sitk.GetArrayFromImage(video)
        # video = video.astype(np.uint16)
        # tracks_path = os.path.join(INPUTPATH, inner_dir, video_name + '_stackreg.tifspots_properties.csv')
        tracks_path = os.path.join(INPUTPATH, "TRACKED_TIPS" + video_name + '_stackreg' + '_' + file_name.split('_')[-1] + '.tifspots_properties.csv')
                
        # Columns: Tracking ID,Timepoint,Time (secs),X pos,Y pos
        tracks = np.genfromtxt(tracks_path, delimiter=',', skip_header=True)
        track_info = []
        if len(tracks.shape) > 1:
            time = tracks[:, 1]
            labels = tracks[:, 0]        
            X = tracks[:, 3]
            Y = tracks[:, 4]
            ID = np.unique(labels)
            
            
            for i in ID:
                index = np.where(labels == i)
                start_i = np.int(np.min(time[index]))
                end_i = np.int(np.max(time[index]))
                time_i = time[index]
                track_info.append([np.int(i+1), np.int(start_i), np.int(end_i), 0]) # track file for ctc evaluation
                if len(index[0]) < (end_i - start_i + 1):
                    x_i = X[index]
                    y_i = Y[index]      
                    for t in range(start_i, end_i+1):
                        if np.sum(time_i==t) == 0:
                            aux = np.array([i,  t, t*120, x_i[time_i==t-1][0], y_i[time_i==t-1][0]])
                            tracks = np.concatenate([tracks, [aux]], axis = 0)                
        else:
            track_info.append([np.int(tracks[0])+1, np.int(tracks[1]), np.int(tracks[1]), 0])
        if len(tracks.shape) == 1:
            time = [tracks[1]]
            labels = [tracks[0]]
            X = [tracks[3]]
            Y = [tracks[4]]
            ID = [np.unique(labels)]
        else:
            time = tracks[:, 1]
            labels = tracks[:, 0]        
            X = (np.round(tracks[:, 3]/pixel_size)).astype(np.int)
            Y = (np.round(tracks[:, 4]/pixel_size)).astype(np.int)
            ID = np.unique(labels)
        k = 0                
        for t in range(start_time, end_time + 1):            
            index_t = np.where(time == k)
            video_t = np.zeros(size, dtype = np.uint16)
            if len(index_t[0])>0:
                labels_t = labels[index_t]
                ID_t = np.unique(labels_t)
                x_t = X[index_t]
                y_t = Y[index_t]
                for i in ID_t:
                    index_i = np.where(labels_t==i)

                    x_i = np.int(x_t[index_i][0])
                    y_i = np.int(y_t[index_i][0])
                    aux = np.zeros(size, dtype = np.uint16)
                    aux = cv2.circle(aux,tuple((x_i,y_i)),5,(1), -1)
                    video_t = video_t + (i+1)*aux
            sitk.WriteImage(sitk.GetImageFromArray(video_t.astype(np.uint16)),
                        os.path.join(seqName, 'mask{0:0>3}.tif'.format(t-start_time)))    
            progress = (t + 1) / (end_time - start_time + 1)
            text = "\r[{0}] {1}%".format("-" * (t + 1) + " " * (end_time-start_time + 1 - t), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
            k += 1
        res_track = open(os.path.join(seqName,'res_track.txt'),'w')
        for line in track_info:
            res_track.write('{} {} {} {}'.format(line[0],line[1],line[2],line[3]))
            res_track.write("\n")
        res_track.close()
        # Update counters
        COUNT = COUNT + end_time-start_time + 1
        SEQ += 1
        sys.stdout.write("\n")  # this ends the progress bar
    print("Results organizing finished.")


def export2CTC_GT_Tips(KEYPOINTPATH, TRACKPATH, OUTPUTPATH, PATH2VIDEOS):
    
    files = [x for x in open(PATH2VIDEOS, "r")]
    files = files[1:]  # First row contains labels
    files = [x.split(';')[0] for x in files]
    
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    
    COUNT = 0
    SEQ = 1
    while COUNT < len(files):
        # Get the name of the original videos and the number of frames that it contains
        file_name = files[COUNT]  # video name
        
        time = file_name.split("_")[-1]
        video_name = file_name.split(time)[0] + "stackreg_" + time + ".tif"
        sys.stdout.write("\rProcessing video {0}:\n".format(video_name))        
        
        video = sitk.ReadImage(os.path.join(KEYPOINTPATH, video_name))
        video = sitk.GetArrayFromImage(video)
        video = video.astype(np.uint16)   
        video = (video>1).astype(np.uint16) # Keypoints == 1 belong to cell centroid.
        
        track = sitk.ReadImage(os.path.join(TRACKPATH, video_name))
        track = sitk.GetArrayFromImage(track)
        index = np.where(track>0)
        track[index] = track[index] - (np.min(track)-1)

        video = np.multiply(track, video)        
        traName = os.path.join(OUTPUTPATH,'{0:0>2}_GT'.format(SEQ), 'TRA')
        # Create the directory to store the results        
        if not os.path.exists(traName):
            os.makedirs(traName)
        # Process the frames of this video
        for i in range(video.shape[0]):   
            # Save the binary mask in the new sequence folder
            labels = np.unique(video[i])
            frame = np.zeros((video.shape[1], video.shape[2]), dtype = np.uint16)
            for l in labels:
                aux = np.zeros_like(frame, dtype=np.uint16)
                y, x = np.where(video[i]==l)
                x = np.round(np.mean(x))
                y = np.round(np.mean(y))
                aux = cv2.circle(aux,tuple((int(x),int(y))),5,(1), -1)
                frame = frame + l*aux
            
            # Save the binary mask in the new sequence folder
            sitk.WriteImage(sitk.GetImageFromArray(frame),
                            os.path.join(traName, 'man_track{0:0>3}.tif'.format(i)))
            
            video[i] = frame # Update video to create the track txt file
                  
            # Update the counter
            progress = (i + 1) / (len(video)  + 1)
            text = "\r[{0}] {1}%".format("-" * (i + 1) + " " * (len(video) - i), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
        
        labels = [l for l in np.unique(video) if l>0]
        track_info = [[l, np.min(np.where(video==l)[0]), np.max(np.where(video==l)[0]), 0] for l in labels]             
        
        man_track = open(os.path.join(traName,'man_track.txt'),'w')
        
        for line in track_info:
            man_track.write('{} {} {} {}'.format(line[0],line[1],line[2],line[3]))
            man_track.write("\n")
        man_track.close()
        # Update counters
        COUNT = COUNT + i + 1
        SEQ += 1
        sys.stdout.write("\n")  # this ends the progress bar


def TrackMate3ISBI(INPUTPATH, OUTPUTPATH):
    files = os.listdir(INPUTPATH)
    
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    for file_name in files:
        
        tree = ET.parse(os.path.join(INPUTPATH, file_name))
        root = tree.getroot()
        
        
        root_ISBI = ET.Element("root")
        trackISBI = ET.SubElement(root_ISBI, "TrackContestISBI2012") 
        for r in root.findall("particle"):
            trackISBI.append(r)
        tree = ET.ElementTree(root_ISBI)
        tree.write(os.path.join(OUTPUTPATH, file_name.split(".tif")[0] + '.xml'))
        
    
    
    

def GT2XML_PTC(KEYPOINTPATH, TRACKPATH, OUTPUTPATH):
    files = os.listdir(KEYPOINTPATH)
    
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    
    for file_name in files:
        
        time = file_name.split("_")[-1]
        video_name = file_name.split("stackreg_")[0] + time
        sys.stdout.write("\rProcessing video {0}:\n".format(video_name))        
        
        video = sitk.ReadImage(os.path.join(KEYPOINTPATH, file_name))
        video = sitk.GetArrayFromImage(video)
        video = video.astype(np.uint16)   
        video = (video>1).astype(np.uint16) # Keypoints == 1 belong to cell centroid.
        
        track = sitk.ReadImage(os.path.join(TRACKPATH, file_name))
        track = sitk.GetArrayFromImage(track)
        index = np.where(track == 32768)
        track[index] = 0
        index = np.where(track>0)
        track[index] = track[index] - (np.min(track[index])-1)

        video = np.multiply(track, video)        
        trackName = os.path.join(OUTPUTPATH,video_name.split('.tif')[0] + '.xml')
        root = ET.Element("root")
        trackISBI = ET.SubElement(root, "TrackContestISBI2012") 
        labels = np.unique(video)
        labels = [l for l in labels if l>0]
        for i in labels:
            
            particle = ET.SubElement(trackISBI, "particle")
            
            index_t = np.where(video==i)
            time_index = np.unique(index_t[0])
            for t in time_index:                
                y, x = np.where(video[t]==i)
                x = np.round(np.mean(x))
                y = np.round(np.mean(y))
                ET.SubElement(particle, "detection", z="0", y = np.str(y), x=np.str(x), t=np.str(t))
            # Update the counter
            progress = (i + 1) / (len(labels)  + 1)
            text = "\r[{0}] {1}%".format("-" * (i + 1) + " " * (len(labels) - i), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
            
        tree = ET.ElementTree(root)
        tree.write(trackName)
            











# Localize tips for test data
from analysis.prepare_videos4track import prepare_test_localizations
PATH = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/FULL-VIDEOS/mobilenet_mobileunet_lstm_tips_large_pure_v01/corrected_normalization/"
# INPUTPATH = os.path.join(PATH, "test", "results")
EXCELS = os.path.join(PATH, "mitotic_information")
path2stacks = os.path.join(PATH, "merge_trackmate_nomitosis")

OUTPUTPATH = os.path.join(PATH, "test", "TRACKED_TIPS")
PATH2VIDEOS = os.path.join(PATH, "test", "videos2im_relation.csv")
# prepare_test_localizations(EXCELS, path2stacks, PATH2VIDEOS, OUTPUTPATH, min_len=20, pixel_size=0.802)

    
PATH = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/FULL-VIDEOS/mobilenet_mobileunet_lstm_tips_large_pure_v01/corrected_normalization/"
# INPUTPATH = os.path.join(PATH, "test", "results")
# INPUTPATH = os.path.join(PATH, "trackmate_protrusions")
# INPUTPATH = os.path.join(PATH, "test", "TRAKCMATE_TIPS")
INPUTPATH = os.path.join(PATH, "test", "TrackMate_XML")
OUTPUTPATH = os.path.join(PATH, "test", "TrackMate_XML_ISBI")
# OUTPUTPATH = os.path.join(PATH, "test", "CTC_evaluation_TIPS")
PATH2VIDEOS = os.path.join(PATH, "test", "videos2im_relation.csv")
# export2CTC_tiptracks(INPUTPATH,OUTPUTPATH,PATH2VIDEOS, size = [983,985], pixel_size = 1)
# export2CTC_format(INPUTPATH,OUTPUTPATH,PATH2VIDEOS)
# TrackMate3ISBI(INPUTPATH, OUTPUTPATH)




# INPUTPATH = os.path.join(PATH, "labels")
# OUTPUTPATH = os.path.join(PATH, "CTC_evaluation")
# export2CTC_GT(INPUTPATH, OUTPUTPATH, PATH2VIDEOS)
INPUTPATH = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/DATA/data_corrected/test"
KEYPOINTPATH = os.path.join(INPUTPATH, "ProtCellTruePositive")
TRACKPATH = os.path.join(INPUTPATH, "tracks")
PATH = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/FULL-VIDEOS/mobilenet_mobileunet_lstm_tips_large_pure_v01/corrected_normalization/"
OUTPUTPATH = os.path.join(PATH, "test", "TrackMate_XML_GT_TP")
# export2CTC_GT_Tips(KEYPOINTPATH, TRACKPATH, OUTPUTPATH, PATH2VIDEOS)
GT2XML_PTC(KEYPOINTPATH, TRACKPATH, OUTPUTPATH)





#### Curate some data
# SEQ = 5
# labels = [2,3, 4]
# location = {"3": [[19, 32, 619, 949], [39, 53, 584, 958], [56, 65, 577, 963], [79, 94, 648, 902]],
#             "2": [[26, 29, 630, 890]],
#             "4": [[34, 40, 588, 956]]}

# path2im = os.path.join(OUTPUTPATH, '{0:0>2}_RES'.format(SEQ))
# for l in labels:
#     for info in location[str(l)]:
#         for t in range(info[0], info[1]+1):
#             im = sitk.ReadImage(os.path.join(path2im, 'mask{0:0>3}.tif'.format(t)))
#             im = sitk.GetArrayFromImage(im)            
#             im = cv2.circle(im,tuple((info[3], info[2])),5,(l), -1)
#             sitk.WriteImage(sitk.GetImageFromArray(im), 
#                             os.path.join(path2im, 'mask{0:0>3}.tif'.format(t)))
            

# SEQ = 8
# labels = [2]
# location = {"2": [[38, 43, 703, 665]]}

# path2im = os.path.join(OUTPUTPATH, '{0:0>2}_RES'.format(SEQ))
# for l in labels:
#     for info in location[str(l)]:
#         for t in range(info[0], info[1]+1):
#             im = sitk.ReadImage(os.path.join(path2im, 'mask{0:0>3}.tif'.format(t)))
#             im = sitk.GetArrayFromImage(im)            
#             im = cv2.circle(im,tuple((info[3], info[2])),5,(l), -1)
#             sitk.WriteImage(sitk.GetImageFromArray(im), 
#                             os.path.join(path2im, 'mask{0:0>3}.tif'.format(t)))
            
            

# SEQ = 9
# labels = [2]
# location = {"2": [[5, 7, 732, 763]]}

# path2im = os.path.join(OUTPUTPATH, '{0:0>2}_RES'.format(SEQ))
# for l in labels:
#     for info in location[str(l)]:
#         for t in range(info[0], info[1]+1):
#             im = sitk.ReadImage(os.path.join(path2im, 'mask{0:0>3}.tif'.format(t)))
#             im = sitk.GetArrayFromImage(im)            
#             im = cv2.circle(im,tuple((info[3], info[2])),5,(l), -1)
#             sitk.WriteImage(sitk.GetImageFromArray(im), 
#                             os.path.join(path2im, 'mask{0:0>3}.tif'.format(t)))


            
            
            