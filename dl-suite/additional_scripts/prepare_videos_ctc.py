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
            
PATH = "C:/Users/egomez/Documents/data/"
INPUTPATH = os.path.join(PATH, "test", "labels")
OUTPUTPATH = os.path.join(PATH, "test", "CTC_evaluation")
PATH2VIDEOS = os.path.join(PATH, "test", "videos2im_relation.csv")
# export2CTC_format(INPUTPATH,OUTPUTPATH,PATH2VIDEOS)


