"""
Created on Tue Apr 21 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import os
import SimpleITK as sitk
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import sys
import shutil
import subprocess
import cv2
from internals.postprocessing import post_processing

def build_videos(INPUTPATH,OUTPUTPATH,PATH2VIDEOS):
    '''
    INPUTPATH: path where the single frames are placed.
    OUTPUTPATH: pathe where the reconstructed videos will be stored.
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
    file_relation = [[x.split(';')[0], x.split(';')[1][:-1]] for x in files]

    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)

    COUNT = 1
    while COUNT <= len(file_relation):
        # Get the name of the original videos and the number of frames that it contains
        file_name = file_relation[COUNT][0] # video name
        # Calculate how many frames you need to process (it is said in the name of the video)
        start_time, end_time = file_name.split('_')[-1].split('-')
        start_time = np.int(start_time)
        end_time = np.int(end_time)
        sys.stdout.write("\rProcessing video {0}:\n".format(file_name))

        for i in range(end_time-start_time+1):
            # Load the single frame
            frame_name = os.path.join(INPUTPATH, 'raw_{0:0>3}.tif'.format(int(COUNT+i)))
            frame = sitk.ReadImage(frame_name)
            frame = sitk.GetArrayFromImage(frame)
            if len(frame.shape)>2:
                # sitk images have channels at the beginning (axis = 0).
                frame = np.argmax(frame, axis=0)
            if i == 0:
                video = frame.reshape((1,frame.shape[0], frame.shape[1]))
            else:
                video = np.concatenate((video,frame.reshape((1,frame.shape[0], frame.shape[1]))), axis=0)
            progress = (i+1)/(end_time-start_time+1)
            text = "\r[{0}] {1}%".format("-" * (i+1) + " " * (end_time-start_time-i), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
        COUNT = COUNT + i + 1
        if video.dtype == 'int64':
            sitk.WriteImage(sitk.GetImageFromArray(video.astype(np.uint16)),
                            os.path.join(OUTPUTPATH, file_name + '.tif'))
        else:
            sitk.WriteImage(sitk.GetImageFromArray(video.astype(np.float32)),
                            os.path.join(OUTPUTPATH, file_name + '.tif'))
        sys.stdout.write("\n")  # this ends the progress bar
    print("All videos have been reconstructed")

def jaccard_index (y_pred, y_true):
    intersection = y_true*y_pred
    intersection = intersection.astype(np.float)
    union = y_true + y_pred - intersection
    union = union.astype(np.float)
    if np.sum(union) == 0.0:
        return 1.0
    else:
        return np.sum(intersection)/np.sum(union)

def dice_coeff (y_pred, y_true):
    intersection = y_true*y_pred
    intersection = intersection.astype(np.float)
    union = y_true + y_pred
    union = union.astype(np.float)
    if np.sum(union) == 0.0:
        return 1.0
    else:
        return 2*np.sum(intersection)/np.sum(union)

def get_accuracy_measures_videos(PATH2RESULTS, PATH2GT, PATH2STORE, PATH2VIDEOS, EXPERIMENT):
    '''
    PATH2RESULTS: path where the results are placed.
    PATH2GT: path to the ground truth data.
    FILE2STORE: path to a file where the results will be saved.
    PATH2VIDEOS: path to a txt file in which each row contains the name of the video and the frame that belongs to that
                video:
        Labels row      Videos ; Frames
                        video 1; raw_001.tif\n
                        video 1; raw_002.tif\n
                        ...
                        video 1; raw_032.tif\n
                        video 2; raw_033.tif\n
    '''

    # FILES = os.listdir(PATH2VIDEOS)

    # Read the file containing the relation between the initial videos and the individual frames.
    files = [x for x in open(PATH2VIDEOS, "r")]
    files = files[1:]  # First row contains labels
    file_relation = [[x.split(';')[0], x.split(';')[1][:-1]] for x in files]

    COUNT = 1
    if not os.path.exists(PATH2STORE):
        os.makedirs(PATH2STORE)

    while COUNT <= len(file_relation):
        # Get the name of the original videos and the number of frames that it contains
        file_name = file_relation[COUNT][0]  # video name
        # Calculate how many frames you need to process (it is said in the name of the video)
        start_time, end_time = file_name.split('_')[-1].split('-')
        start_time = np.int(start_time)
        end_time = np.int(end_time)
        # TODO: check how many frames belong to the same video and reconstruct them
        sys.stdout.write("\rProcessing video {0}:\n".format(file_name))
        for i in range(end_time - start_time + 1):
            # Load the single frame
            frame_name = os.path.join(PATH2RESULTS, 'raw_{0:0>3}.tif'.format(int(COUNT + i)))
            frame = sitk.ReadImage(frame_name)
            frame = sitk.GetArrayFromImage(frame)
            gt_name = os.path.join(PATH2GT, 'instance_ids_{0:0>3}.tif'.format(int(COUNT + i)))
            gt = sitk.ReadImage(gt_name)
            gt = sitk.GetArrayFromImage(gt)
            gt[gt > 0] = 1
            frame[frame < 0.5] = 0
            frame[frame > 0] = 1
            # Calculate the mean accuracy measures for each video
            if i == 0:
                HM = directed_hausdorff(frame, gt)[0]
                HM_total = HM

                JAC = jaccard_index(frame.flatten(), gt.flatten())
                JAC_total = JAC

                DICE = dice_coeff(frame.flatten(), gt.flatten())
                DICE_total = DICE
            else:
                HM = (HM*i + directed_hausdorff(frame, gt)[0])/(i+1)
                HM_total = directed_hausdorff(frame, gt)[0]+HM_total

                JAC = (JAC*i + jaccard_index(frame.flatten(), gt.flatten()))/(i+1)
                JAC_total = jaccard_index(frame.flatten(), gt.flatten()) + JAC_total

                DICE = (DICE*i + dice_coeff(frame.flatten(), gt.flatten()))/(i+1)
                DICE_total = dice_coeff(frame.flatten(), gt.flatten()) + DICE_total
            progress = (i + 1) / (end_time - start_time + 1)
            text = "\r[{0}] {1}%".format("-" * (i + 1) + " " * (end_time - start_time - i), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
        sys.stdout.write("\n")  # this ends the progress bar
        COUNT = COUNT + i + 1
        # Store the measures for each video in a csv file
        if os.path.exists(os.path.join(PATH2STORE, 'video_accuracies.csv')):
            with open(os.path.join(PATH2STORE, 'video_accuracies.csv'), mode='a') as file_:
                file_.write(file_name + ";{};{};{}".format(HM, JAC, DICE))
                file_.write("\n")
        else:
            fields = "video; Hausdorff-distance; Jaccard index; Dice coeffficient"
            with open(os.path.join(PATH2STORE, 'video_accuracies.csv'), 'w') as file_:
                file_.write(fields)
                file_.write("\n")
                file_.write(file_name + ";{};{};{}".format(HM, JAC, DICE))
                file_.write("\n")
    HM_total = HM_total/COUNT
    JAC_total = JAC_total / COUNT
    DICE_total = DICE_total / COUNT
    # Store the measures for each video in a csv file
    if os.path.exists(os.path.join(PATH2STORE, 'accuracies.csv')):
        with open(os.path.join(PATH2STORE, 'accuracies.csv'), mode='a') as file_:
            file_.write(EXPERIMENT + ";{};{};{}".format(HM_total, JAC_total, DICE_total))
            file_.write("\n")
    else:
        fields = "Experiment; Hausdorff distance; Jaccard index; Dice coefficient"
        with open(os.path.join(PATH2STORE, 'accuracies.csv'), 'w') as file_:
            file_.write(fields)
            file_.write("\n")
            file_.write(EXPERIMENT + ";{};{};{}".format(HM_total, JAC_total, DICE_total))
            file_.write("\n")
    print("All videos have been evaluated")

def build_videos_CTC(INPUTPATH,OUTPUTPATH,PATH2VIDEOS, threshold = 0.5, tips = False, postprocessing = True):
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
    threshold: float value between 0 and 1 to binarize the output of the network when it has only one output channel.
    tips: boolean indicating whether the last channel of the predicted image corresponds to the tip localization.
            False by default.
    '''
    files = [x for x in open(PATH2VIDEOS, "r")]
    files = files[1:]  # First row contains labels
    file_relation = [[x.split(';')[0], x.split(';')[1][:-1]] for x in files]

    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)

    COUNT = 1
    SEQ = 1
    while COUNT <= len(file_relation):
        # Get the name of the original videos and the number of frames that it contains
        file_name = file_relation[COUNT][0]  # video name
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
        seqCount = 0
        for i in range(end_time - start_time + 1):
            # Load the image prediction
            frame_name = os.path.join(INPUTPATH, 'raw_{0:0>3}.tif'.format(int(COUNT + i)))
            frame = sitk.ReadImage(frame_name)
            frame = sitk.GetArrayFromImage(frame)
            # Binarize the output of the network
            if len(frame.shape) > 2:
                if tips==True:
                    # sitk images have channels at the beginning (axis = 0).
                    frame = frame[:-1]
                frame = np.argmax(frame, axis=0)
            else:
                frame = frame > threshold
            frame = frame.astype(np.uint8)
            if postprocessing==True:
                frame = post_processing(frame)
            # Save the binary mask in the new sequence folder
            sitk.WriteImage(sitk.GetImageFromArray(frame.astype(np.uint16)),
                            os.path.join(seqName, 'mask{0:0>3}.tif'.format(seqCount)))
            # Update the counter
            seqCount += 1
            progress = (i + 1) / (end_time - start_time + 1)
            text = "\r[{0}] {1}%".format("-" * (i + 1) + " " * (end_time - start_time - i), progress * 100)
            sys.stdout.write(text)
            sys.stdout.flush()
        # Update counters
        COUNT = COUNT + i + 1
        SEQ += 1
        sys.stdout.write("\n")  # this ends the progress bar
    print("Results organizing finished.")

def ctc_evaluation(PATH2RESULTS, PATH2GT, PATH2CTCmeasure = "../EvaluationSoftware/Linux"):
    """
    Args:
        PATH2RESULTS:
        PATH2GT:
        PATH2STORE:

    Returns:

    """
    videos = os.listdir(PATH2GT)
    videos.sort()
    CTCseg = []
    for i in videos:
        # copy the ground truth in the results directory.
        shutil.copytree(os.path.join(PATH2GT,i), os.path.join(PATH2RESULTS,i))
        # check whether the ground truth is empty (CTC measures do not accound for empty frames)
        gt_im = cv2.imread(os.path.join(PATH2RESULTS,i,'SEG/man_seg000.tif'), cv2.IMREAD_ANYDEPTH)
        if np.sum(gt_im) == 0:
            seqSeg = []
            for frames in os.listdir(os.path.join(PATH2RESULTS,i,'SEG')):
                # We assume that empty videos are from the beginning till the end empty
                gt_im = cv2.imread(os.path.join(PATH2RESULTS,i,'SEG', frames), cv2.IMREAD_ANYDEPTH)
                res_im = cv2.imread(os.path.join(PATH2RESULTS, i.split('_')[0]+'_RES', 'mask' + frames.split('man_seg')[-1]), cv2.IMREAD_ANYDEPTH)
                if np.sum(np.max(gt_im)+np.max(res_im)) > 0:
                    seqSeg.append(0.)
                else:
                    seqSeg.append(1.)
            seg = np.mean(seqSeg)
            print('Sequence {0} has SEG measure: {1}'.format(i, seg))
        else:
            # call the CTC segmentation evaluation measure
            cmd = (os.path.join(PATH2CTCmeasure, "SEGMeasure"),
                   PATH2RESULTS, i.split('_')[0], '3')
            seg_measure = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].strip()
            seg = seg_measure.decode('utf8')
            # if seg == "No ground truth object found!":
            #     seg = "SEG measure: 1.00000"
            print('Sequence {0} has {1}'.format(i,seg))
            # Extract the accuracy measure from the resulting message
            seg = np.float(seg.split(':')[-1])
        CTCseg.append(seg)
        shutil.rmtree(os.path.join(PATH2RESULTS,i), ignore_errors=True)

    print("Evaluation done!")
    print("Averaged SEG for all the videos: {0}".format(np.mean(CTCseg)))
    return np.mean(CTCseg)

