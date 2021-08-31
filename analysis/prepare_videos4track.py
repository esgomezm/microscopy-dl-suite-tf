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

def mean_axis_uint(A,B):
    A = A.astype(np.uint16)
    for t in range(A.shape[0]):
        A[t] += B[t].astype(np.uint16)

    A = (0.5*A).astype(np.uint8)
    return A

def smooth_video(video, sigma=2):
    # Time is assumed to be in the first axis (axis=0)
    #Process (rows,time) kind of frames
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
    #Calculate the connected components    
    
    label_im = sitk.GetImageFromArray(im)
    label_im = sitk.GetArrayFromImage(sitk.ConnectedComponent(label_im))    
    labels = np.unique(label_im)
    labels = labels[-1] 
    
    ### create a blank image to print only annotations onto
    PIL_im = PIL.Image.fromarray(label_im)
    # define color scale of image
    mode = 'L' # for color image “L” (luminance)
    # create blank image, define dimentaions as equal to those of the original image
    annotation_im = PIL.Image.new(mode, PIL_im.size)
    # define it as surface to draw on
    draw = ImageDraw.Draw(annotation_im)
    
    # Draw a point on each connected component element    
    for i in range(1,labels):        
        
        # Centroid coordinates of the object
        indexes = np.where(label_im==i)
        if len(indexes[0])>=100:
            y = int(np.sum(indexes[0]) / len(indexes[0]))   # xcoordinate of centroid
            x = int(np.sum(indexes[1]) / len(indexes[1]))   # ycoordinate of centroid
            r = 2 
            # draw the point with the value of the index
            draw.ellipse((x-r, y-r, x+r, y+r), fill = i , outline = i)
    return np.array(annotation_im) 


def detect_connected_components_cv2(im, min_size=100):
    """
    im is assumed to be binary uint8
    r: radious of the circle to draw
    min_size: Small connected components are avoided. Equivalent to 0.64*min_size microns^2
    """
    #Calculate the connected components    
    # label connected components
    idx, res = cv2.connectedComponents(im)
    detections = np.zeros_like(im, dtype=np.uint8)    
    # Draw a point on each connected component element    
    for i in range(1,idx):   
        # Centroid coordinates of the object
        cell = (res==i).astype(np.uint8)        
        if np.sum(cell)>=min_size:
            C = centroidCalculator(cell)
            y = C[0]
            x = C[1]
            # y = int(np.sum(indexes[0]) / len(indexes[0]))   # xcoordinate of centroid
            # x = int(np.sum(indexes[1]) / len(indexes[1]))   # ycoordinate of centroid       
            cv2.circle(detections,tuple((x,y)),0,(1), -1)
    return detections



def cell_detection(video_path,video_name, output_path, th, r, sigma, min_size, STORE_SMOOTH, POSTPROCESS=False):
    
    sys.stdout.write("\rProcessing video:\n")
    
    binary_video = sitk.ReadImage(video_path)
    binary_video = sitk.GetArrayFromImage(binary_video)
    if POSTPROCESS:
        for t in range(binary_video.shape[0]):
            ## We increase the minimum size of the binary segmentations as 
            # small objects are usually non-focused or partly focused cells, 
            # or residuals of some protrusions that we want to avoid.
            binary_video[t] = post_processing(binary_video[t], min_size = 150, remove_objects_boundary=False)
            text = "\rPostprocessing{0}".format(" " + "."*np.mod(t,4))
            sys.stdout.write(text)
            sys.stdout.flush()
    # the video is assumed to have values 0-1. To keep working with 8 bits and
    # reduce some memory consumption, we multiply it by 255. 
    # Otherwise, when filtering, as there are no values between 0 and 1 it 
    # will not process correctly the images.
    
    S = smooth_video(255*binary_video, sigma=sigma)
    del binary_video
    # Threshold the image to make it binary
    S = (S>th).astype(np.uint8)
    points = []
    for t in range(S.shape[0]):     
        points.append(detect_connected_components_cv2(S[t], min_size=min_size))
        text = "\rCell detection{0}".format(" " + "."*np.mod(t,4))
        sys.stdout.write(text)
        sys.stdout.flush()
    # video_name = video_path.split("/")[-1]
    sitk.WriteImage(sitk.GetImageFromArray(np.array(points)), os.path.join(output_path, 'detections_' + video_name))
    if STORE_SMOOTH:
        sitk.WriteImage(sitk.GetImageFromArray(np.array(S)), os.path.join(output_path, 'smooth_' + video_name))

    sys.stdout.write('Detections of video {0} have been stored at {1}\n'.format(video_path, output_path))


def process_videos(path2stacks, OUTPUTPATH, th=0, r=1, sigma=2, min_size=100, STORE_SMOOTH=False, POSTPROCESS=True):
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
            process_videos(os.path.join(path2stacks, f),os.path.join(OUTPUTPATH, f))
        else:
            for video in files:
                video_path = os.path.join(path2stacks, f, video)
                print('Processing {}'.format(video))
                
                t0= time.time()
                cell_detection(video_path, video, os.path.join(OUTPUTPATH, f), th, r, sigma, min_size, STORE_SMOOTH, POSTPROCESS=POSTPROCESS)
                t1 = time.time() - t0
                print('{} already processed.'.format(video_path))
                print("Time elapsed: ", t1) # CPU seconds elapsed (floating point)
    
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
    gdt = pixel_size*gdt

    tips_dist = np.where(tip_pts_mask == True) 
    for tip_y, tip_x in zip(tips_dist[0], tips_dist[1]):
        if gdt[tip_y, tip_x]<min_len:
            tip_pts_mask[tip_y, tip_x] = False
        else:
            cv2.circle(frame,tuple((tip_x,tip_y)),0,(1), -1)            
    return frame


def create_protrusions_localization(path2excels, video_path, OUTPUTPATH, mitotic = False, pixel_size = 0.802, min_len = 20, min_track = 0):
    if mitotic:
        mitotic_flag = "Yes"
    else:
        mitotic_flag = "No"
    xl = pd.ExcelFile(path2excels)
    xl_sheet_names = xl.sheet_names
    for sheet in xl_sheet_names:
        video_name = sheet.split('.xlsx')[0] + '.tif'
        df = pd.read_excel(path2excels, sheet)    
        df = df[df['Mitotic'] == mitotic_flag]    
        
        # Discard short tracks
        labels = np.unique(np.asarray(df['Cell']))    
        printing_labels = []
        for c in labels:
            track = df[df['Cell']==c] 
            index = np.where(track["Cell size"]>0)
            track_len = len(track.iloc[index[0][0]:index[0][-1]+1])
            if track_len >= min_track:
                printing_labels.append(c)
        # Analyse videos and print protrusions frame by frame
        if len(printing_labels) > 0:
            print('********************************')
            print('Processing {}'.format(video_path))
            start_time = time.time()
            track_video = sitk.ReadImage(os.path.join(video_path, video_name))
            track_video = sitk.GetArrayFromImage(track_video)
            for t in range(len(track_video)):
                print(t)
                frame = track_video[t]
                frame_labels = np.unique(frame)
                frame_labels = frame_labels[1:]
                
                frame_labels = [l for l in frame_labels if l in printing_labels]            
                frame_tips = np.zeros_like(frame)
                for l in frame_labels:                    
                    # We need to remove the mitosis to avoid problems with geodesic distance
                    print('Cell with ID {}'.format(l))
                    cell = (frame==l).astype(np.uint8)                
                    idx, concomp = cv2.connectedComponents(cell)
                    if idx>2:
                        cell = connect_connectedcomponents(concomp, thickness = 2)
                    C = centroidCalculator(cell)
                    frame_tips = print_protrusions_localization(frame_tips, cell, C, pixel_size, min_len)
                track_video[t] = frame_tips
            
            #Store the video
            sitk.WriteImage(sitk.GetImageFromArray(track_video), os.path.join(OUTPUTPATH, video_name))
            print("--- %s seconds ---" % (time.time() - start_time))
            
def process_videos_prot_localization(path2excels, path2stacks, OUTPUTPATH, mitotic = False, min_len=20, pixel_size=0.802, min_track=0):
    """
    Full videos were not postprocessed by any closing or hole-filling operation
    
    STORE-SMOOTH: 
    will just store the smooth along the time so we can see why 
    there might be some errors. 
    
    Small objects will not be removed during the detection as those can be the 
    result of the smoothing and might help to track vibrating cells.
    """
    # This function is specific to the experiments conducted and their subfolders
    folders = os.listdir(path2excels)
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH) 
    print(folders)
    for f in folders:
        
        print(f)
        if not f.__contains__('.xlsx'):
            process_videos_prot_localization(os.path.join(path2excels, f),
                           os.path.join(path2stacks, f),
                           os.path.join(OUTPUTPATH, f), mitotic = False,
                           min_len=min_len, pixel_size=pixel_size)
        else:
            print('Processing {}'.format(path2stacks))            
            t0 = time.time()
            create_protrusions_localization(os.path.join(path2excels, f),
                                            path2stacks, 
                                            OUTPUTPATH, mitotic = False,
                                            pixel_size = pixel_size, 
                                            min_len = min_len, 
                                            min_track=min_track)
            t1 = time.time() - t0
            print('{} already processed.'.format(path2stacks))
            print("Time elapsed: ", t1) # CPU seconds elapsed (floating point)


def prepare_test_localizations(EXCELS, path2stacks, PATH2VIDEOS, OUTPUTPATH, min_len=20, pixel_size=0.802):

    files = [x for x in open(PATH2VIDEOS, "r")]
    files = files[1:]  # First row contains labels
    files = [x.split(';')[0] for x in files]
    files = np.unique(files) 
    
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
        
    for file_name in files:
        # Calculate how many frames you need to process (it is said in the name of the video)
        start_time, end_time = file_name.split('_')[-1].split('-')
        
        directories = file_name.split("_")
        if directories[0].__contains__("Video"):
            inner_dir = os.path.join("V", directories[1], directories[2])
            exel_name = directories[2] + ".xlsx"
            video_name = file_name.split(directories[2]+"_")[-1]
            video_name = video_name.split("_" + start_time + "-" + end_time)[0]
        else:
            inner_dir = os.path.join(directories[0], directories[1])
            exel_name = directories[1] + ".xlsx"
            video_name = file_name.split(directories[1]+"_")[-1]
            video_name = video_name.split("_" + start_time + "-" + end_time)[0]
        start_time = np.int(start_time) - 1 # Python counter starts at 0
        end_time = np.int(end_time) - 1 # Python counter starts at 0  
        sys.stdout.write("\rProcessing video {0}:\n".format(file_name))
        
        path2excels = os.path.join(EXCELS, inner_dir, exel_name)
        df = pd.read_excel(path2excels, video_name + '_stackreg')  
    
        # df = df[df['Time'] <= end_time+1]    
        # df = df[df['Time'] >= start_time+1]  
        # Discard short tracks
        labels = np.unique(np.asarray(df['Cell']))    
        # Analyse videos and print protrusions frame by frame
        track_video = sitk.ReadImage(os.path.join(path2stacks,inner_dir, video_name + '_stackreg.tif'))
        track_video = sitk.GetArrayFromImage(track_video)
        track_video = track_video[start_time:end_time]
                    
        for t in range(len(track_video)):
            print(t)
            frame = track_video[t]
            frame_tips = np.zeros_like(frame)
            if len(labels) > 0:                
                frame_labels = np.unique(frame)
                frame_labels = frame_labels[1:]                
                frame_labels = [l for l in frame_labels if l in labels]                        
                for l in frame_labels:                    
                    # We need to remove the mitosis to avoid problems with geodesic distance
                    print('Cell with ID {}'.format(l))
                    cell = (frame==l).astype(np.uint8)                
                    idx, concomp = cv2.connectedComponents(cell)
                    if idx>2:
                        cell = connect_connectedcomponents(concomp, thickness = 2)
                    C = centroidCalculator(cell)
                    frame_tips = print_protrusions_localization(frame_tips, cell, C, pixel_size, min_len)
            track_video[t] = frame_tips
            #Store the video
            sitk.WriteImage(sitk.GetImageFromArray(track_video), os.path.join(OUTPUTPATH, video_name +  '_stackreg_' + file_name.split('_')[-1] + '.tif'))
    
    

# plt.figure(figsize=(10,10))
# plt.imshow(points[125])
# plt.show()
#
# for t in range(1, im.shape[0]-1):
#     filtered_video.append(np.mean(im[t-k:t+k+1], axis=0))
# filtered_video.append(im[-1])
# filtered_video = np.array(filtered_video)
#
# plt.figure(figsize=(2,10))
# for t in range(filtered_video.shape[0]):
#
#     plt.subplot(5,2,2*t+1)
#     plt.imshow(im[t])
#     plt.colorbar()
#     plt.title('original')
#     plt.subplot(5,2,2*(t+1))
#     plt.imshow(filtered_video[t])
#     plt.colorbar()
#     plt.title('filtered')
# plt.show()


# plt.figure(figsize=(15,10))
# for t in range(10):

#     plt.subplot(10,3,3*t+1)
#     plt.imshow(binary_video[156+t, 500:750, 480:])
#     plt.colorbar()
#     plt.title('original')
#     plt.subplot(10,3,3*t+2)
#     plt.imshow(S[156+t, 500:750, 480:])
#     plt.colorbar()
#     plt.title('Smooth')
#     plt.subplot(10,3,3*(t+1))
#     plt.imshow((S[156+t, 500:750, 480:])>50)
#     # plt.colorbar()
#     plt.title('mean')
# plt.show()
