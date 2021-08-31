"""
Created on Wed March 31 2021

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import numpy as np
import SimpleITK as sitk
import cv2
import plantcv
from plantcv import plantcv as pcv
from scipy.ndimage import distance_transform_edt
import pandas as pd
from numba import njit

def connect_connectedcomponents(concomp, thickness = 2):    
    im = (concomp>0).astype(np.uint8)
    labels = np.unique(concomp)
    labels = labels[1:]
    points = []
    for l in labels:
        C = centroidCalculator((concomp==l).astype(np.uint8))
        # Points must be given as (x,y) coordinates
        points.append((C[1],C[0]))
    for i in range(len(points)-1):
        cv2.line(im, points[i], points[i+1], (1), thickness)        
    return im
    

def geodesic_distance_transform(m):
    mask = m.mask
    visit_mask = mask.copy() # mask visited cells
    m = m.filled(np.inf)
    m[m!=0] = np.inf
    distance_increments = np.asarray([np.sqrt(2), 1., np.sqrt(2), 1., 1., np.sqrt(2), 1., np.sqrt(2)])
    connectivity = [(i,j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (not (i == j == 0))]
    cc = np.unravel_index(m.argmin(), m.shape) # current_cell
    while (~visit_mask).sum() > 0:
        neighbors = [tuple(e) for e in np.asarray(cc) - connectivity if not visit_mask[tuple(e)]]
        tentative_distance = [distance_increments[i] for i,e in enumerate(np.asarray(cc) - connectivity) 
                              if not visit_mask[tuple(e)]]
        for i,e in enumerate(neighbors):
            d = tentative_distance[i] + m[cc]
            if d < m[e]:
                m[e] = d
        visit_mask[cc] = True
        m_mask = np.ma.masked_array(m, visit_mask)
        cc = np.unravel_index(m_mask.argmin(), m.shape)
    return m

def roundnessCalculator(object_matrix, projected=False):
    """
    This method provides the roundness as the projected roundness of the 
    element with label object_lab. 
    The object matrix is binary mask with the object object of interest:
    foreground = 1 and background = 0.
    """
    # Discriminate the element of interest and get a binary image
    element = object_matrix.astype(np.uint8)
    # Smooth the segmentation without changing its area and closing any possible hole.
    # element = cv2.dilate(element, np.ones((3,3), np.uint8))
    # element = cv2.erode(element, np.ones((3,3), np.uint8))
    
    if projected == True:
        # In the automtic segmentation there are some disconnected parts in 
        # the cell body. The closing shouldn't affect the final result.         
        A = np.sum(element)
        if A > 0:
            _, contours, _ = cv2.findContours(element, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = cv2.drawContours(np.zeros(element.shape),contours, -1, color=[1, 0, 0], thickness=1)
            P = np.sum(contours)
            projected_roundness = (P**2) / (4*np.pi*A)
        else:
            projected_roundness = 0
        roundness = np.min((projected_roundness,1))
    else:
        element = sitk.GetImageFromArray(element)
        # Define the shape statistic filter and run it on the element
        cell_label = sitk.LabelShapeStatisticsImageFilter()
        cell_label.Execute(element)
        # Obtain the roundness as the ratio between the major and minor axis:
        M = np.max([cell_label.GetPrincipalMoments(1)[0], 
                    cell_label.GetPrincipalMoments(1)[1]])
        m = np.min([cell_label.GetPrincipalMoments(1)[0], 
                    cell_label.GetPrincipalMoments(1)[1]])
        roundness = m/M
        
    #    # Obtain the roundness given by SimpleITK
    #    roundness = cell_label.GetRoundness(1)    
    return roundness


def centroidCalculator(cell):
    """
    Centroid is assumed to be roundish so the euclidean distance will always 
    be larger at that region than along protrusions. 
    """
    edt = distance_transform_edt(cell)
    centroid = np.where(edt==np.max(edt))
    centroid = [centroid[0][0], centroid[1][0]]
    return centroid
   
def protrusionsCalculator(cell, centroid, pixel_size=0.802, min_len=20):
    """
    min len should account for the cell body (~10 microns of radious) + protrusions (5 microns of length)
    15 microns which in pixels translates to 18.70 ~ 20 pixels.
    min_len should be given in microns
    
    """    
    skeleton = pcv.morphology.skeletonize(mask=cell)
    
    tip_pts_mask = pcv.morphology.find_tips(skel_img=skeleton, mask=None, label="default")  
    tip_pts_mask = tip_pts_mask.astype(bool)
    tips_len = []
    tips_diam = []
    
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

    # Obtain the segments in the skeleton (each segment between a branch + tip, or between tips)
    pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, mask=cell)
    seg_img = np.dot(seg_img[...,:3], [0.2989, 0.5870, 0.1140])
    seg_img = np.clip(seg_img, 0, 255)
    seg_img = seg_img.astype(np.uint8)    
    seg_img = np.multiply(seg_img, skeleton>0)   
    
    # The euclidean distance transform approximates the diameter when taken in the skeleton.
    edt = distance_transform_edt(cell)
    edt = pixel_size*edt
    edt = np.multiply(edt, skeleton>0)
    
    # Calculate the segments overlaping each tip
    segm2tips = np.multiply(seg_img, tip_pts_mask)
    segm2tips, counts = np.unique(segm2tips, return_counts=True)
    segm2tips = segm2tips[1:]
    counts = counts[1:] 
    tips_dist = np.where(tip_pts_mask == True)    
    for tip_y, tip_x in zip(tips_dist[0], tips_dist[1]):
        if gdt[tip_y, tip_x]<min_len:
            tip_pts_mask[tip_y, tip_x] = False
        else:
            tips_len.append(gdt[tip_y, tip_x])
            tip_label = seg_img[tip_y, tip_x]
            if counts[segm2tips == tip_label] == 2:
                aux_skel = (seg_img==tip_label).astype(np.uint8)
                # We do a hole of 10 pixels to avoid including too much info from the cell body
                cv2.circle(aux_skel,tuple((centroid[1],centroid[0])),10,(0), -1)
                _, aux_skel = cv2.connectedComponents(aux_skel)
                tips_diam.append(2*np.quantile(edt[aux_skel==aux_skel[tip_y, tip_x]], 0.5))  
            else:
                tips_diam.append(2*np.quantile(edt[seg_img==tip_label], 0.5))  
            
    tip_num = np.sum(tip_pts_mask.astype(np.uint8))

    if tip_num>0:    
        # tips_diam, branch_diam = diameterCalculator(cell, skeleton, tip_pts_mask, tip_labels, pixel_size)
        branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=skeleton, mask=cell, label="default")
        B = np.where(branch_pts_mask>0)
        branch_diam = [2*pixel_size*d for d in edt[B]]
        branch_num = len(branch_diam)  
        
    else:
        tips_diam = []
        branch_diam = []
        branch_num = 0
            
    return tip_num, tips_len, tips_diam, branch_num, branch_diam

def protrusion_instances_morphology(cell, tip_labels, tips_coords, centroid, pixel_size=0.802):   
    """
    cell: binary mask containing a single connected component (i.e. a cell)
    tips_coords: list of tip candidate coordinates given as (rows, cols ) or (y, x)
    centroid: cell centroid position given as (rows, cols ) or (y, x) to calculate the geodesic distance transform
    pixel_size: in microns for example, to provide real measures.
    """
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
    # Obtain the segments in the skeleton (each segment between a branch + tip, or between tips)
    pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, mask=cell)
    seg_img = np.dot(seg_img[...,:3], [0.2989, 0.5870, 0.1140])
    seg_img = np.clip(seg_img, 0, 255)
    seg_img = seg_img.astype(np.uint8)    
    seg_img = np.multiply(seg_img, skeleton>0) 
    # The euclidean distance transform approximates the diameter when taken in the skeleton.
    edt = distance_transform_edt(cell)
    edt = pixel_size*edt
    edt = np.multiply(edt, skeleton>0)
    # Calculate the segments overlaping each tip
    segm2tips = np.multiply(seg_img, tip_pts_mask)
    segm2tips, counts = np.unique(segm2tips, return_counts=True)
    segm2tips = segm2tips[1:]
    counts = counts[1:]    
    lengths = []
    diam = []
    X_position = []
    Y_position = []    
    tips_ID = []
    for tip in range(len(tips_coords)): 
        tip_y = tips_coords[tip][0]
        tip_x = tips_coords[tip][1]
        if gdt[tip_y, tip_x] == np.inf:
            print(tip_y, tip_x)
            nn = 1
            candidates = (np.array([tip_y - nn, tip_y, tip_y + nn,
                                    tip_y - nn, tip_y, tip_y + nn, 
                                   tip_y - nn, tip_y, tip_y + nn]),
                         np.array([tip_x - nn, tip_x - nn, tip_x - nn, 
                                   tip_x, tip_x, tip_x, 
                                   tip_x + nn, tip_x + nn, tip_x + nn]))
            lengths.append(np.min(gdt[candidates]))
            tip_label = np.max(seg_img[candidates])
            print(np.min(gdt[candidates]))
            print(tip_label)
        else:    
            lengths.append(gdt[tip_y, tip_x])
            tip_label = seg_img[tip_y, tip_x]
        if counts[segm2tips == tip_label] == 2:
            aux_skel = (seg_img==tip_label).astype(np.uint8)
            # We do a hole of 10 pixels to avoid including too much info from the cell body
            cv2.circle(aux_skel,tuple((centroid[1],centroid[0])),10,(0), -1)
            _, aux_skel = cv2.connectedComponents(aux_skel)
            
            diam.append(2*np.quantile(edt[aux_skel==aux_skel[tip_y, tip_x]], 0.5))
        else:
            
            diam.append(2*np.quantile(edt[seg_img==tip_label], 0.5))  
        
        X_position.append(tip_x)
        Y_position.append(tip_y)
        tips_ID.append(tip_labels[tip])
    tips_info = pd.DataFrame()
    tips_info['Prot'] = tips_ID
    tips_info['X_position'] = X_position
    tips_info['Y_position'] = Y_position    
    tips_info['Length'] = lengths
    tips_info['Diameter'] = diam
    
    return tips_info


# for l in labels:
#     cell = (frame==l).astype(np.uint8)
#     cell = (cell==2).astype(np.uint8)
#     if cell.shape[0] != cell.shape[1]:
#         aux = np.zeros((np.max(cell.shape), np.max(cell.shape)), dtype=np.uint8)
#         aux[:cell.shape[0], :cell.shape[1]] = cell
#         cell = aux
#     edt = distance_transform_edt(cell)
#     # Centroid is assumed to be roundish so the euclidean distance will always 
#     #be larger at that region than along protrusions.
    
#     centroid = np.where(edt==np.max(edt))
#     centroid = [centroid[0][0], centroid[1][0]]
    
#     skeleton = pcv.morphology.skeletonize(mask=cell)
#     branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=skeleton, mask=cell, label="default")
#     tip_pts_mask = pcv.morphology.find_tips(skel_img=skeleton, mask=None, label="default")  
#     tip_pts_mask = tip_pts_mask.astype(bool)
    
#     mask = ~cell.astype(bool)
#     m = np.ma.masked_array(cell.astype(np.float32), mask)
#     m[centroid[0], centroid[1]] = 0
#     gdt = geodesic_distance_transform(m)
#     tips_dist = np.where(tip_pts_mask == True)
#     for tip_y, tip_x in zip(tips_dist[0], tips_dist[1]):
#         if gdt[tip_y, tip_x]<15:
#             tip_pts_mask[tip_y, tip_x] = False
    
#     # Label each protrusion uniquely, taking branches into account
#     pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, mask=cell)
#     seg_img = np.dot(seg_img[...,:3], [0.2989, 0.5870, 0.1140])
#     seg_img = np.clip(seg_img, 0, 255)
#     seg_img = seg_img.astype(np.uint8)
    
#     tips_dist = np.where(tip_pts_mask == True)    
#     for tip_y, tip_x in zip(tips_dist[0], tips_dist[1]):
#         tip_label = seg_img[tip_y, tip_x]        
#         diameter = np.mean(edt[seg_img == tip_label])
    

    
#     plt.figure(figsize=(25,25))
#     plt.imshow(pruned)
#     plt.show()

    
