"""
Created on Tue Dec 22 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""

import SimpleITK as sitk
import numpy as np

def remove_objects_from_boundary(img):
    """
    The function remove all the binary segmentations touching the edges of the image
    img is a binary image of type uint8.
    """
    # copy the image and set to 0 all the pixels touching the edhges
    image = np.copy(img)
    image[1:-1,1:-1] = 0
    
    # Create sitk objects and run the reconstruction so we get the objects in the edges
    orImage = sitk.GetImageFromArray(img)
    sitkImage = sitk.GetImageFromArray(image)
    recons_filter = sitk.BinaryReconstructionByDilationImageFilter()
    sitkImage = recons_filter.Execute(sitkImage,orImage)
    
    # remove detected objects
    sitkImage = sitk.GetArrayFromImage(sitkImage)
    final = img - sitkImage
    return final

def fill_holes(img):
    
    fh = sitk.BinaryFillholeImageFilter()
    th_image = fh.Execute(sitk.GetImageFromArray(img))
    th_image = sitk.GetArrayFromImage(th_image)
    return th_image

def closing(img, radious=3):
    # Morphological clossing of the binary image
    fh = sitk.BinaryMorphologicalClosingImageFilter()
    fh.SetKernelRadius(radious)
    th_image = fh.Execute(sitk.GetImageFromArray(img))
    th_image = sitk.GetArrayFromImage(th_image)
    return th_image

def remove_small_objects(img, min_size=150):

    # Get a labeled image.
    th_image = sitk.GetImageFromArray(img)
    th_image_ccomp = sitk.ConnectedComponent(th_image)
    # Compute the label statistics for the size of the cell
    labels_stats = sitk.LabelShapeStatisticsImageFilter()
    labels_stats.Execute(th_image_ccomp)
    label_image = sitk.GetArrayFromImage(th_image_ccomp)       
    # Filter out all the cells smaller than min_size
    for l in labels_stats.GetLabels():

        # Get the size (area) in pixels of the object
        a = labels_stats.GetNumberOfPixels(l)

        # Cell body filtering
        if a < min_size:
            label_image[label_image == l] = 0

    # Get the binary image again        
    label_image[label_image>=1] = 1             
    return label_image

def post_processing(img, min_size=150, remove_objects_boundary=True):
    img = img.astype(np.uint8)
    I = closing(img)
    I = fill_holes(I)
    I = remove_small_objects(I, min_size=min_size)
    if remove_objects_boundary:
        I = remove_objects_from_boundary(I)
    
    return I.astype(np.uint8)


    