"""
Created on Mon Oct 5 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import numpy as np


def index_from_pdf(pdf_im):
    prob = np.copy(pdf_im)
    # Normalize values to create a pdf with sum = 1
    prob = prob.ravel() / np.sum(prob)
    # Convert into a 1D pdf
    choices = np.prod(pdf_im.shape)
    index = np.random.choice(choices, size=1, p=prob)
    # Recover 2D shape
    coordinates = np.unravel_index(index, shape=pdf_im.shape)
    # Extract index
    indexh = coordinates[0][0]
    indexw = coordinates[1][0]
    return indexh, indexw


def sampling_pdf(y, pdf, height, width):
    h, w = y.shape[0], y.shape[1]
    if pdf == 1:
        indexw = np.random.randint(np.floor(width // 2), \
                                   w - np.floor(width // 2))
        indexh = np.random.randint(np.floor(height // 2), \
                                   h - np.floor(height // 2))
    else:
        # Assign pdf values to foreground
        pdf_im = np.ones(y.shape, dtype=np.float32)
        pdf_im[y > 0] = pdf
        # crop to fix patch size
        pdf_im = pdf_im[np.int(np.floor(height // 2)):-np.int(np.floor(height // 2)), \
                 np.int(np.floor(width // 2)):-np.int(np.floor(width // 2))]
        indexh, indexw = index_from_pdf(pdf_im)
        indexw = indexw + np.int(np.floor(width // 2))
        indexh = indexh + np.int(np.floor(height // 2))

    return indexh, indexw


def random_crop(x, y, crop_size, pdf=1, sync_seed=None):
    """
    x is 3D data: [height,width, z] or [height,width, t]
    y is assumed to be 2D: [height, width]
    pdf is a value to weight foreground in y and define a sampling distribution to crop the patches
    """
    if x.shape[0] < crop_size[0] or x.shape[1] < crop_size[1]:
        raise Exception("Input image is smaller than the input size. Check the input size of the network.")
    elif x.shape[0] == crop_size[0] or x.shape[1] == crop_size[1]:
        return x[:crop_size[0], :crop_size[1]], y[:crop_size[0], :crop_size[1]]
    else:
        np.random.seed(sync_seed)
        offseth, offsetw = sampling_pdf(y, pdf, crop_size[0], crop_size[1])

        lr = offseth - np.floor(crop_size[0] // 2)
        lr = lr.astype(np.int)
        ur = offseth + np.round(crop_size[0] // 2)
        ur = ur.astype(np.int)

        lc = offsetw - np.floor(crop_size[1] // 2)
        lc = lc.astype(np.int)
        uc = offsetw + np.round(crop_size[1] // 2)
        uc = uc.astype(np.int)

        x = x[lr:ur, lc:uc]
        y = y[lr:ur, lc:uc]
        # if random_crop_size_input != random_crop_size_output:
        #     y_patch = center_crop(y_patch, random_crop_size_output)
        return x, y


def random_crop_complex(x, y
                        , y_marks
                        , weights
                        , random_crop_size_input
                        , random_crop_size_output
                        , pdf=1, sync_seed=None):
    """
    x is 2D data or 3D for channels or time dimenstion. Please let the 3rd dimension that does not affect the crop, in axis=-1
    pdf is a 2D image representing the sampling distribution to crop the patches
    """
    if x.shape[0] < random_crop_size_input[0] or x.shape[1] < random_crop_size_input[1]:
        raise Exception("Input image is smaller than the input size. Check the input size of the network.")
    elif x.shape[0] == random_crop_size_input[0] or x.shape[1] == random_crop_size_input[1]:
        if random_crop_size_input != random_crop_size_output:
            y = center_crop(y[:random_crop_size_input[0], :random_crop_size_input[1]], random_crop_size_output)
            y_marks = center_crop(y_marks[:random_crop_size_input[0], :random_crop_size_input[1]], random_crop_size_output)
            weights = center_crop(weights[:random_crop_size_input[0], :random_crop_size_input[1]], random_crop_size_output)
        return x[:random_crop_size_input[0], :random_crop_size_input[1]], y, y_marks, weights
    else:
        np.random.seed(sync_seed)
        offseth, offsetw = sampling_pdf(y, pdf,
                                        random_crop_size_input[0],
                                        random_crop_size_input[1])

        lr = offseth - np.floor(random_crop_size_input[0] // 2)
        lr = lr.astype(np.int)
        ur = offseth + np.round(random_crop_size_input[0] // 2)
        ur = ur.astype(np.int)

        lc = offsetw - np.floor(random_crop_size_input[1] // 2)
        lc = lc.astype(np.int)
        uc = offsetw + np.round(random_crop_size_input[1] // 2)
        uc = uc.astype(np.int)

        x = x[lr:ur, lc:uc]
        y = y[lr:ur, lc:uc]
        y_marks = y_marks[lr:ur, lc:uc]
        weights = weights[lr:ur, lc:uc]

        if random_crop_size_input != random_crop_size_output:
            y = center_crop(y, random_crop_size_output)
            y_marks = center_crop(y_marks, random_crop_size_output)
            weights = center_crop(weights, random_crop_size_output)

        return x, y, y_marks, weights


def center_crop(x, crop_size):
    """
    Crop function when there is no padding in the the CNN and we need to reduce the ground truth. 
    x is 3D data but patches are only cropped in 2D. x dimensions (height, width, depth)
    """
    halfh = (x.shape[0] - crop_size[0]) // 2
    halfw = (x.shape[1] - crop_size[1]) // 2

    return x[halfh:-halfh, halfw:-halfw]
