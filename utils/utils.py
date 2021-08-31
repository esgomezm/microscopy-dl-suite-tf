"""
Created on Sat March 21 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import h5py
import numpy as np
import glob
import SimpleITK as sitk
import os
from skimage import measure
import skimage.morphology
import scipy.ndimage
import cv2
# import time
# import imreg_dft as ird

def load_mat_files(path):
    f = h5py.File(path, 'r')
    for k, v in f.items():
        matlab_matrix = np.array(v)
    if len(matlab_matrix.shape) == 3:
        matlab_matrix = np.transpose(matlab_matrix, [0, 2, 1])
    else:
        matlab_matrix = np.transpose(matlab_matrix, [1, 0])
    return matlab_matrix


def stack2im(PATH, END='_Segmentationim-label', keypoints=False, COUNT=1):
    # path to the data directory which contains folders such as train, val or test
    # PATH = "./data/"
    # subfolder in which the input and ground truth are
    # mode = 'train/'

    # Old data
    # name of the folder containing the input videos
    PATH2DATA = PATH + "inputs/"
    # name of the folder containing ground truth
    PATH2GT = PATH + 'labels/'

    # Ground truth's name has a different ending. Write it so as to get the correct
    # name of the input file
    # END = "_Segmentation2im_Prot"
    # END = '_Segmentationim-label'

    # New data
    # directory in which new files should be saved
    PATH2OUTPUT = PATH + 'stack2im/'
    # make a new directory called set
    if not os.path.exists(PATH2OUTPUT+'inputs'):
        os.makedirs(PATH2OUTPUT+'inputs')
    if not os.path.exists(PATH2OUTPUT+'labels'):
        os.makedirs(PATH2OUTPUT+'labels')
    if keypoints:
        PATH2KEY = PATH + 'keypoints/'
        if not os.path.exists(PATH2OUTPUT+'keypoints'):
            os.makedirs(PATH2OUTPUT+'keypoints')

    # name of the new ground truth
    masks_names = 'instance_ids_'
    # name of the new input slice
    input_names = 'raw_'

    FILES = glob.glob(PATH2GT + '/*.tif')
    # COUNT = 1
    fields = "Video; Frame"
    with open(os.path.join(PATH2OUTPUT, 'videos2im_relation.csv'), 'w') as file_:
        file_.write(fields)
        file_.write("\n")

    for file in FILES:
        # Load masks from segmented videos
        file_name = file
        masks = sitk.ReadImage(file_name)
        masks = sitk.GetArrayFromImage(masks)
        # if keypoints:
        #     keypoints_name = file_name.split('/')[-1]
        #     keypoints_name = keypoints_name.split('Segmentationim-label.tif')[0]+'keypoints.tif'
        #     print(PATH2KEY + keypoints_name)
        #     tips_loc = sitk.ReadImage(PATH2KEY + keypoints_name)
        #     tips_loc = sitk.GetArrayFromImage(tips_loc)

        # Load original images corresponding to masks.
        file_name = file_name.split('/')[-1].split('.')[0]
        # Option 1 for the input names
        if END is None:
            image_name = file_name + '.tif'
        else:
            time_range = file_name[:-len(END)].split('_')[-1]
            image_name = file_name[:-(len(END + time_range))] + 'stackreg_' + time_range + '.tif'
            # Option 2 for the input names
            # image_name = file_name[:-(len(END))] + '.tif'
        image = sitk.ReadImage(PATH2DATA + image_name)
        image = sitk.GetArrayFromImage(image)
        if keypoints:
            print(PATH2KEY + image_name)
            tips_loc = sitk.ReadImage(PATH2KEY + image_name)
            tips_loc = sitk.GetArrayFromImage(tips_loc)

        for i in range(masks.shape[0]):
            aux = masks[i]
            aux = aux.astype(np.uint8)
            sitk.WriteImage(sitk.GetImageFromArray(aux),
                            PATH2OUTPUT + 'labels/' + masks_names + '{0:0>3}'.format(int(COUNT)) + '.tif')
            print('File {} number {} processed.'.format(masks_names, COUNT))

            if keypoints:
                aux = tips_loc[i]
                aux = aux.astype(np.uint8)
                sitk.WriteImage(sitk.GetImageFromArray(aux),
                                PATH2OUTPUT + 'keypoints/' + masks_names + '{0:0>3}'.format(int(COUNT)) + '.tif')
                print('File {} number {} processed.'.format(masks_names, COUNT))

            aux = image[i]
            aux = aux.astype(np.uint16)
            sitk.WriteImage(sitk.GetImageFromArray(aux),
                            PATH2OUTPUT + 'inputs/' + input_names + '{0:0>3}'.format(int(COUNT)) + '.tif')

            with open(os.path.join(PATH2OUTPUT, 'videos2im_relation.csv'), 'a') as file_:
                if END is None:
                    file_.write('{};'.format(file_name) + input_names + '{0:0>3}.tif'.format(int(COUNT)))
                else:
                    file_.write('{};'.format(file_name[:-len(END)]) + input_names + '{0:0>3}.tif'.format(int(COUNT)))
                file_.write("\n")
            print('File {} number {} processed.'.format(input_names, COUNT))

            COUNT = COUNT + 1
        print('File {} converted.'.format(file_name))
    print('Process finished')

## Semmantic-segmentation suite
def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        if len(label.shape) == 3:
            class_map = np.all(equality, axis=-1)
        else:
            class_map = equality.astype(np.int8)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map


def percentile_normalization(image, low=0.1, upper=99.1, clip=True):
    mi = np.percentile(image, low)
    ma = np.percentile(image, upper)
    image_ = (image - mi) / (ma - mi + 1e-50)
    if clip:
        image_ = np.clip(image_, 0, 1)
    return image_

def median_normalization(image):
    '''
    Normalize the values in the image to have median 0.5
    '''
    # image_ = image / 255 + (.5 - np.median(image / 255)) # Original from MU-Lux_CZ
    # The function assumes that values are in the [0,1] range
    image_ = image + (.5 - np.median(image))
    return np.maximum(np.minimum(image_, 1.), .0)

def hist_equalization(image):
    """
    Contrast limited adaptive histogram equalization for normalization fo values
    """
    # return cv2.equalizeHist(image) / 255 # Original from MU-Lux_CZ
    # The function assumes that values are in the [0,1] range
    return cv2.equalizeHist(image)

def mean_normalization(image):
    '''
    Normalize the values in the image to have mean 0.5
    '''
    # The function assumes that values are in the [0,1] range
    image_ = image + (.5 - np.mean(image))
    return np.maximum(np.minimum(image_, 1.), .0)

def linear_stretch(im, inf_lim, sup_lim):
    '''
    Transforms the dynamic range of values into the [inf_lim, sup_lim] range.
    '''
    new_im = np.copy(im)
    new_im = new_im.astype(np.float32)
    inf_im = np.min(new_im)
    sup_im = np.max(new_im)

    coef = (sup_lim - inf_lim) / (sup_im - inf_im)
    new_im = coef * (new_im - inf_im) + inf_lim

    return new_im

def get_normal_fce(normalization):
    '''
    selects the corresponding normalizing function
    '''
    if normalization == 'HE':
        return hist_equalization
    elif normalization == 'MEDIAN':
        return median_normalization
    elif normalization == 'MEAN':
        return mean_normalization
    elif normalization == 'PERCENTILE':
        return percentile_normalization
    else:
        print('normalization function was not picked')
        return None

def remove_uneven_illumination(img, blur_kernel_size=501, data_type='uint16'):
    '''
    uses LPF to remove uneven illumination
    '''

    img_f = img.astype(np.float32)
    img_mean = np.mean(img_f)

    img_blur = cv2.GaussianBlur(img_f, (blur_kernel_size, blur_kernel_size), 0)
    if data_type == 'uint16':
        result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 65535), 0).astype(np.int32)
        return result
    elif data_type == 'uint8':
        result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 255), 0).astype(np.int32)
        return result
    else:
        print('Unknown datatype {}'.format(data_type))
        return None

def remove_edge_cells(label_img, border=20):
    edge_indexes = get_edge_indexes(label_img, border=border)
    return remove_indexed_cells(label_img, edge_indexes)

def get_edge_indexes(label_img, border=20):
    mask = np.ones(label_img.shape)
    mi, ni = mask.shape
    mask[border:mi - border, border:ni - border] = 0
    border_cells = mask * label_img
    indexes = (np.unique(border_cells))

    result = []

    # get only cells with center inside the mask
    for index in indexes:
        cell_size = sum(sum(label_img == index))
        gap_size = sum(sum(border_cells == index))
        if cell_size * 0.5 < gap_size:
            result.append(index)

    return result

def remove_indexed_cells(label_img, indexes):
    mask = np.ones(label_img.shape)
    for i in indexes:
        mask -= (label_img == i)
    return label_img * mask

def get_image_size(path):
    '''
    returns size of the given image
    '''
    names = os.listdir(path)
    name = names[0]
    o = cv2.imread(os.path.join(path, name), cv2.IMREAD_ANYDEPTH)
    return o.shape[0:2]

def get_new_value(mi, divisor=16):
    if mi % divisor == 0:
        return mi
    else:
        return mi + (divisor - mi % divisor)

def read_image(path):
    # Original from MU-Lux_CZ
    # if 'Fluo' in path:
    #     img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    #     if 'Fluo-N2DL-HeLa' in path:
    #         img = (img / 255).astype(np.uint8)
    #     if 'Fluo-N2DH-SIM+' in path:
    #         img = np.minimum(img, 255).astype(np.uint8)
    # else:

    # all the images are phase contrast of int16
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    return img
    
def read_instances(path, radious=5):
    '''
    Reads the instance segmentations and returns a binary mask and the marks for the detections.
    '''
    labels = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    mi = labels.shape[0]
    ni = labels.shape[1]

    if np.sum(labels) > 0:
        # Obtain the instances
        components = np.unique(labels)
        n_comp = len(components) - 1
        # Create an array with as many channels as instances
        detection_marks = np.zeros((n_comp, mi, ni))
        for c in range(n_comp):
            # Keep current object.
            tmp = (labels == components[c + 1])
            # Transform the label with erosion
            diskelem = skimage.morphology.disk(radious)
            detection_marks[c] = skimage.morphology.binary_erosion(tmp, diskelem)
        # Instances do not overlap, specially after erossion, so we merge all of them with sum (projection)
        detection_marks = np.sum(detection_marks, axis=0)
        detection_marks = detection_marks.reshape((mi, ni))
    else:
        detection_marks = np.zeros((mi, ni))
    labels[labels > 0] = 1
    return labels, detection_marks

def read_input_image(path, normalization='MEAN', uneven_illumination=False):
  normalization_fce = get_normal_fce(normalization)
  im = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
  
  # mi = im.shape[0]
  # ni = im.shape[1]
  if im is None:
      print('image {} was not loaded'.format(path))

  if uneven_illumination:
      # o = np.minimum(o, 255).astype(np.uint8) # Original from MU-Lux_CZ
      im = remove_uneven_illumination(im, data_type=im.dtype.name)

  # convert values to the [0,1] range
  if im.dtype.name == 'uint16':
    im = im.astype(np.float32)
    im = im / 65535
  else:
    im = im.astype(np.float32)
    im = im / 255
  
  im = normalization_fce(im)

  return im


def read_input_videos(video_file, normalization='MEAN'):
    normalization_fce = get_normal_fce(normalization)
    im = sitk.ReadImage(video_file)
    im = sitk.GetArrayFromImage(im)

    # mi = im.shape[0]
    # ni = im.shape[1]
    if im is None:
        print('video {} was not loaded'.format(video_file))

    # convert values to the [0,1] range
    if im.dtype.name == 'uint16':
        im = im.astype(np.float32)
        im = im / 65535
    else:
        im = im.astype(np.float32)
        im = im / 255

    im = normalization_fce(im)

    return im
# read images
# def load_input_images(path, cut=False, new_mi=0, new_ni=0, normalization='MEAN', uneven_illumination=False):
#     names = os.listdir(path)
#     names.sort()

#     mi, ni = get_image_size(path)

#     dm = (mi % 16) // 2
#     mi16 = mi - mi % 16
#     dn = (ni % 16) // 2
#     ni16 = ni - ni % 16

#     total = len(names)
#     normalization_fce = get_normal_fce(normalization)

#     image = np.empty((total, mi, ni, 1), dtype=np.float32)

#     for i, name in enumerate(names):

#         o = read_image(os.path.join(path, name))

#         if o is None:
#             print('image {} was not loaded'.format(name))

#         if uneven_illumination:
#             # o = np.minimum(o, 255).astype(np.uint8) # Original from MU-Lux_CZ
#             o = remove_uneven_illumination(o, data_type=o.dtype.name)

#         # convert values to the [0,1] range
#         if o.dtype.name == 'uint16':
#             o = o / 65535
#         else:
#             o = o / 255
#         image_ = normalization_fce(o)

#         image_ = image_.reshape((1, mi, ni, 1))
#         image[i, :, :, :] = image_

#         print('Loaded image {} from directory {}'.format(name,path))

#     if cut:
#         image = image[:, dm:mi16 + dm, dn:ni16 + dn, :]
#     if new_ni > 0 and new_ni > 0:
#         image2 = np.zeros((total, new_mi, new_ni, 1), dtype=np.float32)
#         image2[:, :mi, :ni, :] = image
#         image = image2

#     print('loaded images from directory {} to shape {}'.format(path, image.shape))
#     return image

# def load_images(path):
#     '''
#     loads all the 2D images in "path" and outputs a vector of 4 dimensions: [number of images, height, width, 1]
#     '''
#     names = os.listdir(path)
#     names.sort()
#     mi, ni = get_image_size(path)
#     total = len(names)
#     image = np.empty((total, mi, ni, 1), dtype=np.float32)

#     for i, name in enumerate(names):
#         o = read_image(os.path.join(path, name))
#         o = o.reshape((1, mi, ni, 1))
#         image[i, :, :, :] = o
#         print('Loaded image {} from directory {}'.format(name, path))
#     print('loaded images from directory {} to shape {}'.format(path, image.shape))
#     return image

# def load_instances(path, radious=3):
#     '''
#     Reads the instance segmentations and returns a binary mask and the marks for the detections.
#     '''
#     names = os.listdir(path)
#     names.sort()
#     mi, ni = get_image_size(path)
#     total = len(names)
#     image = np.empty((total, mi, ni, 4), dtype=np.uint8)
#     for i, name in enumerate(names):
#         o = read_image(os.path.join(path, name))
#         if np.sum(o) > 0:
#             # Obtain the instances
#             components = np.unique(o)
#             n_comp = len(components) - 1
#             detection_marks = np.zeros((n_comp, mi, ni))
#             for c in range(n_comp):
#                 # Only keeps current object.
#                 tmp = (o == components[c + 1])
#                 # Transform the label with erosion
#                 diskelem = skimage.morphology.disk(radious)
#                 detection_marks[c] = skimage.morphology.binary_erosion(tmp, diskelem)
#             detection_marks = np.sum(detection_marks, axis=0)
#             detection_marks = detection_marks.reshape((mi, ni))
#         else:
#             detection_marks = np.zeros((mi, ni))
#         o[o > 0] = 1
#         image[i, :, :, 0] = o
#         image[i, :, :, 1] = 1-o
#         image[i, :, :, 2] = detection_marks
#         image[i, :, :, 3] = 1-detection_marks
#         print('Loaded image {} from directory {}'.format(name, path))
#     print('loaded images from directory {} to shape {}'.format(path, image.shape))
#     return image

# -----------------------------------------------------------------------------

def make_weight_map(label, binary=True, w0=10, sigma=5):
    """
    Generates a weight map in order to make the U-Net learn better the
    borders of cells and distinguish individual cells that are tightly packed.
    These weight maps follow the methodololy of the original U-Net paper.

    The variable 'label' corresponds to a label image.

    The boolean 'binary' corresponds to whether or not the labels are
    binary. Default value set to True.

    The float 'w0' controls for the importance of separating tightly associated
    entities. Defaut value set to 10.

    The float 'sigma' represents the standard deviation of the Gaussian used
    for the weight map. Default value set to 5.
    """

    # Initialization.
    lab = np.array(label)
    lab_multi = lab

    # Get shape of label.
    rows, cols = lab.shape

    if binary==False:

        # Converts the label into a binary image with background = 0
        # and cells = 1.
        lab[lab == 255] = 1

        # Builds w_c which is the class balancing map. In our case, we want cells to have
        # weight 2 as they are more important than background which is assigned weight 1.
        w_c = np.array(lab, dtype=float)
        w_c[w_c == 1] = 1
        w_c[w_c == 0] = 0.5

        # Converts the labels to have one class per object (cell).
        lab_multi = measure.label(lab, neighbors=8, background=0)

    else:

        # Converts the label into a binary image with background = 0.
        # and cells = 1.
        lab[lab > 0] = 1

        # Builds w_c which is the class balancing map. In our case, we want cells to have
        # weight 2 as they are more important than background which is assigned weight 1.
        w_c = np.array(lab, dtype=float)
        w_c[w_c == 1] = 1
        w_c[w_c == 0] = 0.5

        # Converts the labels to have one class per object (cell).
        lab_multi = measure.label(lab, connectivity=1, background=0)

    components = np.unique(lab_multi)

    n_comp = len(components) - 1

    maps = np.zeros((n_comp, rows, cols))

    map_weight = np.zeros((rows, cols))

    if n_comp >= 2:
        for i in range(n_comp):
            # Only keeps current object.
            tmp = (lab_multi == components[i + 1])

            # Invert tmp so that it can have the correct distance.
            # transform
            tmp = ~tmp

            # For each pixel, computes the distance transform to
            # each object.
            maps[i][:][:] = scipy.ndimage.distance_transform_edt(tmp)

        maps = np.sort(maps, axis=0)

        # Get distance to the closest object (d1) and the distance to the second
        # object (d2).
        d1 = maps[0][:][:]
        d2 = maps[1][:][:]

        map_weight = w0 * np.exp(-((d1 + d2) ** 2) / (2 * (sigma ** 2))) * (lab == 0).astype(int);

    map_weight += w_c

    return map_weight

def contours_weight_map(label, contour_thickness = 10, blur_kernel_size = 9,
                        sigma=100, radious = 1, w0 = 10,
                        only_contours = False, background_weight = 0.05):
    """
    Generates a weight map in order to make the U-Net learn better the
    borders of cells
     
    label: labeled image (instances or binary), 2D numpy array
    
    The integer 'contour_thickness' controls for the size of each object contour.
    Default values is set to 10.
     
    The float 'blur_kernel_size' corresponds to the size of the Gaussian
    blurring applied to the edges of the binary objects
     
    The float 'sigma' corresponds to the standard deviation of the Gaussian used
    for the weight map. Default value set to 100.
     
    The integer 'radious' is the radious of the disk used in the erosion of the
    objects to detect the instances marks.
     
    The float 'w0' controls for the importance of correctly predicting cell edges.
    Defaut value set to 10.
     
    If only_contours = True, then the image has only higher weight for the contour of each object. Else, the body of the
    object is also weighted by w0.
     
    The float 'background_weight' controls for the importance of correctly 
    predicting image background. It is recommended to assign a low possitive 
    value rather than zero. Defaut value set to 10.
    """   
    # Initialization.
    lab = np.copy(label.astype(np.uint8))
    # Converts the label into a binary image with background = 0 and cells = 1.
    lab = lab>0
    lab = lab.astype(np.uint8)
    if np.sum(label)>0:
        components = np.unique(label)
        n_comp = len(components) - 1 
        # Image opening to remove any hole inside the segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        lab = cv2.morphologyEx(lab,cv2.MORPH_OPEN,kernel)

        c, hierarchy = cv2.findContours(lab, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.drawContours(np.zeros(lab.shape), c, -1, color=[1, 0, 0],
                                    thickness=contour_thickness)
        
        # Blur the contours of the image to generate the weights
        if blur_kernel_size > 0:
          # Obtain the instances to blurr correctly each contour
          maps = np.zeros((n_comp, lab.shape[0], lab.shape[1]))
          for i in range(n_comp):
              # Only keeps current object.
              tmp = (label == components[i + 1])
              # Transform the label with erosion
              selem = skimage.morphology.disk(radious)
              maps[i] = skimage.morphology.binary_erosion(tmp, selem)
          maps = np.sum(maps, axis=0)
          # maps contains the inner part of each instance
          maps = maps.astype(np.float32)
          lab_blur = cv2.GaussianBlur(contours.astype(np.float32),
                                      (blur_kernel_size, blur_kernel_size),
                                      sigma)
          map_weight = np.multiply(np.maximum(lab_blur,.0),(1-maps))
        else:
          map_weight = contours
        if only_contours == False:
            ## Add also the segmentation of the object
            map_weight[lab > 0] = 1

        # Normalize the values of the weights and enhance them
        map_weight = w0*(np.maximum(map_weight/np.max(map_weight), 0))
        map_weight[map_weight == 0] = background_weight
    else:
        map_weight = np.zeros(label.shape) + background_weight
    return map_weight

# -----------------------------------------------------------------------------

def do_save_wm(path, blur_kernel_size=0, **kwargs):
    """
    Retrieves the label images, applies the weight-map algorithm and save the
    weight maps in a folder.

    The string 'path' refers to the path where the weight maps should be saved.
    """
    names = os.listdir(path + "/labels")
    names.sort()

    for i, name in enumerate(names):
        label = read_image(os.path.join(path, "labels", name))

        labels_ = contours_weight_map(label, **kwargs)
        print('Weights of {} calculated.'.format(name))
        # Count number of digits in n. This is important for the number
        # of leading zeros in the name of the maps.

        # Save path with correct leading zeros.

        if not os.path.exists(path +"/weights"):
            os.makedirs(path +"/weights")
        file_name = name[:-4]
        path_to_save = "{0}/weights/{1}_weight.tif".format(path, file_name)
        # Saving files
        if blur_kernel_size > 0:
            sitk.WriteImage(sitk.GetImageFromArray(labels_.astype(np.float32)), path_to_save)
        else:
            cv2.imwrite(path_to_save, labels_.astype(np.uint8))
    print('Process finished.')
    return None


def do_save_marks(path):
    """
    Retrieves the label images in path/labels, calculates the markers and stores them in a new folder path/markers
    - 'path': directory where the labels and marks are stored.
    """
    names = os.listdir(path + "/labels")
    names.sort()
    for i, name in enumerate(names):
        labels, markers = read_instances(os.path.join(path, "labels", name), radious = 5)
        print('Markers of {} calculated.'.format(name))
        save_path = path +"/markers"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = name[:-4]
        path_to_save = save_path + "/{0}_markers.tif".format(file_name)
        # Saving files
        sitk.WriteImage(sitk.GetImageFromArray(markers.astype(np.uint16)), path_to_save)
    print('Process finished.')
    return None


def bleach_correction(inputI, bSigma):
    aux = sitk.GetImageFromArray(inputI)

    filter_Gauss = sitk.DiscreteGaussianImageFilter()
    filter_Gauss.SetVariance(bSigma ^ 2)

    filter_size = int(2 * np.ceil(2 * bSigma) + 1)
    filter_Gauss.SetMaximumKernelWidth(filter_size)

    blurred_aux = filter_Gauss.Execute(aux)
    blurred_aux = sitk.GetArrayFromImage(blurred_aux)

    bleach_correction = inputI - blurred_aux + np.mean(blurred_aux)

    return bleach_correction


def mean_match(inputI, mMean):
    # inputI input image
    # m is the mean value of your image type (1/2, 255/2, 65535/2, ...)

    outI = np.copy(inputI)
    outI = outI.astype(np.float32)
    O = np.ones((outI.shape))
    M = np.mean(outI)
    outI = outI + (mMean - M) * O;

    return outI

# def rigid_registration(path, sigma, mean_value, bitdepth):
#     # to get a good registration of the movies, all the frames in the videos need to be under the same conditions of
#     # intensity and bleach changes. This is important because the similiarity between the frames is given by the
#     # correlation between pixel intensities.
#
#     # registration is an array of dimensions (z-axis, y-axis, x-axis) and same size as the file in the path.
#     # After the registration, border cropping must be cosidered in order to avoid zeros.
#
#     # path: path to the stack. For example "10000_11-19-13_1002_xy008_266-268.tif","1002_xy008.nd2", "1002_xy008.mhd"
#     # sigma: size of the sigma in the gaussian blurr for the bleach correction
#     # mean_value: the mean intensity value of all the frames is going to be the same (2^15, 2^4,...)
#     if path[-3:] == 'mat':
#         # This command cannot be used for v 7.3 mat files, that is, for the files that are not compressed.
#         # stack = scipy.io.loadmat(path)
#         # stack = stack['V1']
#         f = h5py.File(path)
#         for k, v in f.items():
#             matlab_matrix = np.array(v)
#
#         #             Returned matrix is transposed.
#         stack2 = np.zeros((matlab_matrix.shape[0], matlab_matrix.shape[2], matlab_matrix.shape[1]))
#         for i in range(len(stack2)):
#             stack2[i] = np.transpose(matlab_matrix[i])
#
#     else:
#         if bitdepth == 'uint8':
#             stack2 = sitk.ReadImage(path, sitk.sitkUInt8)
#
#         elif bitdepth == 'uint16':
#             stack2 = sitk.ReadImage(path, sitk.sitkUInt16)
#
#         stack2 = sitk.GetArrayFromImage(stack2)
#
#     stack2 = stack2.astype(np.float32)
#
#     start = time.time()
#     for i in range(len(stack2)):
#         aux_bleach = linear_stretch(stack2[i], 0, mean_value * 2 - 1)
#         aux_bleach = bleach_correction(aux_bleach, sigma)
#         stack2[i] = mean_match(aux_bleach, mean_value)
#
#     end = time.time()
#     print('Illumination correction for registration finished', (end - start), 'secs')
#
#     registration = np.zeros((1, stack2.shape[1], stack2.shape[2]))
#     registration[0] = stack2[0]
#     registration = registration.astype(np.float32)
#
#     for i in range(len(stack2) - 1):
#         start = time.time()
#         result = ird.similarity(stack2[i], stack2[i + 1])
#
#         aux = result["timg"]
#         aux = aux.reshape((1, aux.shape[0], aux.shape[1]))
#
#         registration = np.concatenate((registration, aux))
#         registration = registration.astype(np.float32)
#
#         end = time.time()
#         print('Frame ', np.str(i), ' registered in ', (end - start), 'secs')
#
#     return registration