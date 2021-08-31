import os
import cv2
import numpy as np
# PATH2DATA = "/home/esgomezm/Documents/3D-PROTUCEL/data/Usiigaci"
PATH2DATA = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/praful/stack2im"
set_number = 105
COUNT = 729 # we already have 666 training images
new_pixelsize = 0.65 # 0.87 # um/pixel usiigaci
pixelsize = 0.806 # μm/pixel
portion = new_pixelsize/pixelsize
for s in range(set_number):
    image = cv2.imread(os.path.join(PATH2DATA,'training_data/set{0}'.format(s+1), 'raw.tif'), cv2.IMREAD_ANYDEPTH)
    image = cv2.imread(os.path.join(PATH2DATA, 'inputs', 'raw_{0:0>3}.tif'.format(int(COUNT))), cv2.IMREAD_ANYDEPTH)
    h = np.int(image.shape[0]*portion)
    w = np.int(image.shape[1]*portion)
    image = cv2.resize(image, (w,h),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(PATH2DATA, 'inputs', 'raw_{0:0>3}.tif'.format(int(COUNT))), image)

    image = cv2.imread(os.path.join(PATH2DATA, 'labels', 'instance_ids_{0:0>3}.tif'.format(int(COUNT))), cv2.IMREAD_ANYDEPTH)
    image = cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(PATH2DATA, 'labels', 'instance_ids_{0:0>3}.tif'.format(int(COUNT))), image)

    # Only for those annotations having keypoints
    
    image = cv2.imread(os.path.join(PATH2DATA, 'keypoints', 'instance_ids_{0:0>3}.tif'.format(int(COUNT))),
                       cv2.IMREAD_ANYDEPTH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(PATH2DATA, 'keypoints', 'instance_ids_{0:0>3}.tif'.format(int(COUNT))), image)


    COUNT = COUNT+1

from MU_Lux_CZ.data.data_handling import do_save_wm
do_save_wm(os.path.join(PATH2DATA, 'data2convine'))

### Cell tracking challenge data

PATH2DATA = "/home/esgomezm/Documents/3D-PROTUCEL/data/CTC/PhC-C2DL-PSC/stack2im/"
files = os.listdir(os.path.join(PATH2DATA, 'inputs'))
COUNT = 712 # we already have 666 training images
new_pixelsize = 1.6 # um/pixel usiigaci
pixelsize = 0.806 # μm/pixel
portion = new_pixelsize/pixelsize
for s in files:
    image = cv2.imread(os.path.join(PATH2DATA,'inputs', s), cv2.IMREAD_ANYDEPTH)
    h = np.int(image.shape[0]*portion)
    w = np.int(image.shape[1] * portion)
    image = cv2.resize(image, (w,h),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(PATH2DATA,'inputs/raw_{0:0>3}.tif'.format(int(COUNT))), image)

    image = cv2.imread(os.path.join(PATH2DATA,'labels','man_seg{0:0>3}.tif'.format(s[1:4])), cv2.IMREAD_ANYDEPTH)
    image = cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(PATH2DATA, 'labels/instance_ids_{0:0>3}.tif'.format(int(COUNT))), image)

    COUNT = COUNT+1

from MU_Lux_CZ.data.data_handling import do_save_wm, read_image
do_save_wm(os.path.join(PATH2DATA))


### Cell tracking challenge data
PATH2DATA = "/home/esgomezm/Documents/3D-PROTUCEL/data/CTC/FluoC3DLMDA231/stack2im"
files = os.listdir(os.path.join(PATH2DATA, 'inputs'))
COUNT = 716 # we already have 666 training images
new_pixelsize = 1.2 # um/pixel usiigaci
pixelsize = 0.806 # μm/pixel
portion = new_pixelsize/pixelsize
for s in files:
    image = cv2.imread(os.path.join(PATH2DATA,'inputs', s), cv2.IMREAD_ANYDEPTH)
    h = np.int(image.shape[0]*portion)
    w = np.int(image.shape[1] * portion)
    image = cv2.resize(image, (w,h),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(PATH2DATA,'inputs/raw_{0:0>3}.tif'.format(int(COUNT))), image)

    image = cv2.imread(os.path.join(PATH2DATA,'labels','man_seg_' + s[1:]), cv2.IMREAD_ANYDEPTH)
    image = cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(PATH2DATA, 'labels/instance_ids_{0:0>3}.tif'.format(int(COUNT))), image)

    COUNT = COUNT+1

from MU_Lux_CZ.data.data_handling import do_save_wm, read_image
do_save_wm(os.path.join(PATH2DATA))



#
# draw.ellipse((x-r, y-r, x+r, y+r), fill = 5 , outline = 5)
#                     draw2.ellipse((x-r, y-r, x+r, y+r), fill = 1 , outline = 1)
#
