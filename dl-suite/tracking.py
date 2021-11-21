"""
Created on Wed April 09 2021
@author: E. Gómez de Mariscal
GitHub username: esgomezm
"""
import os


import sys


# Process the binary images and create instance segmentations
# ----------------------------------------------------------------
# from analysis.prepare_videos4track import process_videos
# PATH = sys.argv[1]
# path2mask = sys.argv[2]
# path2mask = os.path.join(PATH, path2mask)
# OUTPUTPATH = os.path.join(PATH,"localizations")
# process_videos(path2mask, OUTPUTPATH, th=0, sigma=2, min_size=100, POSTPROCESS=True)

## Merge tracks and segmentations
# ----------------------------------------------------------------
# from analysis.merge_images_and_trackmate import process_track_dir
# PATH = sys.argv[1]
# path2tracks = os.path.join(PATH, "trackmate")
# path2mask = os.path.join(PATH,"reconstructed_videos")
# OUTPUTPATH = os.path.join(PATH,"merge_trackmate_nomitosis")
# process_track_dir(path2tracks, path2mask, OUTPUTPATH)


## Display calculated protrusions tips
# ----------------------------------------------------------------
from analysis.protrusion_extraction import process_videos_prot_localization
PATH = sys.argv[1]
videos_path = os.path.join(PATH,"merge_trackmate_nomitosis")
OUTPUTPATH = os.path.join(PATH,"protrusions_tips")
process_videos_prot_localization(videos_path, OUTPUTPATH, min_len=20, pixel_size=0.802)
