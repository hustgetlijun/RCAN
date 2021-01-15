import shutil
import argparse
import glob
import sys
import os
import cv2
import numpy as np
import random
from PIL import Image
from functools import partial
import json

from model.generatedata import *

def parse_args():
    '''Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    parser.add_argument("-ff","--foreground_files",
      help="The root directory which contains the images ,mask files and annotations of the foreground,"
           "image file path : */img/*.jpg;mask  file path : */mask/*/*.pbm;json  file path : */anno/*.json")
    parser.add_argument("-bf","--background_files",
      help="The root directory which contains the images,mask files and annotations of the background."
           "image file path : */img/*.jpg;mask  file path : */mask/*/*.pbm;json  file path : */anno/*.json")
    parser.add_argument("-sf","--save_files",
      help="The root directory which contains the images and annotations. ")
    # parser.add_argument("-NO","--noise_object",
    #   help="The image types will  generate. :Ture is positive images ,False is negative images")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    arg=parse_args()
    gen_syn_datas(arg.foreground_files,arg.background_files,arg.save_files)

#python main.py -ff ./data/fore/img -bf ./data/back  -sf ./data/nosie_100  -NO ./data/example/noise_object/
