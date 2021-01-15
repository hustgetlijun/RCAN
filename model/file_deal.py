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
from .global_param import *

def get_fore_path(foreground_files,numberofsurface):
    """
    :param foreground_files:  data/example/fore/img
    :param numberofsurface: 'one' or 'two' or 'three'
    :return: file_path eg:data/example/fore/img/one/1.jpg or data/example/fore/img/two/0/1.jpg or  data/example/fore/img/three/0/1.jpg
    """
    global file_path
    home_paths = os.path.join(foreground_files, numberofsurface)
    for home,files,papers in os.walk(foreground_files):
        if home == home_paths :
            if numberofsurface == 'one':
                paper=random.choice(papers)
                file_path = os.path.join(home,paper)
                # return file_path_one
            elif numberofsurface == 'two':
                file_path = []
                file=random.choice(files)
                paper=os.path.join(home,file)
                file_path = glob.glob(os.path.join(paper, '*.jpg'))
                # return file_path_two
            elif numberofsurface == 'three':
                # print('numberofsurface is :', numberofsurface)
                file_path = []
                file=random.choice(files)
                paper = os.path.join(home, file)
                file_path = glob.glob(os.path.join(paper, '*.jpg'))
            else:
                return -1
        else:
            continue
    # print('pre json file path is :',file_path)
    return file_path

def get_list_of_images(root_dir):
    """
    image file path : */img/*.jpg
    mask  file path : */mask/*/*.pbm
    json  file path : */anno/*.json
    """
    print("root_dir:%s" % root_dir)
    img_list = glob.glob(os.path.join(root_dir, 'img/*.jpg'))
    return img_list

def get_mask_path(image_path):
    """
    :param image_path: choiced image :string
    :return: all mask path :strings
    """
    image_name=image_path.split('/')[-1]
    img_file=image_name.split('.')[-2]
    # num = re.findall(r"\d+", image_name)
    mask_file=image_path.replace(image_name,img_file)
    mask_file = mask_file.replace('img', 'mask')
    mask_file_list=glob.glob(os.path.join(mask_file, '*.pbm'))
    return mask_file_list

def get_list_of_labels(image_path):
    """
    :param image_path: object instance: string
    :return: the annonation  path: string
    """
    mask_file=image_path.replace('.jpg','.json')
    # mask_file.split('/')[-2].replace('img', 'anno')
    mask_file = mask_file.replace('img', 'anno')
    return mask_file

def get_annotation_from_mask(mask):
    '''Given a mask, this returns the bounding box annotations
    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations

    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file)
        if INVERTED_MASK:
            mask = 255 - mask
        rows = np.any(mask, axis=1)#行
        cols = np.any(mask, axis=0)#列
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        print("%s not found. Using empty mask instead."%mask_file)
        return -1, -1, -1, -1
# def get_data(img_root,label_root,imgfile_save,jsonfile_save,flag):
#    imgfiles=get_list_of_images(img_root)
#    labels=get_list_of_labels(label_root)