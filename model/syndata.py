
import argparse
import glob
import sys
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import cv2
import numpy as np
import random
from PIL import Image
import scipy
from multiprocessing import Pool
from functools import partial
import signal
import time

# from defaults import *
import math
from pyblur import *
from collections import namedtuple
from .file_deal import *
from .global_param import *
sys.path.insert(0, POISSON_BLENDING_DIR)
from .pb import *

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def resizeImg(image):
    h, w = image.shape[:2]
    M=max(h,w)
    if M <=Max_size and Min_size<=M:
        return image
    scale=random.randint(Min_size,Max_size)
    if w > h :
        weight=scale
        height=int(h*(weight/w))
        size=(weight,height)
    else:
        height=scale
        weight=int(w*(height/h))
        size=(weight,height)
    img = cv2.resize(image, size)
    return img




def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:
        kerneldim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])



def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes))
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img




def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


def anno_point_to_mask(object_instance,mask_lists,img_size):
    """
    :param object_instance: one object instance
    :param mask_lists:  all the mask files
    :param img_size:  the size of scale images
    :return: the relation of one instance with mask file,tuple (points,maskfile)
    """
    w,h=img_size
    x =[]
    y=[]
    for point in object_instance['points']:
        x.append(point[0])
        y.append(point[1])
    xmin=min(x)/w
    xmax=max(x)/w
    ymin=min(y)/h
    ymax=max(y)/h
    point2mask=[]
    for mask in mask_lists:
        xmin1,xmax1,ymin1,ymax1=get_annotation_from_mask_file(mask)
        x0=xmin1-xmin
        y0=ymin1-ymin
        x1=xmax1-xmax
        y1=ymax1-ymax
        dis1=math.pow(x0*x0+y0*y0, 0.5)
        dis2=math.pow(x1*x1+y1*y1, 0.5)
        if dis1< dis_limit and dis2 <dis_limit:
            point2mask.append((object_instance,mask))
            break
    if len(point2mask)==0:
        print('error : there is not fitted mask for point')
        return -1
    return point2mask

def create_image_anno(objects, bg_img,save_img_file,label_json_data,save_anno_file,blending_list=['none']):
    if 'none' not in save_img_file:
        print('Error: save img file is error')
        return -1
    all_objects = objects
    assert len(all_objects) > 0
    while True:
        background = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
        backgrounds = []
        for i in range(len(blending_list)):
            backgrounds.append(background.copy())
        for obj in all_objects:
            foreground = Image.fromarray(cv2.cvtColor(obj[0], cv2.COLOR_BGR2RGB))
            img_mask=Image.fromarray(cv2.cvtColor(obj[1], cv2.COLOR_BGR2RGB))
            img_mask=img_mask.convert('L')
            xmin, xmax, ymin, ymax = get_annotation_from_mask(img_mask)
            x=xmin
            y=ymin
            if xmin == -1 or ymin == -1 or xmax - xmin < MIN_WIDTH or ymax - ymin < MIN_HEIGHT:
                continue
            foreground = foreground.crop((xmin, ymin, xmax, ymax))
            orig_w, orig_h = foreground.size
            mask = img_mask
            mask = mask.crop((xmin, ymin, xmax, ymax))
            # mask.show()
            if INVERTED_MASK:
                mask = Image.fromarray(255 - PIL2array1C(mask))
            for i in range(len(blending_list)):
                if blending_list[i] == 'none' or blending_list[i] == 'motion':
                    backgrounds[i].paste(foreground, (x, y), mask)
                elif blending_list[i] == 'poisson':
                    offset = (y, x)
                    img_mask = PIL2array1C(mask)
                    img_src = PIL2array3C(foreground).astype(np.float64)
                    img_target = PIL2array3C(backgrounds[i])
                    img_mask, img_src, offset_adj \
                        = create_mask(img_mask.astype(np.float64),
                                      img_target, img_src, offset=offset)
                    background_array = poisson_blend(img_mask, img_src, img_target,
                                                     method='normal', offset_adj=offset_adj)
                    backgrounds[i] = Image.fromarray(background_array, 'RGB')
                elif blending_list[i] == 'gaussian':
                    backgrounds[i].paste(foreground, (x, y),
                                         Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask), (5, 5), 2)))
                elif blending_list[i] == 'box':
                    backgrounds[i].paste(foreground, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask), (3, 3))))
        break
    for i in range(len(blending_list)):
        if blending_list[i] == 'motion':
            backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
        imgpath=save_img_file.replace('none', blending_list[i])
        label_json_data['imagePath']=imgpath.split('/')[-1]
        label_json_data['imageData']=[]
        backgrounds[i].save(imgpath)
        json.dump(label_json_data, open(save_anno_file.replace('none', blending_list[i]), 'w'))
    return backgrounds