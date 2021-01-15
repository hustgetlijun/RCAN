from matplotlib import pyplot as plt
import cv2
import numpy as np
# from get_lines import *
"""
@get_perspective_pram
input: src: source square 4 points
       dis: destination square 4 points 
output: M warp params
"""
def get_perspective_pram(src,dis):
    src=np.array(src)
    src=np.float32(src)
    dis=np.array(dis)
    dis = np.float32(dis)
    M = cv2.getPerspectiveTransform(src, dis)
    return M
def image_warp(img,img_size,src_point,dis_point):
    M=get_perspective_pram(src_point,dis_point)
    # cv2.imshow('src img',img)
    dst = cv2.warpPerspective(img, M, img_size)
    # cv2.imshow('dis img',dst)
    # cv2.waitKey()
    return dst

