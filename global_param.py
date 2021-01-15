import cv2 as cv
import numpy as np

apl_1=2
apl_2=5
apl_3=10

resize_h=800
resize_w=600

dis_limit=5#表示点与点距离度量或者点与直线之间的距离度量阈值
"""
决定缩放后图像最大边的大小
max_size=1024
min_size=800
"""
# Max_size=1024
# Min_size=800
area_ratio =2/3 #contour reconstrucion algorithm

# BLENDING_LIST = ['gaussian','poisson', 'none', 'box', 'motion']
# BLENDING_LIST = ['gaussian', 'poisson', 'none', 'motion']
BLENDING_LIST = ['gaussian']
Indentically_distrubuted_probability = 0.2 #决定一幅图像中前景bu同的概率
Background_probability = 0.2 #决定背景被选取的概率
Pre_choose_probability = [0.1,0.2,0.7] #决定三面，双面,单面前景实例被选取的概率

INVERTED_MASK = False # Set to true if white pixels represent background
SYN_IMAGE_NUM= 3589#the numbers of that should generator images num_tall=SYN_IMAGE_NUM*len(BLENDING_LIST)

POISSON_BLENDING_DIR = './pb'
MIN_WIDTH = 15 # Minimum width of object to use for data generation
MIN_HEIGHT = 15 # Minimum height of object to use for data generation
dis_surface_limit=25 #faceted algorithm

#dis_limit=15  # to decided line1 to cross line2 and have the corner (in square_check.py @square_check)
def make():
    pass