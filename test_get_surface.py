import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math
import json
import glob

# from model.global_param import *
# from model.file_deal import  *
# from square_check import distance_point2point
# from model.global_param import *
from  model.get_surface import *

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def polygons_to_mask_surface(img, polygons,color):
    frame_ID_list_np = np.array(list(polygons), dtype=int)
    # cv2.polylines(img, [frame_ID_list_np], True, (0, 0, 255))
    cv2.fillPoly(img, [frame_ID_list_np], color)
    cv2.polylines(img, [frame_ID_list_np], True, color,thickness=8)
    return img
def parse_args():
    '''Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    # parser.add_argument("-ff","--foreground_files",
    #   help="The root directory which contains the images ,mask files and annotations of the foreground,"
    #        "image file path : */img/*.jpg;mask  file path : */mask/*/*.pbm;json  file path : */anno/*.json")
    parser.add_argument("-bf","--background_files",
      help="The root directory which contains the images,mask files and annotations of the background."
           "image file path : */img/*.jpg;mask  file path : */mask/*/*.pbm;json  file path : */anno/*.json")
    parser.add_argument("-sf","--save_files",
      help="The root directory which contains the images and annotations. ")
    # parser.add_argument("-NO","--noise_object",
    #   help="The image types will  generate. :Ture is positive images ,False is negative images")
    args = parser.parse_args()
    return args

def get_surface_result(file,save_file):
    img_files = get_list_of_images(file)
    for file_path in img_files:
        print('file path is :', file_path)
        img = cv2.imread(file_path)
        anno_path = get_list_of_labels(file_path)
        json_data = json.load(open(anno_path))
        all_data = json_data['shapes']

        for data in all_data:
            points = data['points']
            points = calc_surface(points)
            i = 0
            for point_list in points:
                img = polygons_to_mask_surface(img, point_list, colors[i])
                i = i + 1
        # img_save = file_path.replace('img', 'save')
        # cv2.imwrite(img_save, img)
        file_name = file_path.split('/')[-1]
        print('file name is :',file_name)
        img_save=os.path.join(save_file)
        if not os.path.exists(img_save):
            os.makedirs(img_save)
        img_save = os.path.join(img_save, file_name)
        cv2.imwrite(img_save, img)

if __name__ == "__main__":
    arg=parse_args()
    get_surface_result(arg.background_files,arg.save_files)
