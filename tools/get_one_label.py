import argparse
import glob
import sys
import os
import cv2
import json
import random
import numpy as np
import shutil

def get_list_of_labels(label_root):
    print("label_dir:%s" % label_root)
    label_list = glob.glob(os.path.join(label_root, '*.json'))
    print("label_json_list:%d" % len(label_list))
    return label_list

def get_data(label_root,jsonfile_save):
   labels = get_list_of_labels(label_root)
   num0=0
   for label_path in labels:
       print('the number of json file is :',num0)
       json_data=json.load(open(label_path))
       json_data['imageData']=[]
       imagePath=json_data['imagePath']
       img_num=imagePath.split('.')[0]
       for num in range(len(json_data['shapes'])):
           json_data['shapes'][num]['label']='carton'
       num_file = label_path.split('/')[-1]
       json_num=num_file.split('.')[0]
       # if json_num != img_num:
       #     print('EEROR')
       #     print('json path is :',label_path)
       #     return -1

       if not os.path.exists(jsonfile_save):
           os.makedirs(jsonfile_save)
       save_anno_file = os.path.join(jsonfile_save, '{}.json'.format(json_num))
       # json.dump(json_data, open(save_anno_file), 'w')
       with open(save_anno_file, 'w') as fd:
           json.dump(json_data, fd)
       num0=num0+1



if __name__ == '__main__':
    # get_data('./experiment1/dataset1/train/image/train3000/test','./experiment1/dataset1/train/anotations/train3000/test','experiment1/dataset1/test500/image','experiment1/dataset1/test500/anotations','test')
    get_data('./train_data/merge/anno',
             './train_data/merge/one_anno')
