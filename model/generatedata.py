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
import PIL.Image
import argparse
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
# from labelme import utils
import numpy as np
import glob
import PIL.Image
import cv2
import math
import json
import copy

from .syndata import *
from .file_deal import *
from .global_param import *
from .image_transform import *
from .surface_deal import *
from .get_surface import *


def getbbox(self, points):
    x_ = []
    y_ = []
    for point in points:
        x_.append(point[0])
        y_.append(point[1])
    xmin = int(min(x_))
    xmax = int(max(x_))
    ymin = int(min(y_))
    ymax = int(max(y_))
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    frame_ID_list_np = np.array(list(polygons), dtype=int)
    cv2.polylines(mask, [frame_ID_list_np], True, (255, 255, 255))
    cv2.fillPoly(mask,[frame_ID_list_np],(255,255,255))
    return mask

def polygons_to_mask_merge(mask, polygons):
    frame_ID_list_np = np.array(list(polygons), dtype=int)
    cv2.polylines(mask, [frame_ID_list_np], True, (255, 255, 255))
    cv2.fillPoly(mask,[frame_ID_list_np],(255,255,255))
    return mask

def get_min_and_max(points):
    xmin=9999
    ymin=9999
    xmax=-1
    ymax=-1
    # print('points is ', points)
    for i in range(len(points)):
        points_new = np.concatenate(([points[i][0], points[i][1]], [points[i][2], points[i][3]]), axis=0)
        # print('points_new is :', points_new)
        line_new = points_new.transpose()
        # print('line_new is :',line_new)
        xmin_0 = min(line_new[0])
        xmax_0 = max(line_new[0])
        ymin_0 = min(line_new[1])
        ymax_0 = max(line_new[1])
        if xmin_0 < xmin:
            xmin = xmin_0
        if ymin_0 < ymin :
            ymin=ymin_0
        if xmax < xmax_0:
            xmax=xmax_0
        if ymax < ymax_0:
            ymax = ymax_0
    return ([xmin,ymin,xmax,ymax])

def coordinate_translation(trans,points):
    x,y=copy.copy(trans)
    numberofsurface=len(points)
    if numberofsurface ==1:
        points1 = [[]]
    elif numberofsurface ==2:
        points1=[[],[]]
    elif numberofsurface ==3:
        points1=[[],[],[]]
    # print('points input is:',points)
    for j in range(len(points)):
        for i in range(len(points[j])):
            x0=0
            y0=0
            x0,y0 = points[j][i]
            # print('point is :',point)
            points1[j].append([int(x0 + x),int(y0 + y)])
    # print('points output is :',points1)
    return points1

def get_sequence_points(surfaces):
    number_face=len(surfaces)
    # print('surfaces  is ------------------  :',surfaces)
    result=[]
    if number_face == 2:
        surface0 = copy.copy(surfaces[0])
        surface1 = copy.copy(surfaces[1])
        edg_point = []
        for j in range(len(surface0)):
            if surface0[j] in surface1:
                edg_point.append(surface0[j])
        index0_0 = surface0.index(edg_point[0])
        index0_1 = surface0.index(edg_point[1])
        if index0_0 == 0 and index0_1 == 3:
            surface_1 = [surface0[index0_0], surface0[index0_1], surface0[2], surface0[1]]
        else:
            if index0_1 == 3:
                surface_1 = [surface0[index0_0], surface0[index0_1], surface0[0], surface0[index0_0-1]]
            else:
                surface_1 = [surface0[index0_0], surface0[index0_1], surface0[index0_1+1], surface0[index0_0-1]]
        index1_0 = surface1.index(edg_point[0])
        index1_1 = surface1.index(edg_point[1])
        if index1_0 < index1_1 :
            if index1_0 == 0 and index1_1 == 3:
                surface_2 = [surface0[index0_0], surface0[index0_1], surface1[2], surface1[1]]
            else:
                if index1_1 == 3:
                    surface_2 = [surface0[index0_0], surface0[index0_1], surface1[0], surface1[index1_0 - 1]]
                else:
                    surface_2 = [surface0[index0_0], surface0[index0_1], surface1[index1_1 + 1],surface1[index1_0 - 1]]
        else:
            if index1_1 == 0 and index1_0 == 3:
                surface_2 = [surface0[index0_0], surface0[index0_1], surface1[1], surface1[2]]
            else:
                if index1_0 == 3:
                    surface_2 = [surface0[index0_0], surface0[index0_1], surface1[index1_1-1], surface1[0]]
                else:
                    surface_2 = [surface0[index0_0], surface0[index0_1], surface1[index1_1 - 1],surface1[index1_0 + 1]]
        result.append({'face': surface_1, 'comm_edg': [edg_point]})
        result.append({'face': surface_2, 'comm_edg': [edg_point]})
    elif number_face == 3:
        surface_0 = surfaces[0]
        surface_1 = surfaces[1]
        surface_2 = surfaces[2]
        comm_point=[]
        for number_point in range(len(surface_0)):#求公共点
            if surface_0[number_point] in surface_1 and surface_0[number_point] in surface_2:
                comm_point=surface_0[number_point]
        neighbor_points=[]
        centre_point=[]
        for idofsurface in range(len(surfaces)): #计算每个面的中心点坐标，当y最小时对应的面为顶面
            surface = surfaces[idofsurface]
            c_x=0
            c_y=0
            for idofpoint in range(len(surface)):
                c_x=c_x + surface[idofpoint][0]
                c_y=c_y + surface[idofpoint][1]
            c_x=c_x/4
            c_y=c_y/4
            centre_point.append([c_x,c_y])
        y_list = [centre_point[0][1],centre_point[1][1],centre_point[2][1]]
        ##############################
        x_list = [centre_point[0][0],centre_point[1][0],centre_point[2][0]]
        y_min = min(y_list)
        x_=[]
        for k_num in range(3):
            if y_min == y_list[k_num]:
                x_.append(centre_point[k_num][0])
        if len(x_) >1 :
            x_min = min(x_)
            id_min = x_list.index(x_min)
        else:
            id_min = y_list.index(y_min)  # 求解顶部对应的面


        sur_1 = surfaces[id_min]
        indexofcomm = sur_1.index(comm_point)
        finnal_sur_1=[]
        if indexofcomm ==3:
            neighbor_points.append(sur_1[indexofcomm - 1])
            neighbor_points.append(sur_1[0])
        else:
            neighbor_points.append(sur_1[indexofcomm - 1])
            neighbor_points.append(sur_1[indexofcomm + 1])
        if neighbor_points[0][0] < neighbor_points[1][0] :
            first_id = sur_1.index(neighbor_points[0])
            finnal_sur_1=[neighbor_points[0],comm_point,neighbor_points[1],sur_1[first_id-1]]
        else:
            first_id = sur_1.index(neighbor_points[1])
            finnal_sur_1 = [neighbor_points[1], comm_point, neighbor_points[0], sur_1[first_id - 1]]
        first_point = finnal_sur_1[0]
        # print('surfacs1 is :',finnal_sur_1)
        first_surface=[]
        last_surface=[]
        # print('surfacs is :',surfaces)
        for number in range(3): #根据前景标注规则（顶面逆时针，左顺顶面，右面逆时针）确定另外两个面的顺序
            if first_point in surfaces[number] :
                if finnal_sur_1[2] in surfaces[number]:
                    # print('surface num ', (number, surfaces[number]))
                    continue
                else:
                    first_surface=copy.deepcopy(surfaces[number])
                    # print('surface num ', (number, surfaces[number]))
                    # break

            else:
                last_surface=surfaces[number]
        finnal_sur_0=[]
        # print('first point su :',first_point)
        # print('first surface is :',first_surface)
        if len(first_surface) ==0:
            return -1
        id_firstpoint=first_surface.index(first_point)
        if first_surface[id_firstpoint-1] == comm_point :
            if id_firstpoint ==3:
                finnal_sur_0 = [first_point, comm_point, first_surface[1],
                                first_surface[0]]
            else:
                id_com = first_surface.index(comm_point)
                finnal_sur_0 = [first_point, comm_point, first_surface[id_com-1],
                                first_surface[id_firstpoint+1]]
        else:
            id_lastpoint = first_surface.index(first_surface[id_firstpoint-1])
            finnal_sur_0=[first_point,comm_point,first_surface[id_lastpoint-1],first_surface[id_firstpoint-1]]
        # id_lastpoint = first_surface.index(first_surface[id_firstpoint-1])
        # finnal_sur_0=[first_point,comm_point,first_surface[id_lastpoint-1],first_surface[id_firstpoint-1]]
        result.append({'face': finnal_sur_0, 'comm_edg': [finnal_sur_0[0], finnal_sur_0[2]]})
        result.append({'face': finnal_sur_1, 'comm_edg': [finnal_sur_1[0], finnal_sur_1[2]]})
        finnal_sur_2=[]
        for num in range(4):
            if last_surface[num] in [finnal_sur_1[2], comm_point, finnal_sur_0[2]] :
                continue
            else:
                finnal_sur_2 = [finnal_sur_1[2], comm_point, finnal_sur_0[2],
                                last_surface[num]]
        result.append({'face': finnal_sur_2, 'comm_edg': [finnal_sur_2[0], finnal_sur_2[2]]})
    return result

def gen_syn_datas(foreground_files,background_files,save_img_file):
    assert os.path.exists(foreground_files)
    assert os.path.exists(background_files)
    background_files = get_list_of_images(background_files)
    # NUM=len(background_files)
    syn_number = 0
    global flag
    flag = True
    print('/*******begain***********/')
    while flag:
            same_object_in_img = random.random() #确定是否选取同一目标
            back_img_path=random.choice(background_files)
            print(' the number generated is :', syn_number)
            back_json_file = get_list_of_labels(back_img_path)
            back_label_data = json.load(open(back_json_file))
            back_objects = back_label_data['shapes']
            back_img = cv2.imread(back_img_path)
            # objs_num=len(back_objects)

            fore_img_path_global_one_ = copy.deepcopy(get_fore_path(foreground_files, 'one'))
            fore_img_path_global_two_ = copy.deepcopy(get_fore_path(foreground_files, 'two'))
            fore_img_path_global_three =copy.deepcopy(get_fore_path(foreground_files, 'three'))
            surface_choose_probability = random.random()

            h_back, w_back = back_img.shape[:2]
            pre_obj_mask=[]
            objects_inf = []
            for data_num in range(len(back_objects)): #选取背景中的实例
                noise_choose_flag = False
                if surface_choose_probability <= Pre_choose_probability[0]:
                    fore_img_path_global_one = random.choice(fore_img_path_global_three)
                    fore_img_path_global_two = random.sample(fore_img_path_global_three, 2)
                elif Pre_choose_probability[0] < surface_choose_probability and surface_choose_probability <= (
                        Pre_choose_probability[0] + Pre_choose_probability[1]):
                    fore_img_path_global_one = random.choice(fore_img_path_global_two_)
                    fore_img_path_global_two = fore_img_path_global_two_
                else:
                    fore_img_path_global_one = fore_img_path_global_one_
                    fore_img_path_global_two = fore_img_path_global_two_

                data = back_objects[data_num]
                label = copy.deepcopy(data['label'])
                data['label']= 'carton'
                flag_label = label.split('-')[-1]
                points_ori = data['points']
                # print('points ori ', points_ori)
                object_inf = copy.deepcopy(data)

                points = calc_surface(points_ori) #计算一个实例由几个面组成
                state_flag = False
                for surface_nums in range(len(points)):
                    if len(points[surface_nums]) <3 :
                        state_flag = True
                if state_flag :
                    print('state flage is :',state_flag)
                    continue
                # print('points is :',points)
                surface_ori=copy.deepcopy(points)
                full_surface=construction_surface(points, flag_label) # 构造完整轮廓
                # print('full_surface is :',full_surface)
                surface=full_surface.surface

                img_shape = back_img.shape
                back_mask_imgs = np.zeros(img_shape, dtype=np.uint8)
                backmaskimg=copy.deepcopy(back_mask_imgs)
                for i in range(len(surface_ori)): #求整并转换为mask
                    back_points_list = []
                    for point in surface_ori[i]:
                        x = int(point[0])
                        y = int(point[1])
                        back_points_list.append([x, y])
                    back_mask_imgs = polygons_to_mask_merge(back_mask_imgs, back_points_list)
                if surface == -1 or len(surface) == 0:
                    continue
                surface_flag = False
                for surface_id in range(len(surface)):
                    if len(surface[surface_id]) !=4:
                        surface_flag = True
                if surface_flag:
                    print('surface is:',surface)
                    continue
                xmin, ymin, xmax, ymax = get_min_and_max(surface)  # 平移小于0的坐标点
                if xmin < 0:
                    deta_x = abs(xmin)
                else:
                    deta_x = 0
                if ymin < 0:
                    deta_y = abs(ymin)
                else:
                    deta_y = 0
                deta = [deta_x, deta_y]  # 由于构造面超出图像边界故需要对实例进行平移
                # print('deta  is      :',deta)
                # print('surface is :',surface)
                surface_tran = copy.copy(coordinate_translation(deta, surface))  # 平移构造后的坐标
                points_tran = copy.copy(coordinate_translation(deta, surface_ori)) # 平移原始坐标
                if len(surface) ==1: #针对单面进行图像融合
                    print('-------------------------pre image one-----------------')
                    background_as_pre_object = random.random()
                    if background_as_pre_object < Background_probability:
                        w_nosie = random.randint(480, 800)
                        h_nosie = random.randint(480, 800)
                        toall = w_nosie * h_nosie * 3
                        Random_byte = bytearray(os.urandom(toall))
                        flatarry = np.array(Random_byte)
                        pre_img = flatarry.reshape(h_nosie, w_nosie, 3)
                        W_bbox = random.randint(30, int(w_nosie/2))
                        H_bbox = random.randint(30, int(h_nosie / 2))
                        x_bbox = random.randint(0, int(w_nosie/2))
                        y_bbox = random.randint(0, int(h_nosie / 2))
                        fore_objects = [{'points':[[x_bbox,y_bbox],[x_bbox+W_bbox,y_bbox],[x_bbox+W_bbox,y_bbox+H_bbox],[x_bbox,y_bbox+H_bbox]]}]
                        noise_choose_flag = True
                    else:
                        if same_object_in_img < Indentically_distrubuted_probability:
                            fore_img_path = get_fore_path(foreground_files, 'one')
                        else:
                            fore_img_path = fore_img_path_global_one
                        fore_img_path_one = fore_img_path
                        fore_json_path = get_list_of_labels(fore_img_path_one)
                        fore_label_data = json.load(open(fore_json_path))
                        fore_objects = fore_label_data['shapes']
                        print('fore_img_path_one is :', fore_img_path_one)
                        pre_img = cv2.imread(fore_img_path_one)

                    h_pre, w_pre = pre_img.shape[:2]
                    pre_points_list = []
                    for point in fore_objects[0]['points']:
                        x = int(point[0])
                        y = int(point[1])
                        pre_points_list.append([x, y])
                    pre_img_mask = polygons_to_mask([h_pre, w_pre, 3], pre_points_list)
                    pre_img_object = cv2.bitwise_and(pre_img, pre_img_mask) #提取目标实例
                    if len(pre_points_list) !=4:
                        noise_choose_flag = True
                        break
                    if len(surface_tran[0]) !=4:
                        noise_choose_flag = True
                        break
                    pre_img_change = image_warp(pre_img_object, (2*w_back, 2*h_back), pre_points_list, surface_tran[0])

                    back_img_mask_2x = polygons_to_mask([2*h_back, 2*w_back, 3], points_tran[0])
                    pre_img2back_img= cv2.bitwise_and(pre_img_change, back_img_mask_2x) #提取目标实例
                    finall_pre_img=pre_img2back_img[int(deta_y):int(deta_y+h_back),int(deta_x):int(deta_x+w_back)]
                    pre_img_merge = cv2.addWeighted(finall_pre_img, 0.7, backmaskimg, 0.7, 0)
                    pre_obj_mask.append((pre_img_merge,back_mask_imgs))
                    # choose_syn[0] = choose_syn[0] - 1
                elif len(surface) ==2:
                    # continue
                    print('-------------------------pre image two-----------------')

                    background_as_pre_object = random.random()
                    pre_img =[]
                    fore_point_list =[]
                    back_mask_imgs_trans = np.zeros([2*h_back,2*w_back,3], dtype=np.uint8)
                    for i in range(len(points_tran)):  # 求整并转换为mask
                        back_points_list = []
                        for point in points_tran[i]:
                            x = int(point[0])
                            y = int(point[1])
                            back_points_list.append([x, y])
                        back_mask_imgs_trans = polygons_to_mask_merge(back_mask_imgs_trans, back_points_list)
                    surface_com=get_sequence_points(surface_tran)

                    # fore_img_path = get_fore_path(foreground_files, 'two')

                    # print('fore img path is :',fore_img_path)
                    if background_as_pre_object < Background_probability:
                        # fore_img_path = get_fore_path(noise_files,'two')
                        for num_noise in range(2):
                            w_nosie = random.randint(480, 800)
                            h_nosie = random.randint(480, 800)
                            toall = w_nosie * h_nosie * 3
                            Random_byte = bytearray(os.urandom(toall))
                            flatarry = np.array(Random_byte)
                            pre_img_ = flatarry.reshape(h_nosie, w_nosie, 3)
                            W_bbox = random.randint(100, int(w_nosie / 2))
                            H_bbox = random.randint(100, int(h_nosie / 2))
                            x_bbox = random.randint(0, int(w_nosie / 2))
                            y_bbox = random.randint(0, int(h_nosie / 2))
                            fore_objects_ = [{'points': [[x_bbox, y_bbox], [x_bbox + W_bbox, y_bbox],
                                                        [x_bbox + W_bbox, y_bbox + H_bbox], [x_bbox, y_bbox + H_bbox]]}]
                            pre_img.append(pre_img_)
                            fore_point_list.append(fore_objects_[0]['points'])
                        noise_choose_flag = True
                    else:
                        if same_object_in_img < Indentically_distrubuted_probability:
                            fore_img_path = get_fore_path(foreground_files, 'two')
                        else:
                            fore_img_path = fore_img_path_global_two
                        fore_img_path_0 = fore_img_path[0]
                        fore_img_path_1 = fore_img_path[1]
                        fore_img_files=[fore_img_path_0,fore_img_path_1]
                        for num_fore in range(2):
                           pre_img.append(cv2.imread(fore_img_files[num_fore]))
                        # print('two path is :', fore_img_files)
                        fore_json_path_0 = get_list_of_labels(fore_img_path_0)
                        fore_json_path_1 = get_list_of_labels(fore_img_path_1)
                        # print('fore json path is :', [fore_json_path_0,fore_json_path_1])
                        fore_label_data_0 = json.load(open(fore_json_path_0))
                        fore_label_data_1 = json.load(open(fore_json_path_1))
                        fore_point_list=[fore_label_data_0['shapes'][0]['points'],fore_label_data_1['shapes'][0]['points']]
                    pre_img_change=[]
                    for indexofsurface in range(len(surface_com)):
                        surface_deal=surface_com[indexofsurface]['face'] #原始数据点
                        fore_objects = fore_point_list[indexofsurface]   #前景数据点
                        h_pre, w_pre = pre_img[indexofsurface].shape[:2]
                        pre_points_list = []
                        for point in fore_objects:
                            x = int(point[0])
                            y = int(point[1])
                            pre_points_list.append([x, y])
                        pre_img_mask = polygons_to_mask([h_pre, w_pre, 3], pre_points_list)
                        pre_img_object = cv2.bitwise_and(pre_img[indexofsurface], pre_img_mask)  # 提取目标实例
                        pre_img_change.append(image_warp(pre_img_object, (2 * w_back, 2 * h_back), pre_points_list,
                                                    surface_deal))
                    pre_img_merge=cv2.addWeighted(pre_img_change[0],0.7,pre_img_change[1],0.7,0)
                    pre_img2back_img= cv2.bitwise_and(pre_img_merge, back_mask_imgs_trans) #提取目标实例
                    finall_pre_img=pre_img2back_img[int(deta_y):int(deta_y+h_back),int(deta_x):int(deta_x+w_back)]
                    pre_obj_mask.append((finall_pre_img,back_mask_imgs))
                elif len(surface) == 3:
                    # continue
                    print('-------------------------pre image three-----------------')
                    background_as_pre_object = random.random()
                    pre_img =[]
                    fore_point_list =[]
                    back_mask_imgs_trans = np.zeros([2 * h_back, 2 * w_back, 3], dtype=np.uint8)
                    for i in range(len(points_tran)):  # 求整并转换为mask
                        back_points_list = []
                        for point in points_tran[i]:
                            x = int(point[0])
                            y = int(point[1])
                            back_points_list.append([x, y])
                        back_mask_imgs_trans = polygons_to_mask_merge(back_mask_imgs_trans, back_points_list)
                    surface_pairs=get_sequence_points(surface_tran)
                    if surface_pairs == -1:
                        print('three erro:2')
                        continue
                    if background_as_pre_object < Background_probability:
                        # fore_img_path = get_fore_path(noise_files,'three')
                        for num_noise in range(3):
                            w_nosie = random.randint(480, 800)
                            h_nosie = random.randint(480, 800)
                            toall = w_nosie * h_nosie * 3
                            Random_byte = bytearray(os.urandom(toall))
                            flatarry = np.array(Random_byte)
                            pre_img_ = flatarry.reshape(h_nosie, w_nosie, 3)
                            W_bbox = random.randint(100, int(w_nosie / 2))
                            H_bbox = random.randint(100, int(h_nosie / 2))
                            x_bbox = random.randint(0, int(w_nosie / 2))
                            y_bbox = random.randint(0, int(h_nosie / 2))
                            fore_objects_ = [{'points': [[x_bbox, y_bbox], [x_bbox + W_bbox, y_bbox],
                                                        [x_bbox + W_bbox, y_bbox + H_bbox], [x_bbox, y_bbox + H_bbox]]}]
                            pre_img.append(pre_img_)
                            fore_point_list.append(fore_objects_[0]['points'])
                        noise_choose_flag = True
                    else:
                        if same_object_in_img < Indentically_distrubuted_probability:
                            fore_img_path = get_fore_path(foreground_files, 'three')
                        else:
                            fore_img_path = fore_img_path_global_three
                        for numsurfacein in range(3):
                            img_path = fore_img_path[numsurfacein]
                            json_path = get_list_of_labels(img_path)
                            fore_jsondata = json.load(open(json_path))
                            fore_object_points = fore_jsondata['shapes'][0]['points']
                            fore_point_list.append(fore_object_points)
                            pre_img.append(cv2.imread(img_path))
                    pre_img_change = []
                    for indexofsurface in range(len(surface_pairs)):
                        surface_deal = surface_pairs[indexofsurface]['face']  # 原始数据点
                        h_pre, w_pre = pre_img[indexofsurface].shape[:2]
                        pre_points_list = []
                        for point in fore_point_list[indexofsurface]:
                            x = int(point[0])
                            y = int(point[1])
                            pre_points_list.append([x, y])
                        pre_img_mask = polygons_to_mask([h_pre, w_pre, 3], pre_points_list)
                        pre_img_object = cv2.bitwise_and(pre_img[indexofsurface], pre_img_mask)  # 提取目标实例
                        if len(pre_points_list) !=4 or len(surface_deal) != 4:
                            # print('pre img path is :',img_path)
                            # print('pre points list is :', pre_points_list)
                            # print('surface_deal is :', surface_deal)
                            return -1
                            # continue
                        pre_img_change.append(image_warp(pre_img_object, (2 * w_back, 2 * h_back), pre_points_list,
                                                         surface_deal))
                    pre_img_merge = cv2.addWeighted(pre_img_change[0],0.7, pre_img_change[1],0.7, 0)
                    pre_img_merge = cv2.addWeighted(pre_img_merge, 1, pre_img_change[2], 0.7, 0)
                    pre_img2back_img = cv2.bitwise_and(pre_img_merge, back_mask_imgs_trans)  # 提取目标实例
                    finall_pre_img = pre_img2back_img[int(deta_y):int(deta_y + h_back),
                                     int(deta_x):int(deta_x + w_back)]
                    pre_obj_mask.append((finall_pre_img, back_mask_imgs))
                # choose_syn[0] = choose_syn[0] - 1
                if noise_choose_flag:
                    print('no object is choosed')
                else :
                    objects_inf.append(object_inf)
            if len(pre_obj_mask)==0:
                print('pre_obj_mask is zero')
                continue
            if not os.path.exists(os.path.join(save_img_file, 'save_img')):
                os.makedirs(os.path.join(save_img_file, 'save_img'))
            if not os.path.exists(os.path.join(save_img_file, 'save_anno')):
                os.makedirs(os.path.join(save_img_file, 'save_anno'))
            # path_num=back_img_path.split('/')[-1]
            # syn_numbers=path_num.split('.')[0]
            # save_img_file1 = os.path.join(save_img_file, 'save_img/{}_none_{}.jpg'.format(syn_number,syn_numbers))
            # save_anno_file = os.path.join(save_img_file, 'save_anno/{}_none_{}.json'.format(syn_number,syn_numbers))

            save_img_file1 = os.path.join(save_img_file, 'save_img/{}_MUL_none.jpg'.format(syn_number))
            save_anno_file = os.path.join(save_img_file, 'save_anno/{}_MUL_none.json'.format(syn_number))

            back_label_data['shapes'] = objects_inf

            last_img = create_image_anno(pre_obj_mask, back_img, save_img_file1, back_label_data, save_anno_file,
                                         BLENDING_LIST)
            syn_number = syn_number + 1
            # if syn_number == NUM:SYN_IMAGE_NUM
            if syn_number == SYN_IMAGE_NUM :
                print('generated data  finished')
                print('/*******end***********/')
                flag=False
                return 1
            else:
                flag = True
