import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math
import json
import glob

from .global_param import *
from .file_deal import  *
from .square_check import distance_point2point
from .global_param import *
from .TSP import *


def polygons_to_mask_(img, polygons,color):
    frame_ID_list_np = np.array(list(polygons), dtype=int)
    # cv2.polylines(img, [frame_ID_list_np], True, (0, 0, 255))
    # cv2.fillPoly(img, [frame_ID_list_np], color)
    cv2.polylines(img, [frame_ID_list_np], True, color,thickness=8)
    return img

def find_shortest_ring(start_point ,points,cost_matrix):
    """

    :param start_point: 指定寻找面的不重复开始点
    :param points:点序列
    :param cost_matrix:每个点之间的代价矩阵
    :return:最小的能构成回路的点的集合（面）
    """
    start_id=points.index(start_point)
    # print('start id is :',start_id)
    S = TSP(cost_matrix, start_id)
    min_dis=S.tsp()
    M = S.array
    lists = list(range(len(S.X)))
    start = S.start_node
    result=[]
    result.append(start)
    while len(lists) > 0:
        lists.pop(lists.index(start))
        m = S.transfer(lists)
        next_node = S.array[start][m]
        # print(start, "--->", next_node)
        # if next_node in result:
        #     start = next_node
        #     continue
        start = next_node
        if next_node==S.start_node:
            break
        result.append(next_node)
    return result

def calc_surface(point_lists):
    """
    :param point_lists:输入掩码点列表，list
    :return:返回每个面的点集 [[surface],[surface],[]]
    """
    list_num=len(point_lists)
    for i in range(list_num):
        point_lists[i][0]=round(point_lists[i][0],4)
        point_lists[i][1] = round(point_lists[i][1], 4)
    ring_points=point_lists[:]
    ring_points.append(point_lists[0])
    # print('ring points is ',ring_points)
    ids=[]
    for i in range(list_num+1): #为每个坐标寻找相似的兄弟
        point_0=ring_points[i]
        # print('point0 is :',point_0)
        cyc_num=[]

        for j in range(list_num+1):
            if i==j:
                cyc_num.append(i)
                continue
            point_1=ring_points[j]
            dis=distance_point2point(point_0,point_1)
            if dis < dis_surface_limit:
                cyc_num.append(j)
        cyc_num.sort()
        if cyc_num in ids:
            continue
        ids.append(cyc_num)#去重后数据
    point_list_new=[] #存储去重后的点坐标
    for i in range(len(ids)):
        # print('id is ',ids[i])
        if len(ids[i])>1:
            # print('id point is ',ring_points[ids[i][0]])
            x=0
            y=0
            num=0
            for id in ids[i]:
                x=x+ring_points[id][0]
                y=y+ring_points[id][1]
                num=num+1
            x=round(x/num,4)
            y=round(y/num,4)
            for id in ids[i]:
                ring_points[id][0] = x
                ring_points[id][1] = y
            point_list_new.append(ring_points[ids[i][0]])
        else:
            # print('id point is ',ring_points[ids[i][0]])
            point_list_new.append(ring_points[ids[i][0]])
            continue
    # print('point_ list _new is :',point_list_new)
    n=len(point_list_new)
    cost_matrix=np.full((n,n),255) #255表示初始状态两个点无法相连，1表示相连
    for k in range(n): #权值矩阵赋值
        for v in range(n):
            if k==v:
                continue
            else:
                for j in ids[k]:
                    if j+1 in ids[v]:
                        cost_matrix[k][v]=1
    # print('cost matrix is :',cost_matrix)
    ids[0].remove(max(ids[0]))
    single_id=[]
    surfaces_point = []
    surfaces_m=[] #中间变量，判断新出现的目标是否在列表内
    for i in range(len(ids)):
        if len(ids[i])==1:
            start_point = point_list_new[i]
            # print('start point is :',start_point,i)
            surface = find_shortest_ring(start_point, point_list_new, cost_matrix)
            # print('surface is :',surface)
            surface1=surface[:]
            surface.sort()
            if surface in surfaces_m:
                continue
            point_m=[]
            for i in surface1:
                # point_list_new[i][0]=round(point_list_new[i][0],2)
                # point_list_new[i][1] = round(point_list_new[i][1], 2)
                point_m.append(point_list_new[i])
            surfaces_point.append(point_m)
            surfaces_m.append(surface)
            single_id.append(i)
        else:
            continue
    return surfaces_point


# if __name__ == "__main__":
#     # arg=parse_args()
#     # gen_syn_data(arg.foreground_files,arg.background_files,arg.save_files,arg.img_type)
#     file='./data/example/'
#     colors=[(255,0,0),(0,255,0),(0,0,255)]
#     img_files=get_list_of_images(file)
#     for file_path in img_files:
#         img= cv2.imread(file_path)
#         anno_path=get_list_of_labels(file_path)
#         json_data= json.load(open(anno_path))
#         all_data=json_data['shapes']
#
#         for data in all_data:
#             points=data['points']
#             points = calc_surface(points)
#             i=0
#             for point_list in points:
#
#                 img=polygons_to_mask_(img,point_list,colors[i])
#                 i=i+1
#         img_save =file_path.replace('img', 'save')
#         cv2.imwrite(img_save, img)