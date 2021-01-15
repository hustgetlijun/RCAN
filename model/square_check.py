import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math
# import get_lines as get
from .global_param import *

def cross_point(line1, line2):  # 计算交点函数
    """
    @cross_point 求两条线的交点
    :param line1: vector
    :param line2: vector
    :return: -1 is not cross
        or [x,y] is the cross point
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    if x1==x2:
        k1=None
        x=x1
        if x3 == x4:
            if y1!=y3:
                return -1
            k2 = None
            y = int((y1+y2+y3+y4)/4)
        elif y3 == y4:
            k2 = 0
            b2 = y3 * 1.0 - x3 * k2 * 1.0  # 整型转浮点型是关键
            y = int(k2 * x * 1.0 + b2 * 1.0)
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 计算k1,由于点均为整数，需要进行浮点数转化
            b2 = y3 * 1.0 - x3 * k2 * 1.0  # 整型转浮点型是关键
            y = int(k2 * x * 1.0 + b2 * 1.0)
    elif y1==y2:
        if x3 == x4:
            x=x3
            y=y1
        elif y3 == y4:
            if y1!=y3:
                return -1
            y=y3
            x=int((x1+x2+x3+x4)/4)
        else:
            y=y1
            k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 计算k1,由于点均为整数，需要进行浮点数转化
            b2 = y3 * 1.0 - x3 * k2 * 1.0  # 整型转浮点型是关键
            x=int((y-b2*1.0)/k2)
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
        if x3 == x4:
            x=x3
            y=int(k1 * x * 1.0 + b1 * 1.0)
        elif y3 == y4:
            y=y3
            x = int((y - b1 * 1.0) / k1)
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 计算k1,由于点均为整数，需要进行浮点数转化
            b2 = y3 * 1.0 - x3 * k2 * 1.0  # 整型转浮点型是关键
            if k2==k1:
                return -1
            else:
                x=int((b2-b1)/(k1-k2))
                y=int(k1 * x * 1.0 + b1 * 1.0)
    return [x, y]

def check_all_points_on_side(lines_prarm,points):
    """
    @check_all_points_on_side 判断给定点集是否都在给定直线的一侧
    :param lines: float tuple (A,B,C) :Ax+By+C=0
    :param points: the points list of surface
    :return: Ture :on side ,False: not on side
    """
    A,B,C=lines_prarm
    # print('A,B,C in check all points on side :',(A,B,C))

    n=len(points)
    g_cost=0
    numbersofzero=0
    for point in points:
        x, y = point
        if A*x + B*y + C > 3: #由于计算机计算误差需要设置一个浮动量
            g_cost = g_cost + 1
        elif A*x + B*y + C < -3:
            g_cost = g_cost - 1
        else:
            numbersofzero=numbersofzero+1
            g_cost = g_cost + 0
    if n ==numbersofzero:
        return  False
    # print('check point on side :',abs(g_cost)+numbersofzero)
    if abs(g_cost)+numbersofzero==n:
        return True
    else:
        return False

def distance_point2point(point1,point2):
    x1,y1=point1
    x2,y2=point2
    x=x1-x2
    y=y1-y2
    distance=math.pow(x*x + y*y, 0.5)
    return distance
"""
@ square_check
input: lines :vector<[[x1,y1],[x2,y2]]> (4 lines)
output: point:vector<[x,y]>(4 corner)
        or -1 Error:it is not a square
"""


def area_of_points(point0, point1, point2):
    """
    求三角形的面积
    :param point0:[X,Y]
    :param point1:
    :param point2:
    :return:
    """
    a = distance_point2point(point0, point1)
    b = distance_point2point(point0, point2)
    c = distance_point2point(point1, point2)
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area

def square_check(lines):
    """
    :param lines:
    :return:
    """
    points_list=[]
    # intersection_points=[]
    Distance=[]
    for line in lines:
        x1,y1,x2,y2=line[0]
        points_list.append([x1,y1])
        points_list.append([x2,y2])
        Distance.append(distance_point2point([x1,y1],[x2,y2]))
    dis_limit=min(Distance)
    for line in lines:
        for line_next in lines:
            if np.all(line==line_next):
                continue
            else:
                x1, y1, x2, y2 = line[0]
                x3,y3,x4,y4=line_next[0]
                dis1=distance_point2point((x1,y1),(x3,y3))
                dis2=distance_point2point((x1, y1), (x4, y4))
                dis3=distance_point2point((x2,y2),(x3,y3))
                dis4=distance_point2point((x2, y2), (x4, y4))
                dis=min([dis1,dis2,dis3,dis4])

                if dis < dis_limit:
                   point=cross_point(line,line_next)
                   if dis1==dis:
                       if point !=-1:
                           flag = 0
                           if [x1,y1] in points_list and [x3,y3] in points_list:
                               flag=1
                           if flag ==1:
                               points_list.remove([x1, y1])
                               points_list.remove([x3, y3])
                               points_list.append(point)
                   elif dis2==dis:
                       if point !=-1:
                           flag = 0
                           if [x1,y1] in points_list and [x4,y4] in points_list:
                               flag=1
                           if flag == 1:
                               points_list.remove([x1, y1])
                               points_list.remove([x4, y4])
                               points_list.append(point)
                   elif dis3==dis:
                       if point !=-1:
                           if point != -1:
                               flag = 0
                               if [x2, y2] in points_list and [x3, y3] in points_list:
                                   flag = 1
                               if flag == 1:
                                   points_list.remove([x2, y2])
                                   points_list.remove([x3, y3])
                                   points_list.append(point)
                   elif dis4==dis:
                       if point !=-1:
                           flag = 0
                           if [x2,y2] in points_list and [x4,y4] in points_list:
                               flag=1
                           if flag == 1:
                               print('[x4,y4] is :',[x4,y4])
                               print('points list is ',points_list)
                               points_list.remove([x2, y2])
                               points_list.remove([x4, y4])
                               points_list.append(point)
    if len(points_list)==4:
        points_list=get.point_sequence(points_list)
        return points_list
    else:
        print('Error:it is not a square')
        return -1
