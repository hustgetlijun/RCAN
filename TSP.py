import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math
import json
import glob


class TSP:
    """
    X:cost matrxi(m,m)
    start_node: the id of the start point in points
    """
    def __init__(self,X,start_node):
        self.X = X #距离矩阵(m,m)
        self.start_node = start_node #开始的节点
        self.array = [[0]*(2**(len(self.X)-1)) for i in range(len(self.X))]#记录处于x节点，未经历M个节点时，矩阵储存x的下一步是M中哪个节点(m*2^(m-1))

    def transfer(self, sets):
        su = 0
        for s in sets:
            if s< self.start_node:
                su=su + 2**(s)
            else:
                su = su + 2**(s-1) # 二进制转换
        return su
    # tsp总接口
    def tsp(self):
        s = self.start_node
        num = len(self.X)
        cities = list(range(num)) #形成节点的集合
        # past_sets = [s] #已遍历节点集合
        cities.pop(cities.index(s)) #构建未经历节点的集合
        node = s #初始节点
        return self.solve(node, cities) #求解函数

    def solve(self, node, future_sets):
        # 迭代终止条件，表示没有了未遍历节点，直接连接当前节点和起点即可
        if len(future_sets) == 0 or (self.X[node][self.start_node]==1):
            c = self.transfer(future_sets)
            # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
            self.array[node][c] = self.start_node
            return self.X[node][self.start_node]
        d = 999
        # node如果经过future_sets中节点，最后回到原点的距离
        distance = []
        # 遍历未经历的节点
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            if self.X[node][s_i]==255  :
                distance.append(self.X[node][s_i]+255)
                continue
            copy = future_sets[:]
            copy.pop(i) # 删除第i个节点，认为已经完成对其的访问
            distance.append(self.X[node][s_i] + self.solve(s_i,copy))
        # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 未遍历节点集合
        c = self.transfer(future_sets)
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.array[node][c] = next_one
        return d
# D=[[255,1,255,255,255,255,255,255],
#    [255,255,1,1,255,255,255,255],
#    [1,1,255,255,255,255,255,255],
#    [255,255,255,255,1,255,255,255],
#    [255,255,255,255,255,1,255,1],
#    [255,255,1,255,1,255,255,255],
#    [255,255,255,255,255,255,255,1],
#    [255,255,255,255,255,1,255,255]]
#
# S = TSP(D, 3)
# min_dis=S.tsp()
# # 开始回溯
# M = S.array
# lists = list(range(len(S.X)))
# # print('lists is :',lists)
# start = S.start_node
# a=[start]
# while len(lists) > 0:
#     lists.pop(lists.index(start))
#     m = S.transfer(lists)
#     next_node = S.array[start][m]
#     a.append(next_node)
#     print(start, "--->", next_node)
#     start = next_node
#     if next_node==S.start_node:
#         break
# print(a)





