import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math
import copy

#import get_lines as get
from .global_param import *
from .square_check import *
class construction_surface:
    def __init__(self, point_list,label):
        self.p_list = copy.copy(point_list)
        self.surface=self.get_square(self.p_list,label)
    def area_of_points(self,point0, point1, point2):
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

    def distance_of_point2line(self,point, line):
        A, B, C = line
        x, y = point
        distance = abs(A * x + B * y + C) / (A * A + B * B) ** 0.5
        return distance

    def campare2line(self,param0, param1):
        """
        @campare2line 比较两个直线是否相同
        :param param0: (A,B,C)
        :param param1: (A,B,B)
        :return: True:the same line ,False:different line
        """
        A0, B0, C0 = param0
        A1, B1, C1 = param1
        if B0 == 0:
            if abs(A0 - A1) < 0.1 and abs(C0 - C1) < 20:
                return True
            else:
                return False
        else:
            if abs(A0 / B0 - A1 / B1) < 0.1 and abs(C0 / B0 - C1 / B1) < 20:
                return True
            else:
                return False

    def corss_point_with_lines(self,param0, param1):
        """
        求两条直线的交点
        :param param0: A0x+B0y+C0=0
        :param param1:A1x+B1y+C1 =0
        :return: -1:error or [X,Y]
        """
        A0, B0, C0 = param0
        A1, B1, C1 = param1
        # print('***********************corss point with lines*********************')
        # print('A0,B0,C0 is :',param0)
        # print('A1,B1,C1 is :', param1)
        # print('--------------------------------')
        if B0 != 0 and B1 != 0:
            if A0 / B0 == A1 / B1:
                return -1
        if A0 == 0:
            y = -C0 / B0
            x = B1 * C0 / (B0 * A1) - C1 / A1
        elif A1 == 0:
            y = -C1 / B1
            x = B0 * C1 / (B1 * A0) - C0 / A0
        elif B0 == 0:
            x = -C0 / A0
            y = A1 * C0 / (A0 * B1) - C1 / B1
        elif B1 == 0:
            x = -C1 / A1
            y = A0 * C1 / (A1 * B0) - C0 / B0
        else:
            y = (A0 * C1 - C0 * A1) / (A1 * B0 - B1 * A0)
            x = (C0 * B1 - C1 * B0) / (A1 * B0 - A0 * B1)
        x = round(x, 4)
        y = round(y, 4)
        return [x, y]

    def get_point2line_distance(self,point, line):
        """
        求点到直线的距离
        :param point:
        :param line:
        :return:
        """
        point_x, point_y = point
        line_s_x, line_s_y = line[0]
        line_e_x, line_e_y = line[1]
        if abs(line_e_x - line_s_x) < dis_limit:
            return math.fabs(point_x - line_s_x)
        if abs(line_e_y - line_s_y) < dis_limit:
            return math.fabs(point_y - line_s_y)
        k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
        b = line_s_y - k * line_s_x
        dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
        return dis

    def get_line_with_points(self,points):
        """
        获取直线的参数
        :param points: vector
        :return:(A,B,C)(AX+By+C=0)
        """
        A = 0
        B = 0
        C = 0
        # print('in the get line with points, the points is :', points)

        x1, y1 = points[0]
        x2, y2 = points[1]
        if x1 == x2:
            B = 0
            A = 1
            C = -x1
        elif y1 == y2:
            A = 0
            B = 1
            C = -y1
        else:
            A = (y1 - y2) / (x1 - x2)
            B = -1
            C = y1 - A * x1
        A = round(A, 4)
        C = round(C, 4)
        return [A, B, C]

    def get_praram_one_points(self,params, point):
        """
         给定参考直线求与他平行的直线
        :param params: 输入参考直线（AX+BY+C=0）
        :param point:直线上一点 Point
        :return: 斜率为参考直线,且过点point的直线（AX+BY+C0=0）
        """
        x, y = point
        A, B, C = params
        A1 = 0
        B1 = 0
        C1 = 0
        if A == 0:
            A1 = 0
            B1 = -1
            C1 = y
        elif B == 0:
            A1 = 1
            B1 = 0
            C1 = -x
        else:
            A1 = A
            B1 = B
            C1 = -(A * x + B * y)
        A1 = round(A1, 4)
        C1 = round(C1, 4)
        return [A1, B1, C1]

    def get_full_common_edg_two(self,points, surfaces_datas):
        """
        两个面之间补全公共边的长度
        :param surfaces_data:[{'face':[],'comm_edg':[[line]]},{...}]
        :return: surfaces_data
        """
        # print('************************get_full_common_edg_two()***********')
        # print('surfaces datas is :',surfaces_datas)

        surfaces_data = copy.copy(surfaces_datas)

        common_line0 = copy.copy(surfaces_data[0]['comm_edg'][0])
        common_line1 = copy.copy(surfaces_data[1]['comm_edg'][0])
        param = self.get_line_with_points(common_line0)
        A, B, C = param
        if common_line0 == common_line1:
            return surfaces_data
        else:
            # print('common line is : ',(common_line0,common_line1))
            line_new = np.concatenate((common_line0, common_line1), axis=0)
            # print('line new 22222 is :',line_new)
            line_new = line_new.transpose()

            # print('33333333333 is ', line_new)
            xmin = min(line_new[0])
            xmax = max(line_new[0])
            ymin = min(line_new[1])
            ymax = max(line_new[1])
            line_new = line_new.tolist()
            # point0=[]
            # point1=[]
            if B == 0:  # 4个点组成的线段确定其端点，作为最终的公共线段
                index_min = line_new[1].index(ymin)
                index_max = line_new[1].index(ymax)
                point0 = [line_new[0][index_min], ymin]
                point1 = [line_new[0][index_max], ymax]
            else:
                index_min = line_new[0].index(xmin)
                index_max = line_new[0].index(xmax)
                point0 = [xmin, line_new[1][index_min]]
                point1 = [xmax, line_new[1][index_max]]
            # print('point0  and point1 is :',(point0,point1))

            for i in range(2):  # 每个面根据最终公共线段的长度调整面的大小
                surface = surfaces_data[i]['face']

                common_line = surfaces_data[i]['comm_edg'][0]
                param = self.get_line_with_points(common_line)
                A, B, C = param
                if B == 0:  # 为了使端点与上一步求得全局端点对应（大对大，小对小）
                    if common_line[0][1] < common_line[1][1]:
                        point_0 = [common_line[0][0], common_line[0][1]]  # 存储下端点
                        point_1 = [common_line[1][0], common_line[1][1]]
                    else:
                        point_0 = [common_line[1][0], common_line[1][1]]
                        point_1 = [common_line[0][0], common_line[0][1]]

                else:
                    if common_line[0][0] < common_line[1][0]:
                        point_0 = [common_line[0][0], common_line[0][1]]  # 存储左端点
                        point_1 = [common_line[1][0], common_line[1][1]]
                    else:
                        point_0 = [common_line[1][0], common_line[1][1]]
                        point_1 = [common_line[0][0], common_line[0][1]]

                # print('point_0  and point_1 is :', (point_0, point_1))
                index_0 = surface.index(point_0)
                index_1 = surface.index(point_1)

                # print('index_0,index_1 is :',(index_0,index_1))
                line_to_edg = []

                for j in range(4):
                    if surface[j] in [point_0, point_1]:
                        continue
                    else:
                        line_to_edg.append(surface[j])
                param_to_edg = self.get_line_with_points(line_to_edg)
                # print(' to edg is ___________________is :',line_to_edg)
                if point0 not in [point_0]:  # 确定非公共边点
                    if index_0 < index_1:
                        if index_0 == 0 and index_1 == 3:
                            line = [surface[1], surface[index_0]]
                            p_next_0 = surface[1]
                        else:
                            line = [surface[index_0 - 1], surface[index_0]]
                            p_next_0 = surface[index_0 - 1]
                    else:
                        if index_0 == len(surface) - 1:
                            if index_1 == 0:
                                line = [surface[index_0], surface[2]]
                                p_next_0 = surface[2]
                            else:
                                line = [surface[index_0], surface[0]]
                                p_next_0 = surface[0]
                        else:
                            line = [surface[index_0], surface[index_0 + 1]]
                            p_next_0 = surface[index_0 + 1]
                    index_next_0 = surface.index(p_next_0)
                    param0 = self.get_line_with_points(line)
                    param0 = self.get_praram_one_points(param0, point0)
                    surfaces_data[i]['face'][index_next_0] = self.corss_point_with_lines(param0, param_to_edg)
                    surfaces_data[i]['face'][index_0] = point0
                if point1 not in [point_1]:
                    if index_1 < index_0:
                        if index_1 == 0 and index_0 == 3:
                            line = [surface[0], surface[1]]
                            p_next_1 = surface[1]
                        else:
                            line = [surface[index_1 - 1], surface[index_1]]
                            p_next_1 = surface[index_1 - 1]
                    else:
                        if index_1 == len(surface) - 1:
                            if index_0 == 0:
                                line = [surface[2], surface[index_1]]
                                p_next_1 = surface[2]
                            else:
                                line = [surface[index_1], surface[0]]
                                p_next_1 = surface[0]
                        else:
                            line = [surface[index_1], surface[index_1 + 1]]
                            p_next_1 = surface[index_1 + 1]
                    index_next_1 = surface.index(p_next_1)
                    param0 = self.get_line_with_points(line)
                    param1_0 = self.get_praram_one_points(param0, point1)
                    surfaces_data[i]['face'][index_next_1] = self.corss_point_with_lines(param1_0, param_to_edg)
                    # print('points is  findal is   2 :', points)
                    surfaces_data[i]['face'][index_1] = point1
                    # print('points is  findal is   3 :', points)
            return surfaces_data

    def get_full_common_edg_three(self,surfaces_data):
        """
        三个面时公共边补全
        :param surfaces_data:
        :return:
        """
        # print('************************get_full_common_edg_three()***********')
        # print('surface data is Three :',surfaces_data)
        surface_datas = []
        traverse = []
        for i in range(3):  # 计算两个面之间的公共边
            traverse.append(i)
            for k in range(3):
                if k in traverse:
                    continue
                else:
                    surface0 = copy.copy(surfaces_data[i]['face'])
                    line0_0 = copy.copy(surfaces_data[i]['comm_edg'][0])
                    line0_1 = copy.copy(surfaces_data[i]['comm_edg'][1])
                    param0_0 = self.get_line_with_points(line0_0)
                    param0_1 = self.get_line_with_points(line0_1)

                    surface1 = copy.copy(surfaces_data[k]['face'])
                    line1_0 = copy.copy(surfaces_data[k]['comm_edg'][0])
                    line1_1 = copy.copy(surfaces_data[k]['comm_edg'][1])
                    Line = [line1_0, line1_1]
                    param1_0 = self.get_line_with_points(line1_0)
                    param1_1 = self.get_line_with_points(line1_1)
                    para = [param1_0, param1_1]
                    # print(' param is :',para)
                    """
                    注意斜率计算时由于计算机的计算误差导致相同的直线有一个误差
                    """
                    commLine0 = []
                    commLine1 = []

                    flag0 = False
                    flag1 = False
                    id0 = 0
                    id1 = 0
                    for j in range(2):
                        flag0 = self.campare2line(param0_0, para[j])
                        if flag0 is True:
                            id0 = j
                            break
                    for j in range(2):
                        flag1 = self.campare2line(param0_1, para[j])
                        if flag1 is True:
                            id1 = j
                            break

                    if flag0:  # 通过斜率判断是否为同一条直线
                        commLine0 = [line0_0]
                        commLine1 = [Line[id0]]
                    if flag1:
                        commLine0 = [line0_1]
                        commLine1 = [Line[id1]]
                    sur_data = [{'face': surface0, 'comm_edg': commLine0}, {'face': surface1, 'comm_edg': commLine1}]
                    # print('sur data is 11111111111:',sur_data)
                    # surface_datas.append(get_full_common_edg_two(sur_data))
                    surface_datas.append(sur_data)
        """
        获取补全后的公共边
        """
        # print('获取补全后的公共边 is :',surface_datas)
        comm_line0_0 = surface_datas[0][0]['comm_edg'][0]  # 1-2`
        comm_line0_1 = surface_datas[0][1]['comm_edg'][0]  # 1-2

        comm_line1_0 = surface_datas[1][0]['comm_edg'][0]  # 1-3
        comm_line1_1 = surface_datas[1][1]['comm_edg'][0]  # 1-3

        comm_line2_0 = surface_datas[2][0]['comm_edg'][0]  # 2-3
        comm_line2_1 = surface_datas[2][1]['comm_edg'][0]  # 2-3
        line_com = [[comm_line0_0, comm_line0_1], [comm_line1_0, comm_line1_1], [comm_line2_0, comm_line2_1]]
        comm_line_ = []
        for i in range(3):

            A, B, C = self.get_line_with_points(line_com[i][0])
            line_new = np.concatenate(line_com[i], axis=0)
            # print('line new 22222 is :',line_new)
            line_new = line_new.transpose()
            # print('33333333333 is ', line_new)
            xmin = min(line_new[0])
            xmax = max(line_new[0])
            ymin = min(line_new[1])
            ymax = max(line_new[1])
            line_new = line_new.tolist()
            # point0=[]
            # point1=[]
            if B == 0:  # 4个点组成的线段确定其端点，作为最终的公共线段
                index_min = line_new[1].index(ymin)
                index_max = line_new[1].index(ymax)
                point0 = [line_new[0][index_min], ymin]
                point1 = [line_new[0][index_max], ymax]
            else:
                index_min = line_new[0].index(xmin)
                index_max = line_new[0].index(xmax)
                point0 = [xmin, line_new[1][index_min]]
                point1 = [xmax, line_new[1][index_max]]
            comm_line_.append([point0, point1])
        com_line = [[comm_line_[0], comm_line_[1]], [comm_line_[0], comm_line_[2]], [comm_line_[1], comm_line_[2]]]
        for i in range(3):  # 计算新的完整的面轮廓
            line0 = com_line[i][0]
            line1 = com_line[i][1]
            com_point = []
            other_point_0 = []
            other_point_1 = []
            for point in line0:
                if point in line1:
                    com_point = point

            for point in line0:
                if point == com_point:
                    continue
                else:
                    other_point_0 = point
            for point in line1:
                if point == com_point:
                    continue
                else:
                    other_point_1 = point

            param0 = self.get_line_with_points(line0)
            param1 = self.get_line_with_points(line1)

            Line_new0 = self.get_praram_one_points(param0, other_point_1)
            Line_new1 = self.get_praram_one_points(param1, other_point_0)
            opposing_point = self.corss_point_with_lines(Line_new0, Line_new1)
            surface = {'face': [com_point, other_point_1, opposing_point, other_point_0], 'comm_edg': [line0, line1]}
            surface_datas[i] = surface
        return surface_datas

    def Calculate_common_edg(self,surfaces_points):
        """
        @Calculate_common_edg 计算公共边 [[p1,p2,p3,...],[],[]]
        :param surfaces_points:[[surface],[surface],[...]]
        :return: -1 error or result:[{'face':[],'comm_edg':[{...},{...}]
        """
        number_face = len(surfaces_points)
        result = []
        face_edg = {'face': [], 'comm_edg': []}

        # print('**************calculate common edg  START************ ')
        # print('the number of face is :',number_face)
        # # print('the INPUNT of surfaces is :', surfaces_points)
        # print('**************calculate common edg END************ ')

        if number_face == 1:
            face_edg['face'] = surfaces_points[0]
            face_edg['comm_edg'] = []
            result.append(face_edg)
            return result
        elif number_face == 2:
            surface0 = surfaces_points[0]
            surface1 = surfaces_points[1]
            edg_point = []
            for j in range(len(surface0)):
                if surface0[j] in surface1:
                    edg_point.append(surface0[j])
            if len(edg_point) < 2:
                return -1
            result.append({'face': surface0, 'comm_edg': [edg_point]})
            result.append({'face': surface1, 'comm_edg': [edg_point]})

        elif number_face == 3:
            for i in range(number_face):
                surface = surfaces_points[i]
                edg = []
                for k in range(number_face):
                    edg_point = []
                    if i == k:
                        continue
                    else:
                        for m in range(len(surface)):
                            if surface[m] in surfaces_points[k]:
                                edg_point.append(surface[m])
                        if len(edg_point) < 2 :
                            continue
                        edg.append(edg_point)

                if len(edg) !=2:
                    # print('hello------------------',edg)
                    return -1
                result.append({'face': surface, 'comm_edg': edg})
        else:
            print('the common edg is : -1')
            return -1
        # print('in the caculate common edg function   the common edg is : ',result)
        # for num_com in range(len(result)):
        #     for num_c in range(len(result[num_com]['comm_edg'])):
        #        if len(result[num_com]['comm_edg'][num_c]) < 2:
        #           return -1
        return result

    def construct_square(self,common_line, surface):
        """
        对于存在三个面的实例只构造三条公共边都存在的情况。
        @construct_square：输入选定面的公共直线和面 构造完整四边形
        :param common_points: [[line1],[line2],...] ，line1=[[x1,y1],[x2,y2]]
        :param surface: points:[[x,y],[],[],...]
        :return: min_square_pointsdierci and new common line 输出新构造的面和新的公共线 {'face':[],'comm_edg':[[],[]]}
        """
        numberofpoints = len(surface)
        surfaces = []
        # surface_0=surface
        # print('suface is :', surface)
        # surface_0.append(surface[0])
        # print('surface_0 is :',surface_0)

        common_line_new = []
        if len(common_line) == 0:
            # str=" one surface should be constructed"
            # print(str)
            area = 0
            for i in range(numberofpoints):
                if i == numberofpoints - 1:
                    line0 = [surface[i - 1], surface[i]]
                    line1 = [surface[i], surface[0]]
                else:
                    line0 = [surface[i - 1], surface[i]]
                    line1 = [surface[i], surface[i + 1]]
                param0 = self.get_line_with_points(line0)
                param1 = self.get_line_with_points(line1)
                # print('param0 is :', param0)
                # print('param1 is :', param1)
                flag0 = check_all_points_on_side(param0, surface)
                flag1 = check_all_points_on_side(param1, surface)
                if flag0 == True and flag1 == True:
                    # print('flag is true')
                    corss_point_0 = []
                    corss_point_1 = []
                    corss_point_3 = copy.copy(surface[i])
                    p0 = []
                    p1 = []
                    for j in range(numberofpoints):
                        if surface[j] not in line0:
                            param_0 = self.get_praram_one_points(param0, surface[j])
                            flag_0 = check_all_points_on_side(param_0, surface)
                            if flag_0 == True:
                                corss_point_0 = self.corss_point_with_lines(param_0, param1)
                                p0 = param_0
                                for k in range(numberofpoints):
                                    if surface[k] not in line1:
                                        param_1 = self.get_praram_one_points(param1, surface[k])
                                        flag_1 = check_all_points_on_side(param_1, surface)
                                        if flag_1 == True:
                                            p1 = param_1
                                            corss_point_1 = self.corss_point_with_lines(param_1, param0)
                    corss_point_2 = self.corss_point_with_lines(p0, p1)
                    area0 = self.area_of_points(corss_point_3, corss_point_0, corss_point_2)
                    area1 = self.area_of_points(corss_point_3, corss_point_1, corss_point_2)
                    if area == 0:
                        area = area0 + area1
                        surfaces = [corss_point_3, corss_point_0, corss_point_2, corss_point_1]
                    elif (area0 + area1) < area:
                        area = area0 + area1
                        surfaces = [corss_point_3, corss_point_0, corss_point_2, corss_point_1]
        elif len(common_line) == 1:
            # str=" two surface should be constructed"
            # print(str)
            area = -1
            point0 = common_line[0][0]
            point1 = common_line[0][1]
            index0 = surface.index(point0)
            index1 = surface.index(point1)
            if index0 < index1:
                # point2=surface[index0-1]
                if index1 == numberofpoints - 1 and index0 == 0:
                    point3 = copy.copy(surface[numberofpoints - 2])
                else:
                    if index1 == numberofpoints - 1:
                        point3 = copy.copy(surface[0])
                    else:
                        point3 = copy.copy(surface[index1 + 1])
                if index0 == 0 and index1 == len(surface) - 1:
                    point2 = copy.copy(surface[1])
                else:
                    point2 = copy.copy(surface[index0 - 1])

                line0 = [point2, point0]
                line1 = [point0, point1]
                line2 = [point1, point3]
            else:
                if index0 == numberofpoints - 1 and index1 == 0:
                    point2 = copy.copy(surface[numberofpoints - 2])
                    point3 = copy.copy(surface[1])
                else:
                    if index0 == numberofpoints - 1:
                        point2 = copy.copy(surface[0])
                    else:
                        point2 = copy.copy(surface[index0 + 1])
                    point3 = copy.copy(surface[index1 - 1])
                line0 = [point3, point1]
                line1 = [point1, point0]
                line2 = [point0, point2]

            # print('111111111111 line0,line1,line2 is :',(line0,line1,line2))
            param0 = self.get_line_with_points(line0)
            param1 = self.get_line_with_points(line1)
            param2 = self.get_line_with_points(line2)
            flag0 = check_all_points_on_side(param0, surface)
            flag2 = check_all_points_on_side(param2, surface)
            # print('flag0 flag2 is:',(flag0,flag2))

            if flag0 == True:
                corss_point_3 = line1[0]
                p0_0 = []
                p0_1 = []
                corss_point_0 = []
                corss_point_1 = []
                for j in range(numberofpoints):
                    if surface[j] not in line0:
                        param_0 = self.get_praram_one_points(param0, surface[j])
                        flag_0 = check_all_points_on_side(param_0, surface)
                        if flag_0 == True:
                            corss_point_0 = self.corss_point_with_lines(param_0, param1)
                            p0_0 = param_0
                    if surface[j] not in line1:
                        param_1 = self.get_praram_one_points(param1, surface[j])
                        flag_1 = check_all_points_on_side(param_1, surface)
                        if flag_1 == True:
                            corss_point_1 = self.corss_point_with_lines(param_1, param0)
                            p0_1 = param_1
                corss_point_2 = self.corss_point_with_lines(p0_0, p0_1)
                area0 = self.area_of_points(corss_point_3, corss_point_0, corss_point_2)
                area1 = self.area_of_points(corss_point_3, corss_point_1, corss_point_2)
                area = area0 + area1
                surfaces = [corss_point_3, corss_point_0, corss_point_2, corss_point_1]
                # print('surface number of 1 is :',surfaces)
                # print('the area of 1 is :',area)
                common_line_new.append([corss_point_3, corss_point_0])
            if flag2 == True:
                corss_point_3 = line1[1]
                param_1 = []
                param_2 = []
                p1_1 = []
                p1_2 = []
                corss_point_0 = []
                corss_point_1 = []
                j = 0
                for j in range(numberofpoints):
                    if surface[j] not in line1:
                        param_1 = self.get_praram_one_points(param1, surface[j])
                        flag_1 = check_all_points_on_side(param_1, surface)
                        if flag_1 == True:
                            corss_point_0 = self.corss_point_with_lines(param_1, param2)
                            p1_1 = param_1
                    if surface[j] not in line2:
                        param_2 = self.get_praram_one_points(param2, surface[j])
                        flag_2 = check_all_points_on_side(param_2, surface)
                        if flag_2 == True:
                            corss_point_1 = self.corss_point_with_lines(param_2, param1)
                            p1_2 = param_2
                corss_point_2 = self.corss_point_with_lines(p1_1, p1_2)
                area0 = self.area_of_points(corss_point_3, corss_point_0, corss_point_2)
                area1 = self.area_of_points(corss_point_3, corss_point_1, corss_point_2)
                if (area0 + area1) < area or area == -1:
                    surfaces = [corss_point_3, corss_point_0, corss_point_2, corss_point_1]
                    common_line_new.clear()
                    common_line_new.append([corss_point_3, corss_point_1])
                    # print('surface number of 2 is :',surfaces)
                    # print('the area of 2 is :', area0+area1)
        elif len(common_line) == 2:
            # str=" three surface should be constructed"
            # print(str)
            print('common edg is :', common_line)
            line0 = common_line[0]
            line1 = common_line[1]
            # print('line0 and line1 is :',(line0,line1))
            param0 = self.get_line_with_points(line0)
            param1 = self.get_line_with_points(line1)
            # print('param0 is :',param0)
            # print('param1 is :',param1)
            common_point = []
            for i in range(len(line0)):
                if line0[i] in line1:
                    common_point = line0[i]
            corss_point_3 = common_point
            param1_1 = 0
            param0_0 = 0
            corss_point_0 = []
            corss_point_1 = []
            for j in range(numberofpoints):
                if surface[j] not in line0:
                    if self.get_point2line_distance(surface[j], line0) < 10:
                        # print('..............................................')
                        continue
                    param_0 = self.get_praram_one_points(param0, surface[j])
                    flag_0 = check_all_points_on_side(param_0, surface)
                    if flag_0 == True:
                        corss_point_0 = self.corss_point_with_lines(param_0, param1)
                        param0_0 = param_0
                if surface[j] not in line1:
                    if self.get_point2line_distance(surface[j], line1) < 10:
                        # print('..............................................')
                        continue
                    param_1 = self.get_praram_one_points(param1, surface[j])
                    flag_1 = check_all_points_on_side(param_1, surface)
                    if flag_1 == True:
                        corss_point_1 = self.corss_point_with_lines(param_1, param0)
                        param1_1 = param_1
            corss_point_2 = self.corss_point_with_lines(param0_0, param1_1)
            surfaces = [corss_point_3, corss_point_0, corss_point_2, corss_point_1]
            common_line_new.append([corss_point_3, corss_point_0])
            common_line_new.append([corss_point_3, corss_point_1])
        return ({'face': surfaces, 'comm_edg': common_line_new})

    def get_square(self,points, label):
        """
        :param points: surface points:[[[x,y],[x,y],...],...]
        :param label: all or occlusion
        :return: surface points:[[[x,y],[x,y],[x,y],[x,y]],...]
        """
        # print('points iput is :', points)
        # points = copy.copy(points_)
        # points_.clear()
        number = len(points)
        surface = []
        sur = []
        if number == 1:
            if label == 'all':
                surface.append(points[0])
            else:
                comm_s = self.Calculate_common_edg(points)
                if comm_s == -1 or len(comm_s) == 0:
                    return -1
                comm_line = comm_s[0]['comm_edg']
                pointlist = comm_s[0]['face']
                sur = self.construct_square(comm_line, pointlist)

                """通过面积筛选构造的四边形"""
                face_0 = sur['face']
                face_1 = comm_s[0]['face']
                if len(face_0) !=4 :
                    return -1
                area_0 = self.area_of_points(face_0[0], face_0[2], face_0[3]) + self.area_of_points(face_0[0], face_0[2],
                                                                                          face_0[1])
                if len(face_1) == 4:
                    area0 = self.area_of_points(face_1[0], face_1[2], face_1[3])
                    area1 = self.area_of_points(face_1[0], face_1[2], face_1[1])
                    area = area0 + area1
                    if area_0 * area_ratio < area:
                        sur['face'] = comm_s[0]['face']
                surface.append(sur['face'])
        elif number == 2:
            comm_s = self.Calculate_common_edg(points)
            # print('comm surface edg is :',comm_s)
            if comm_s == -1 or len(comm_s) == 0 :
                return -1
            else:
                sur0 = self.construct_square(comm_s[0]['comm_edg'], comm_s[0]['face'])
                sur1 = self.construct_square(comm_s[1]['comm_edg'], comm_s[1]['face'])
                sur = [sur0, sur1]
                # print('sur is :',sur0)
                for surface_number in range(2):
                    face_0 = sur[surface_number]['face']
                    if len(face_0) ==0 :
                        return -1

                    face_1 = comm_s[surface_number]['face']
                    # if len(face_0) != 4 or len(face_1) != 4:
                    #     return -1
                    area_0 = self.area_of_points(face_0[0], face_0[2], face_0[3]) + self.area_of_points(face_0[0], face_0[2],
                                                                                              face_0[1])
                    # if len(face_1)==4 and label == 'all' :
                    if len(face_1) == 4:
                        area0 = self.area_of_points(face_1[0], face_1[2], face_1[3])
                        area1 = self.area_of_points(face_1[0], face_1[2], face_1[1])
                        area = area0 + area1
                        if area_0 * area_ratio < area:
                            sur[surface_number] = comm_s[surface_number]
                sur = self.get_full_common_edg_two(points, sur)
                for s in sur:
                    surface.append(s['face'])
        elif number == 3:
            # print('points is :',points)
            comm_s = self.Calculate_common_edg(points)
            # print('comm_s is :',comm_s)
            if comm_s == -1 or len(comm_s) == 0:
                return -1
            else:
                for i in range(3):
                    sur = self.construct_square(comm_s[i]['comm_edg'], comm_s[i]['face'])
                    # if len(sur['face']) !=4:
                    #     return -1
                    surface.append(sur)
                # print('surface is :',surface)
                sur0 = self.get_full_common_edg_three(surface)
                for num in range(3):
                    face_0 = sur0[num]['face']
                    # print('surface o is :',sur0[num])
                    dis0_0 = distance_point2point(sur0[num]['comm_edg'][0][0], sur0[num]['comm_edg'][0][1])
                    dis0_1 = distance_point2point(sur0[num]['comm_edg'][1][0], sur0[num]['comm_edg'][1][1])

                    face_1 = comm_s[num]['face']
                    dis1_0 = distance_point2point(comm_s[num]['comm_edg'][0][0], comm_s[num]['comm_edg'][0][1])
                    dis1_1 = distance_point2point(comm_s[num]['comm_edg'][1][0], comm_s[num]['comm_edg'][1][1])
                    # print('surface comm is :', comm_s[num])
                    area_0 = self.area_of_points(face_0[0], face_0[2], face_0[3]) + self.area_of_points(face_0[0],
                                                                                              face_0[2],
                                                                                              face_0[1])

                    if len(face_1) == 4:
                        area0 = self.area_of_points(face_1[0], face_1[2], face_1[3])
                        area1 = self.area_of_points(face_1[0], face_1[2], face_1[1])
                        area = area0 + area1
                        if area_0 * area_ratio < area and abs(dis1_0 - dis0_0) < 5 and abs(dis1_1 - dis0_1) < 5:
                            sur0[num] = comm_s[num]

                surface.clear()
                for s in sur0:
                    surface.append(s['face'])
        if len(surface) == 0:
            return -1
        else:
            return surface













