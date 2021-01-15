import cv2 as cv
import numpy as np
# from get_lines import *
# from square_check import *
# from image_transform import *
from model.get_surface import *
from model.surface_deal import *

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 100, 0)]

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

def get_construction_result(file,save_file):
    img_files = get_list_of_images(file)
    for file_path in img_files:
        img = cv2.imread(file_path)
        print('file  path is :', file_path)
        h, w, c = img.shape
        anno_path = get_list_of_labels(file_path)
        print('json path is :', anno_path)
        json_data = json.load(open(anno_path))

        all_data = json_data['shapes']
        k = 0
        for data in all_data:
            label = data['label']
            flag = label.split('-')[-1]
            points = data['points']

            print('-----------calculate surface START---------')
            points = calc_surface(points)
            state_flage = False
            for num_surface in range(len(points)):
                if len(points[num_surface]) < 4:
                    state_flage = True
                    break
            if state_flage:
                continue
            print('-----------construct square  END---------')
            # surface=get_square(points,flag)
            hello = construction_surface(points, flag)
            surface = hello.surface
            # print('surface is :',surface)
            print('')
            if surface == -1:
                print('error of surface :')
                continue
            j = 0
            for point_list in surface:
                for i in range(len(point_list)):
                    if point_list[i][0] < 0:
                        point_list[i][0] = 0
                    if point_list[i][1] < 0:
                        point_list[i][1] = 0
                    point_list[i][0] = int(point_list[i][0])
                    point_list[i][1] = int(point_list[i][1])
                    if point_list[i][0] > w:
                        point_list[i][0] = w
                    if point_list[i][1] > h:
                        point_list[i][1] = h
                img = polygons_to_mask_(img, point_list, colors[j])
                j = j + 1
        file_name = file_path.split('/')[-1]
        print('file name is :',file_name)
        img_save=os.path.join(save_file)
        if not os.path.exists(img_save):
            os.makedirs(img_save)
        img_save = os.path.join(img_save, file_name)
        cv2.imwrite(img_save, img)


if __name__ == "__main__":
    arg=parse_args()
    get_construction_result(arg.background_files,arg.save_files)

# python test_construction_surface.py -bf ./data/back -sf ./data/cons
