
import cv2
import sys
import os
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt


def preprocess(img):
    new_img = cv2.medianBlur(img, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    new_img = cv2.morphologyEx(new_img, cv2.MORPH_TOPHAT, kernel) # 同上
    new_img = cv2.convertScaleAbs(new_img,alpha=8,beta=0)
    return new_img


def sharpen_mask(img, n, a):
    #  n is the window para of blur, odd
    img_blur = cv2.medianBlur(img, 5)
    img_mask = img - img_blur
    #     print(img.max(), img_mask.max())
    img_o = img + img_mask
    img_o[img_o < 50] = 200
    print(img_o.shape)


def overlap(point1,point2):
    x1, y1, w1, h1 = point1
    x2, y2, w2, h2 = point2
    endx = max(x1+w1,x2+w2)
    startx = min(x1,x2)
    w = w1+w2-(endx-startx)
    endy = max(y1+h1,y2+h2)
    starty = min(y1,y2)
    h = h1+h2-(endy-starty)
    if w<=0 or h<=0:
        return False
    else:
        area = w*h  # 重叠面积
        area1 = w1*h1   # 分裂前面积
        area2 = w2*h2   # 分裂后面积
        if area1>2*area2:
            return True
        else:
            return False
        # ratio = area/(area1+area2-area)
        # if ratio>=0.5:
        #     return True
        # else:
        #     return False


def draw_bounding_box(dataset, sequence):
    """
    dataset: the name of dataset
    sequence: sequence number
    """
    # Data Path
    TEST_PATH = '{}/Sequence {}/'.format(dataset, sequence)
    test_ids = next(os.walk(TEST_PATH))[2]
    test_ids.sort()
    RES_PATH = "{}/Sequence {} mask/".format(dataset, sequence)
    if not os.path.exists(RES_PATH):
        os.mkdir(RES_PATH)

    bound_box = []
    reserve_contours = []
    # get all bound box locations, begin to detect division
    divide_cell = {}  # 存分裂细胞坐标

    for i in range(0, len(test_ids)):
        id = test_ids[i]
        img = cv2.imread(TEST_PATH + id)
        new_img = preprocess(img)
        bound_box.append([])
        ret, thresh = cv2.threshold(new_img, 190, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        reserve_contours.append(contours)
        for j in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[j])
            if w * h <= 10:
                continue
            bound_box[i].append((x, y, w, h))
        print(id, ":", len(contours))

    for i in range(0, len(test_ids) - 1):
        id = test_ids[i]
        next_id = test_ids[i + 1]
        img = cv2.imread(TEST_PATH + id)
        count = 0
        for j in range(len(bound_box[i])):
            flag = 0
            x, y, w, h = bound_box[i][j]
            rect = cv2.minAreaRect(reserve_contours[i][j])
            cv2.circle(img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 0, 255), 2)
            for q in range(len(bound_box[i + 1])):
                n_x, n_y, n_w, n_h = bound_box[i + 1][q]
                if overlap((x, y, w, h), (n_x, n_y, n_w, n_h)):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    count = count + 1
                    flag = 1
            if flag == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        new_id = "{}/Sequence {} mask/{}_res.tif".format(dataset, sequence, id[:-4])
        print("devide cells in ", id, ': ', count)
        divide_cell[new_id] = list
        cv2.imwrite(new_id, img)
    # return bound_box, divide_cell

draw_bounding_box("PhC-C2DL-PSC", 3)
