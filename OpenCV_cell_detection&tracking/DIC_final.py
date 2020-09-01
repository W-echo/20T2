
import keras
from keras.models import load_model
import numpy as np
import os
import sys
import cv2
from keras.utils import Progbar
from matplotlib import pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import tensorflow.python.keras.backend as K

smooth = 1.


# Metric function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


new_model = load_model('U-net_v2.3.h5', custom_objects={'dice_coef': dice_coef})

# Data Path
dataset = "DIC-C2DH-HeLa"
sequence = 1
TEST_PATH = '{}/Sequence {}/'.format(dataset, sequence)
test_ids = next(os.walk(TEST_PATH))[2]


# Function to read test images and return as numpy array
def read_test_data(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=3):
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('\nGetting and resizing test images ... ')
    sys.stdout.flush()
    b = Progbar(len(test_ids))
    for i, id_ in enumerate(test_ids):
        path = TEST_PATH + id_
        img = cv2.imread(path)
#         img = cv2.medianBlur(img, 3)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#         img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel) # 同上
#         img = cv2.convertScaleAbs(img,alpha=8,beta=0)
#         print(np.max(img))
#         ret, img = cv.threshold(img, 190, 255, cv.THRESH_BINARY)
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[i] = img
        b.update(i)
#         break
    return X_test,sizes_test


def overlap(point1,point2):
    x1, y1, w1, h1 = point1
    x2, y2, w2, h2 = point2
    endx = max(x1+w1,x2+w2)
    startx = min(x1,x2)
    w = w1+w2-(endx-startx)
    endy = max(y1+h1,y2+h2)
    starty = min(y1,y2)
    h = h1+h2-(endy-starty)
    if w<=0 or h<=0 :
        return False
    else:
        area = w*h  # 重叠面积
        area1 = w1*h1   # 分裂前面积
        area2 = w2*h2   # 分裂后面积
        # if area*2<area2:
        #     return False
        if area1 > 1.5*area2:
            return True
        else:
            return False

def same_contour(point1,point2):
    x1, y1, w1, h1 = point1
    x2, y2, w2, h2 = point2
    endx = max(x1+w1,x2+w2)
    startx = min(x1,x2)
    w = w1+w2-(endx-startx)
    endy = max(y1+h1,y2+h2)
    starty = min(y1,y2)
    h = h1+h2-(endy-starty)
    if w<=0 or h<=0 :
        return False
    else:
        # return True
        area = w*h  # 重叠面积
        area1 = w1*h1   # 原面积
        area2 = w2*h2   # 亮度检测面积
#         print(area, area1, area2)
        if area >= 2 / 3 * min(area1, area2):
            return True
        else:
            return False


test_img,test_img_sizes = read_test_data()
test_mask = new_model.predict(test_img, verbose=1)

bound_box = []
reserve_contours = []
divide_cell = {}  # 存分裂细胞坐标
for i, id_ in enumerate(test_ids):
    kernel_size = 19
    img_path = TEST_PATH + id_
    img = cv2.imread(img_path)
    ret, thresh = cv2.threshold(test_mask[i], 0.9999, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size - 4, kernel_size - 4))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel).astype(np.uint8)

    thresh = thresh.astype(np.uint8)
    # 确定背景区域
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    # 寻找前景区域
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    if i <= 13 or 43 < i <= 51:
        k = 0.69
    elif 13 < i <= 20 or i > 58:
        k = 0.7
    elif 20 < i <= 30 or 40 < i <= 43:
        k = 0.73
    elif 30 < i <= 36 or 58 < i <= 66:
        k = 0.72
    elif 36 < i <= 40:
        k = 0.74
    elif 51 < i <= 58:
        k = 0.71
    ret, sure_fg = cv2.threshold(dist_transform, k * dist_transform.max(), 255, 0)
    # 找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers + 1
    # 现在让所有的未知区域为0
    markers[unknown == 255] = 0
    thresh = cv2.merge([thresh, thresh, thresh])
    markers = cv2.watershed(thresh.astype(np.uint8), markers)
    thresh[markers == -1] = [0, 0, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    #     cv2.imshow("", thresh)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    thresh = resize(thresh, (512, 512, 3), mode='constant', preserve_range=True).astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print("sum of cells in", id_,": ", len(contours))
    #
    # for j in range(len(contours)):
    #     x, y, w, h = cv2.boundingRect(contours[j])
    bound_box.append([])
    reserve_contours.append(contours)
    for j in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[j])
        bound_box[i].append((x, y, w, h))

for i, id_ in enumerate(test_ids[:-1]):
    img_path = TEST_PATH + id_
    img = cv2.imread(img_path)
    count = 0
    list = []  # 存分裂细胞坐标
    ret, img1 = cv2.threshold(img[:, :, 0], 90, 255, cv2.THRESH_BINARY)
    img1 = cv2.medianBlur(img1, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    img1 = cv2.morphologyEx(img1, cv2.MORPH_ERODE, kernel)
    img1 = cv2.merge([img1, img1, img1])
    img1 = resize(img1, (512, 512, 3), mode='constant', preserve_range=True).astype(np.uint8)
    img1 = 255 - img1
    #     cv2.imshow("", img1)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(img1[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new_contours = []
    for j in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[j])
        if w * h < 3000:
            continue
        else:
            new_contours.append(contours[j])

    for j in range(len(bound_box[i])):
        flag = 0
        x, y, w, h = bound_box[i][j]
        rect = cv2.minAreaRect(reserve_contours[i][j])
        cv2.circle(img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 0, 255), 2)
        for contour in new_contours:
            if same_contour((x, y, w, h), cv2.boundingRect(contour)):
                for q in range(len(bound_box[i + 1])):
                    n_x, n_y, n_w, n_h = bound_box[i + 1][q]
                    if overlap((x, y, w, h), (n_x, n_y, n_w, n_h)):
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        count = count + 1
                        flag = 1
                        list.append((x, y, w, h))
        if flag == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    RES_PATH = '{}/Sequence {} mask/'.format(dataset, sequence)
    if not os.path.exists(RES_PATH):
        os.mkdir(RES_PATH)
    new_id = RES_PATH + "{}_res.tif".format(id_[:-4])
    divide_cell[id_[:-4]] = list
    print("devide cells in ", id_, ': ', count)
    cv2.imwrite(new_id, img)

    # return bounding_box, divide_cell