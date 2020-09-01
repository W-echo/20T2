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


model = load_model('U-net.h5', custom_objects={'dice_coef': dice_coef})


dataset = "Fluo-N2DL-HeLa"
sequence = 1
TEST_PATH = '{}/Sequence {}/'.format(dataset, sequence)
test_ids = next(os.walk(TEST_PATH))[2]
test_ids.sort()
RES_PATH = '{}/Sequence {} mask/'.format(dataset, sequence)
if not os.path.exists(RES_PATH):
    os.mkdir(RES_PATH)

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
        img = cv2.medianBlur(img, 5)
#         print(np.min(img))
        ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
#         print(img.shape)
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[i] = img
        b.update(i+1)
    return X_test,sizes_test


test_img,test_img_sizes = read_test_data()
test_mask = model.predict(test_img, verbose=1)


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
        if area1>1.2*area2:
            return True
        else:
            return False
        # ratio = area/(area1+area2-area)
        # if ratio>=0.5:
        #     return True
        # else:
        #     return False


bound_box = []
reserve_contours = []
divide_cell = {}  # 存分裂细胞坐标
for i, id_ in enumerate(test_ids):
    path = TEST_PATH + id_
    img = cv2.imread(path)
    bound_box.append([])
    #     print(img.shape)

    ret, thresh = cv2.threshold(test_mask[i], 0.9, 255, cv2.THRESH_BINARY)
    thresh = resize(thresh, (700, 1100), mode='constant', preserve_range=True).astype(np.uint8)
    thresh = cv2.medianBlur(thresh, 5)
#     cv2.imshow("", thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    img = cv2.merge([thresh, thresh, thresh])
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    reserve_contours.append(contours)
    print(id_, ":", len(contours))

    for j in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[j])
        if w * h <= 30:
            continue
        bound_box[i].append((x, y, w, h))
    new_id = "{}/Sequence {} mask/{}.tif".format(dataset, sequence, id_[:-4])
    cv2.imwrite(new_id, img)

for i in range(0,len(test_ids)-1):
    id = test_ids[i]
    next_id = test_ids[i+1]
    img = cv2.imread(RES_PATH + id)
    count = 0
    list = []  # 存分裂细胞坐标
    img = cv2.threshold(img, 129, 255, cv2.THRESH_BINARY)[1]
    for j in range(len(bound_box[i])):
        flag = 0
        x, y, w, h = bound_box[i][j]
        if w * h <= 30:
            continue
        rect = cv2.minAreaRect(reserve_contours[i][j])
        cv2.circle(img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 0, 255), 2)

        for q in range(len(bound_box[i+1])):
            n_x, n_y, n_w, n_h = bound_box[i+1][q]
            if overlap((x, y, w, h),(n_x, n_y, n_w, n_h)):
                # rect = cv2.minAreaRect(reserve_contours[i][j])
                # cv2.circle(img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                count = count+1
                flag = 1
                list.append((x,y,w,h))
        if flag == 0:
            # rect = cv2.minAreaRect(reserve_contours[i][j])
            # cv2.circle(img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    new_id = "{}/Sequence {} mask/{}.tif".format(dataset, sequence, id[:-4])
    divide_cell[new_id] = list
    print("devide cells in ", id, ': ', count)
    cv2.imwrite(new_id, img)
    # return bound_box, divide_cell

