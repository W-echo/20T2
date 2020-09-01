import cv2
import sys
import os
import numpy as np

dataset1 = "DIC-C2DH-HeLa"
dataset3 = "PhC-C2DL-PSC"

t_img = cv2.imread("t425.tif")

def preprocess(img):
    new_img = cv2.medianBlur(img, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    new_img = cv2.morphologyEx(new_img, cv2.MORPH_TOPHAT, kernel)
    new_img = cv2.convertScaleAbs(new_img,alpha=8,beta=0)
    return new_img
    ret, thresh = cv2.threshold(new_img, 190, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        rect = cv2.minAreaRect(contours[i])
        cv2.circle(img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 0, 255), 2)
    #cv2.imshow("", new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def draw_bounding_box(dataset, sequence):
    """
    dataset: the name of dataset
    sequence: sequence number
    """
    # Data Path
    TEST_PATH = '{}/Sequence {}/'.format(dataset, sequence)
    test_ids = next(os.walk(TEST_PATH))[2]
    RES_PATH = "{}/Sequence {} mask/".format(dataset, sequence)
    if not os.path.exists(RES_PATH):
        os.mkdir(RES_PATH)
    bound_box=[]
    for id in test_ids:
        bound_box.append([])
        img = cv2.imread(TEST_PATH + id)
#         print(img.shape)
        new_img = preprocess(img)
        ret, thresh = cv2.threshold(new_img, 190, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            bound_box[-1].append((x,y,w,h))
            if w * h <= 10:
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            rect = cv2.minAreaRect(contours[i])
            cv2.circle(img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 0, 255), 2)
        #cv2.imshow(id,img)
        #cv2.waitKey(0)
        #cv2.cv2.destroyAllWindows()
        #print(id, ":", len(contours))
        new_id = "{}/Sequence {} mask/{}_res.tif".format(dataset, sequence, id[:-4])
        cv2.imwrite(new_id, img)
    return bound_box
        
#BBox_list=draw_bounding_box("PhC-C2DL-PSC", 1)

def Tracking(dataset, sequence,num):
    TEST_PATH = '{}/Sequence {}/'.format(dataset, sequence)
    pic_name=os.path.join(TEST_PATH, os.listdir(TEST_PATH)[num])
    ima=cv2.imread(pic_name)
    bound_box=draw_bounding_box(dataset, sequence)
    if num>len(bound_box)-1:
        print("Wrong id.")
        return 0
    else:
        tracking_aim=bound_box[num]
    line_set=[]
    o=0
    for box in tracking_aim:
        if o<=30:
            line=single_track(box,dataset, sequence,num,10)
            o+=1
            print(o)
        else:
            pass
        line_set.append(line)
    for line in line_set:
        for i in range(len(line)-1):
	        cv2.line(ima,line[i],line[i+1],(0,0,255),1,4)
    cv2.imshow("2",ima)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def single_track(box,dataset, sequence,num,beginning):
    TEST_PATH = '{}/Sequence {}/'.format(dataset, sequence)
    pic_name=os.path.join(TEST_PATH, os.listdir(TEST_PATH)[num])
    ima=cv2.imread(pic_name)
    multiTracker=cv2.MultiTracker_create()
    if beginning==10:
        begin_point=max(0,num-10)
    elif beginning==1:
        begin_point=max(0,num-1)
    elif beginning==0:
        begin_point=0
    tracker=cv2.TrackerCSRT_create()
    multiTracker.add(tracker,ima,box)
    line=[(int(box[0])+int(box[2])//2,int(box[1])+int(box[3])//2)]
    for i in range(num,begin_point-1,-1):
        pic_name = os.listdir(TEST_PATH)[i]
        pic_name = os.path.join(TEST_PATH, pic_name)
        pic=cv2.imread(pic_name)
        success, boxes = multiTracker.update(pic)
        line.append((int(boxes[-1][0])+int(boxes[-1][2])//2,int(boxes[-1][1])+int(boxes[-1][3])//2))
        #cv2.line(pic,line[-1],line[-2],(0,0,255),5,4)
        #cv2.imshow("1",pic)
        #cv2.waitKey(0)
    return line
    
def distance(pixel_1,pixel_2):
    square_sum=pow(pixel_1[0]-pixel_2[0],2)+pow(pixel_1[1]-pixel_2[1],2)
    return pow(square_sum,0.5)

def in_box(box_1,box_2):
    if ((box_1[0]<=box_2[0]) and (box_1[1]<=box_2[1])):
        if ((box_1[0]+box_1[2]>=box_2[0]+box_1[2]) and (box_1[1]+box_1[3]>=box_2[1]+box_2[3])):
            return True
        else:
            return False
    elif ((box_1[0]>=box_2[0]) and (box_1[1]>=box_2[1])):
        if ((box_1[0]+box_1[2]<=box_2[0]+box_2[2]) and (box_1[1]+box_1[3]<=box_2[1]+box_2[3])):
            return True
        else:
            return False
    else:
        return False

def Speed(dataset, sequence,num):
    b_box=Box_select(dataset, sequence,num)
    if b_box==None:
        return 0
    beginning=0
    #beginning=begin_detecting(dataset, sequence,num,b_box)
    line=single_track(b_box,dataset, sequence,num,beginning)
    if len(line)==1:
        return 0
    else:
        frame_interval=1
        cell_speed=distance(line[0],line[-1])//frame_interval
        return cell_speed
    
def Total_distance(dataset, sequence,num):
    b_box=Box_select(dataset, sequence,num)
    if b_box==None:
        return 0
    beginning=0
    #beginning=begin_detecting(dataset, sequence,num,b_box)
    line=single_track(b_box,dataset, sequence,num,beginning)
    if len(line)==1:
        return 0
    else:
        sum_distance=0
        for i in range(len(line)-1):
            sum_distance+=distance(line[i],line[i+1])
        return sum_distance
    
def Net_distance(dataset, sequence,num):
    b_box=Box_select(dataset, sequence,num)
    if b_box==None:
        return 0
    beginning=0
    #beginning=begin_detecting(dataset, sequence,num,b_box)
    line=single_track(b_box,dataset, sequence,num,beginning)
    if len(line)==1:
        return 0
    else:
        NetDistance=distance(line[0],line[-1])
        return NetDistance
      
def Box_select(dataset, sequence,num):
    TEST_PATH = '{}/Sequence {}/'.format(dataset, sequence)
    pic_name=os.path.join(TEST_PATH, os.listdir(TEST_PATH)[num])
    ima=cv2.imread(pic_name)
    box=cv2.selectROI('Select a cell',ima)
    #print(box)
    bound_box=draw_bounding_box(dataset, sequence)
    if num>len(bound_box)-1:
        print("Wrong id.")
        return 0
    else:
        tracking_aim=bound_box[num]
    for b_box in bound_box[num]:
        if not in_box(b_box,box):
            pass
        else:
            return b_box
    print("Selecting failed")
        
Tracking("PhC-C2DL-PSC", 2  ,11)
#speed=Speed("PhC-C2DL-PSC", 2  ,11)
#print("Speed:",speed)
#sum_distance=Total_distance("PhC-C2DL-PSC", 2  ,22)
#print("Total distance:",sum_distance)
net_distance=Net_distance("PhC-C2DL-PSC", 2  ,22)
print("Net distance:",net_distance)
#print("ratio:",sum_distance/net_distance)
'''
Speed: 0.0
Total distance: 19.242640687119287
Net distance: 3.605551275463989
ratio: 5.336948282518324
'''
