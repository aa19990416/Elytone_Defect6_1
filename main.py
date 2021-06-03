from __future__ import print_function
import argparse
import cv2 as cv
import img as img
import numpy as np
import math
"""
N6.1 或產品之部件中孔被類似于凡立水,膠,或其他異物堵住有影響客戶
"""
max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'img Capture'
window_detection_name = 'Object Detection'
window_Trackbar_name = 'Trackbar '
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
kernel = np.ones((5, 5), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
min_canny_val = 0
max_canny_val = 100

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_min_canny_val_trackbar(val):
    global min_canny_val
    min_canny_val = val
    cv.setTrackbarPos("min_canny_val", window_detection_name, min_canny_val)

def on_max_canny_val_trackbar(val):
    global max_canny_val
    max_canny_val = val
    cv.setTrackbarPos("max_canny_val", window_detection_name, max_canny_val)

cv.namedWindow(window_Trackbar_name, 0)
cv.namedWindow(window_capture_name,0)
cv.namedWindow(window_detection_name,0)
cv.namedWindow(window_Trackbar_name, 0)
cv.createTrackbar(low_H_name, window_Trackbar_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_Trackbar_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_Trackbar_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_Trackbar_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_Trackbar_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_Trackbar_name, high_V, max_value, on_high_V_thresh_trackbar)
cv.createTrackbar("min_canny_val", window_Trackbar_name, min_canny_val, 100, on_min_canny_val_trackbar)
cv.createTrackbar("max_canny_val", window_Trackbar_name, max_canny_val, 100, on_max_canny_val_trackbar)

while True:
    #original = cv.imread('./venv/1/PXL_20210301_104459952.jpg')
    original = cv.imread('./venv/1/PXL_20210301_104518521.jpg')
    #original = cv.imread('./venv/1/PXL_20210301_104542324.jpg')

    #original = cv.imread('./venv/Side4/PXL_20210503_085205771.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085215920.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085224660.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085234884.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085244987.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085259812.jpg')#****
    #original = cv.imread('./venv/Side4/PXL_20210503_085309757.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085319272.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085327558.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085339416.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085350420.jpg')
    #original = cv.imread('./venv/Side4/PXL_20210503_085359888.jpg')

    #original = cv.imread('./venv/Side3/PXL_20210503_084739018.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_084753341.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_084803891.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_084823012.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_084835430.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_084847922.jpg')#****
    #original = cv.imread('./venv/Side3/PXL_20210503_084901025.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_084914980.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_084931268.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_084947494.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_085000436.jpg')
    #original = cv.imread('./venv/Side3/PXL_20210503_085014739.jpg')

    original = cv.resize(original, (800, 650), interpolation=cv.INTER_AREA)
    frame = np.copy(original)

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))#測試用
    frame_threshold = cv.inRange(frame_HSV, (20, 0, 180), (75, 255, 255))
    frame_threshold = cv.morphologyEx(frame_threshold, cv.MORPH_OPEN, kernel)
    frame_threshold = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)

    # 最大面積最小矩形 #
    contours, hierarchy = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = []

    for c in range(len(contours)):
        areas.append(cv.contourArea(contours[c]))
    max_id = areas.index(max(areas))
    max_rect = cv.minAreaRect(contours[max_id])
    max_box = cv.boxPoints(max_rect)
    max_box = np.int0(max_box)
    min_x = min(max_box[0][0], max_box[1][0], max_box[2][0], max_box[3][0])
    max_x = max(max_box[0][0], max_box[1][0], max_box[2][0], max_box[3][0])
    min_y = min(max_box[0][1], max_box[1][1], max_box[2][1], max_box[3][1])
    max_y = max(max_box[0][1], max_box[1][1], max_box[2][1], max_box[3][1])
    max_box = np.sort(max_box,axis=0)
    image = frame[min_y:max_y, min_x:max_x]
    points = np.array([[min_x, max_y-20], [min_x, min_y+20], [max_x, min_y+20], [max_x, max_y-20]])
    black = (0, 0, 0)  # BGR
    image_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    blurred = cv.GaussianBlur(image_HSV, (3, 3), 0)
    canny = cv.Canny(blurred, min_canny_val, max_canny_val, 3, 3, True)
    #canny = cv.Canny(blurred, 0, 99, 3, 3, True)

    try:
        imagebt_bt_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        #imagebt_threshold = cv.inRange(imagebt_bt_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        imagebt_threshold = cv.inRange(imagebt_bt_HSV, (0, 90, 0), (255, 255, 255))
        #imagebt_threshold = cv.morphologyEx(imagebt_threshold, cv.MORPH_OPEN, kernel)
        imagebt_threshold = cv.morphologyEx(imagebt_threshold, cv.MORPH_CLOSE, kernel)
        imagebt_threshold = cv.morphologyEx(imagebt_threshold, cv.MORPH_OPEN, kernel)
        imagebt_threshold = cv.morphologyEx(imagebt_threshold, cv.MORPH_CLOSE, kernel)
        imagebt_threshold = cv.morphologyEx(imagebt_threshold, cv.MORPH_OPEN, kernel)
        imagebt_threshold = cv.cv2.dilate(imagebt_threshold,kernel,iterations = 1)
        bt_image = cv.bitwise_and(canny,imagebt_threshold)
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bt_image, connectivity=8)
        # print('num_labels = ', num_labels)
        # 连通域的信息：对应各个轮廓的x、y、width、height和面积
        # print('stats = ', stats)
        # 连通域的中心点
        # print('centroids = ', centroids)
        # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
        # print('labels = ', labels)
        for i in range(1, num_labels):
            mask = labels == i
            #print('stats = ', stats[i][4])
            #image[:, :, 0][mask] = np.random.randint(0, 255)
            #image[:, :, 1][mask] = np.random.randint(0, 255)
            #image[:, :, 2][mask] = np.random.randint(0, 255)
            if( stats[i][2] < 60 and stats[i][2] > 0 and stats[i][3] < 60 and stats[i][3] > 0):
                point = (stats[i][0],stats[i][1])
                cv.rectangle(image, (stats[i][0],stats[i][1]), (stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]),(0,0,0), -1)
                #cv.circle(image, point, 1, (0, 0, 255), 4)
                #print('stats = ', stats)
                #image[:, :, 0][mask] = np.random.randint(0, 255)
                #image[:, :, 1][mask] = np.random.randint(0, 255)
                #image[:, :, 2][mask] = np.random.randint(0, 255)
        cv.imshow("imagebt_threshold", imagebt_threshold)
        cv.imshow("imagebt", bt_image)

    except (ValueError, TypeError):
        print("nomal")
    try:
        image_canny_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # image_threshold = cv.inRange(image_HSV, (24, 111, 213), (58, 168, 234))
        #image_threshold = cv.inRange(image_canny_HSV, (20, 123, 200), (75, 213, 255))
        image_threshold = cv.inRange(image_canny_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        image_threshold = cv.inRange(image_canny_HSV, (15, 80, 0), (70, 255, 255))#**************
        #image_threshold = cv.inRange(image_canny_HSV, (15, 115, 0), (65, 187, 255))
        image_threshold = cv.morphologyEx(image_threshold, cv.MORPH_OPEN, kernel)
        # image_threshold = cv.morphologyEx(image_threshold, cv.MORPH_CLOSE, kernel)
        cv.imshow("image_threshold_window", image_threshold)
        # 最大面積最小矩形 #
        contours1, hierarchy1 = cv.findContours(image_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        areas1 = []
        for c in range(len(contours1)):
            areas1.append(cv.contourArea(contours1[c]))
            max_id1 = areas1.index(max(areas1))
            max_rect1 = cv.minAreaRect(contours1[max_id1])
            max_box1 = cv.boxPoints(max_rect1)
            max_box1 = np.int0(max_box1)
            #max_box1_point = max_box1
            #max_box1_point = np.sort(max_box1_point,axis=0)
            #if (abs(max_box1_point[0][0] - max_box1_point[1][0] )>10 and abs(max_box1_point[2][0] - max_box1_point[3][0] )>10) :
            cv.drawContours(image, [max_box1], 0, (0, 0, 255), 1)
            #print(max_box1_point)
    except (ValueError, TypeError):
        print("complete")
    cv.imshow(window_capture_name, frame)

    cv.imshow(window_detection_name, frame_threshold)

    cv.imshow("original_window", original)

    cv.imshow("img_window", image)

    cv.imshow("blurred_window", blurred)
    cv.imshow("image_canny_HSV", image_canny_HSV)
    cv.imshow("canny_window", canny)
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
print(max_box)
