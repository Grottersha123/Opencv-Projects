import cv2
import math
import numpy as np
import os
import Detect_lines as dec
import Copy_delete as C


def max_min(ap):
    a = np.array(ap)
    x = a[:, 0][:, 0]
    y = a[:, 0][:, 1]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    return (min_x, max_y), (max_x, min_y)
def crop_the_segments(im,p1,p2,size):
    w,h = size
    roi_im = dec.roi_area(im,p1,p2)
    res = cv2.resize(roi_im, (w, h), interpolation=cv2.INTER_CUBIC)
    return res

def seg_in_seg(point1, point2, point):
    x_min, max_y = point1
    max_x, min_y = point2
    a = inPolygon(point[0], point[1], (x_min, max_x, max_x, x_min), (min_y, min_y, max_y, max_y))
    return a


def inPolygon(x, y, xp, yp):
    c = 0
    for i in range(len(xp)):
        if (((yp[i] <= y and y < yp[i - 1]) or (yp[i - 1] <= y and y < yp[i])) and \
                    (x > (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) + xp[i])): c = 1 - c
    return c


def take_character(H, K, H_s, K_s):
    #  (min_x, max_y), (max_x, min_y)
    H_s1 = abs(K_s[0] - H_s[0])  # height
    K_s1 = abs(H_s[1] - K_s[1])
    if H[0] <= H_s1 <= H[1] and K_s1 >= K[0] and K_s1 <= K[1]:
        return True
    else:
        return False


def Squre(point1, point2):
    height = abs(point2[0] - point1[0])
    width = abs(point2[1] - point1[1])
    return height * width


def delete_crosses(arr):
    # arr = [((22, 58), (42, 20)), ...]

    result = []
    banned = []
    for i in range(len(arr)):
        if i in banned:
            continue
        point1, point2 = arr[i]
        flag = True
        for j in range(i + 1, len(arr)):
            temppoint1, temppoint2 = arr[j]
            if seg_in_seg(point1, point2, temppoint1) \
                    or seg_in_seg(point1, point2, temppoint2):
                sq = Squre(point1, point2) > Squre(temppoint1, temppoint2)
                if sq:
                    banned.append(j)
                else:
                    flag = False
                    break
        if flag:
            result.append((point1, point2))
    return result


def find_license(path,i):
    im = cv2.imread(path)
    height = np.size(im, 0)
    width = np.size(im, 1)
    h_coef_min =12
    h_coef_max = 90

    w_coef_min = 10
    w_coef_max = 90

    h_max = height * h_coef_max / 100
    h_min = height * h_coef_min / 100

    w_max = width * w_coef_max / 100
    w_min = width * w_coef_min / 100
    #imgray = black_white(im)
    gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #dec.open_picture(gray_img)
    ret, thresh = cv2.threshold(gray_img , 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    a = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.02 * peri, True)
        oh = max_min(ap)
        point1, point2 = max_min(ap)
        t = take_character((h_min, h_max), (w_min, w_max), point1, point2)
        if t:
            # print(len(ap))
            a.append(oh)

            # cv2.rectangle(im, point1, point2, (255, 22, 0), 2)
            # print(a)
            # dec.open_picture(im)

    a.sort()

    ter = delete_crosses(a)
    oh = 0
    for point1, point2 in ter:
        oh += 1
        new_oh = str(oh)
        cv2.rectangle(im , point1, point2, 255, 1)
        #cv2.rectangle(im,point1,point2,(0,0,0),-1)
        #dec.open_picture(im)
        #res = crop_the_segments(im,point1,point2,(50,70))
        #dec.save_picture(r'D:\Git_project\VKR\NEW_ROI',res,new_oh+i)
    return im

if __name__ == '__main__':
    path = r'D:\Github_project\VKR\ROI_PICTURE'
    path_out = r'D:\Git_project\VKR\BAD_ROI'
    # im = find_license(path,'1.png')
    # dec.open_picture(im)
    # os.listdir(path) ["1.png"]
    list_dir = os.listdir(path)
    # list_dir.sort()
    # print(list_dir)
    for i in list_dir:
        im = find_license("{0}\{1}".format(path, i),i)
        dec.open_picture(im,i)

