import cv2
import math
import numpy as np
import os
import Segmentation as seg

# import Cascade_Haar as C
from matplotlib import pyplot as plt

def save_picture(OUT, img, i):
    cv2.imwrite(OUT + '\{0}'.format(i), img)

def roi_area(img,p1,p2):
    #roi_color = img[y:y + h, x:x + w]
    roi_color = img[p2[1]:p1[1], p1[0]:p2[0]]
    return roi_color

def open_picture(img,i):
    # img = cv2.imread(img)
    cv2.imshow(i, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # D:\Git_project\VKR\CARS_ANOTHER    #D:\Git_project\VKR\CARS_ANOTHEROUPUT_ANOTHER
# cохраняет все файлы из ппаки с картинками и списком файлов из этой папки в папку OUT (все картинки на градус по алгоритму houghLine)
def save_rorates(list_dir,path_in,path_OUT):
    for i in list_dir:
        try:
            ang,img= detect_lines(r'{0}\{1}'.format(path_in,i))
            save_img = rotate(ang,img)
            #open_picture(save_img)
            save_picture(path_OUT,save_img,i)
        except Exception as inst:
            print(i,inst)
# находит прямые линии и выход выдет угол поворота

def detect_lines(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.blur(img, (5, 5))
    ret, th = cv2.threshold(blr, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(th, 50, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=20, maxLineGap=100)
    angles = []
    for i in range(0, len(lines)):
        for x, y, h, w in lines[i]:
            #print(x, y, h, w)
            tg = (w - y) / (h - x)
            #cv2.line(img, (x, y), (h, w), (255, 255, 0), 2)
            tg1 = math.degrees(math.atan(tg))
            angles.append(tg1)
    a = [i for i in angles if i != 0.0]
    try:
        av = sum(angles) / len(a)
        return (av, img)
    except:
        return (0,img)


# поворот на угол
def  rotate(an,img):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), an, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst
# поиск контуров в номерном знаке
def find_license(path):
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy =cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    i = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.02 * peri, True)
        if i ==1:
            point1, point2 = seg.max_min(ap)
            #cv2.rectangle(im, point1, point2, (255, 22, 0), 2)
            #open_picture(im)
        if i == 2:
            point1, point2 = seg.max_min(ap)
            #cv2.rectangle(im, point1, point2, (255, 22, 0), 2)
            #open_picture(im)
            print(len(ap))
        if i == 3:
            point1, point2 = seg.max_min(ap)
            #cv2.rectangle(im, point1, point2, (255, 22, 0), 2)
            #open_picture(im)
            print(len(ap))

        if i == 0:
            res = ap
            #open_picture(im)

            #point1,point2 = tuple(a[0][0]),tuple(a[2][0])
            point1, point2 = seg.max_min(ap)
            #cv2.rectangle(im, point1,point2, (255, 22, 0), 2)
            a = roi_area(im,point1,point2)
            #open_picture(im)
        i += 1
# Получаем две максимальные точки

        #cv2.drawContours(im, [ap], -1, (0, 255, 0), 3)
    #open_picture(im)
        # compute the bounding box of the of the paper region and return it
    return a


    # compute the bounding box of the of the paper region and return it
"""
if __name__ == '__main__':
    path_OUT = r'D:\Git_project\VKR\RORATE_CAR'
    path_in = r'D:\Git_project\VKR\THRESHOLD'
    path = r'D:\Git_project\VKR\RORATE_CAR'
    list_dir = os.listdir(path)
    for i in list_dir:
    #save_rorates(list_dir,path_in,path_OUT)
        ang,img = detect_lines(r'{0}\{1}'.format(path,i))
        open_picture(img)
        print(ang)
"""

if __name__ == '__main__':
    path = r'D:\Git_project\VKR\BAD_ROI'
    list_dir = ["1.png","100.png","101.png"]
    path = r'D:\Git_project\VKR\RORATE_CAR'
    for i in list_dir:
        #save_rorates(list_dir,path_in,path_OUT)
        a = find_license(r'{0}\{1}'.format(path,i))
        #open_picture(a)
        save_picture(r'D:\Git_project\VKR\ROI_PICTURE',a,i)
        #print(i)
        #print(a)








