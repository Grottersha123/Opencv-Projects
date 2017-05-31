import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
def open_picture(img,i):
    # img = cv2.imread(img)
    cv2.imshow('{0}'.format(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == '__main__':
    # listdir = os.listdir(r'D:\Github_project\OPENCV\OUPUT_ANOTHER')
    # for i in listdir:
    #     img = cv2.imread(r'D:\Github_project\OPENCV\OUPUT_ANOTHER\{0}'.format(i), 0)
    #     ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #     ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     blur = cv2.GaussianBlur(img,(5,5),0)
    #     ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     images = [img, 0, th1,img, 0, th2,blur, 0, th3]
    #     titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
    # 'Original Noisy Image','Histogram',"Otsu's Thresholding",
    #         'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    #     for i in range(3):
    #         plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    #         plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #         plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    #         plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #         plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    #         plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    #         plt.show()

        listdir = os.listdir(r'D:\Github_project\VKR\OUPUT_ANOTHER')
        for j in listdir:
            img = cv2.imread(r'D:\Github_project\VKR\OUPUT_ANOTHER\{0}'.format(j), 0)
            blur = cv2.GaussianBlur(img,(5,5),0)

            # find normalized_histogram, and its cumulative distribution function
            hist = cv2.calcHist([blur],[0],None,[256],[0,256])
            hist_norm = hist.ravel()/hist.max()
            Q = hist_norm.cumsum()

            bins = np.arange(256)
            fn_min = np.inf
            thresh = -1

            for i in range(1,256):
                p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
                q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
                b1,b2 = np.hsplit(bins,[i]) # weights
             # finding means and variances
             #    print(q1,q2)
                m1 = np.sum(p1*b1)/q1
                m2 = np.sum(p2*b2)/q2
                v1 =np.sum(((b1-m1)**2)*p1)/q1
                v2 = np.sum(((b2-m2)**2)*p2)/q2

                fn = v1*q1 + v2*q2
                if fn < fn_min:
                    fn_min = fn
                    thresh = i
         # find otsu's threshold value with OpenCV function
            ret, th = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY)
            ret, th1 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
            ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            images = [th, th1, otsu,img]
            titles = ['Global Thresholding {0}'.format(thresh),'Global Thresholding 127','Otsu Thresholding','Original']
            for i in range(4):
                plt.subplot(2,3,i+3),plt.imshow(images[i],'gray')
                plt.title(titles[i])
                plt.xticks([]),plt.yticks([])
            plt.show()
            plt.imsave
            print(thresh,fn_min)
            # open_picture(otsu ,j)



