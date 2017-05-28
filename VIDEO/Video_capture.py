import cv2
import numpy as np
import Cascade_Haar as C
import hashlib
from time import time
if __name__ == '__main__':
    cap = cv2.VideoCapture(r'D:\Github_project\VKR\VIDEO\Capture.wmv')
    i = 0
    t = time()
    while(cap.isOpened()):

        re,frame = cap.read()
        cv2.imshow("capture", frame)
        cv2.waitKey(60)
        # if time()-t >=1:
        a = C.Cascade(frame)

        if a[0] != False and i > 1:
            path = r"D:\Github_project\VKR\VIDEO\capture\%.4d.jpg" % i # Уникальное имя для каждого кадра
            print(i)
            cv2.imwrite(path, a[1])
            i += 1
            t = time()
    cap.release()
    cv2.destroyAllWindows()
