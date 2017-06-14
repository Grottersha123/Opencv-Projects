import cv2
import numpy as np
import Cascade_Haar as C
import hashlib
from time import time
if __name__ == '__main__':
    cap = cv2.VideoCapture(r'D:\Github_project\OPENCV_Examples\VIDEO\Capture.wmv')
    i = 0
    t = time()
    while(cap.isOpened()):

        re,frame = cap.read()

        cv2.waitKey(60)
        # if time()-t >=1:
        a = C.Cascade(frame)
        if a[0] != False:
            i+=1
            if i == 155 or i == 368 or i == 789:
                cv2.imwrite(r'D:\Github_project\OPENCV_Examples\VIDEO\CASCADE\{0}.png'.format(i),a[1])
        # i = 0
        print(a[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("capture", frame)
            # t = time()
    cap.release()
    cv2.destroyAllWindows()
