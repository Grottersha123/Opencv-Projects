import numpy as np
import cv2
import Cascade_Haar as C
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plaques = plate_cascade.detectMultiScale(gray, 1.3, 10)
        for i, (x, y, w, h) in enumerate(plaques):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # cv2.imwrite(path, a[1])
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
