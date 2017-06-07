from urllib.request import urlopen
import cv2
import math
import numpy as np

from matplotlib import pyplot as plt
import os
class Operation:
    def __init__(self,img,i):
        self.image = img
        self.count = i

    def save_picture_p(self,OUT):
        cv2.imwrite('{0}\\{1}'.format(OUT,self.count), self.image)

    def save_picture(self,img,OUT,i):
        cv2.imwrite('{0}\\{1}'.format(OUT,i), img)
    def open_picture(self):
        # img = cv2.imread(img)
        cv2.imshow(self.count, self.image)
        res = cv2.waitKey(0)
        # print(res)
        cv2.destroyAllWindows()
    def open_picture(self,img,name):
        # img = cv2.imread(img)
        cv2.imshow(name, img)
        res = cv2.waitKey(0)
        # print(res)
        cv2.destroyAllWindows()


class Detection(Operation):

    URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml'
    current_path = os.getcwd()
    path_tresh ="{0}\\{1}".format(current_path,'Tresholding')
    path_tresh_otsu ="{0}\\{1}".format(current_path,'Tresholding_127')
    path_tresh_tr = "{0}\\{1}".format(current_path,'Tresholding_tr')
    path_native = "{0}\\{1}".format(current_path,'Haar')
    path_thres_all = ''
    # path_rorate = "{0}\\{1}".format(current_path,'Rorated')
    # path_loc ="{0}\\{1}".format(current_path,'Local')
    def __init__(self,img,name):
        self.original_image = img
        self.tresh = None
        self.native = None
        self.rorate = None
        self.loc = None
        self.img_name = name
        self.roi_segment = []
        # if  os.path.exists(self.path_tresh) == False or os.path.exists(self.path_tresh_otsu) == False:
        #     os.mkdir(self.path_tresh)
        #     os.mkdir(self.path_native)
        #     os.mkdir(self.path_tresh_otsu)
            # os.mkdir(self.path_tresh_tr)
            # os.mkdir(self.path_loc)
    # Скачивание файла каскада хаара
    def download(self):
        logo = urlopen(self.URL).read()
        f = open("haarcascade_russian_plate_number.xml", "wb")
        f.write(logo)
        f.close()

    def thretholding(self,flag=False):
        if self.native is  None:
            print("You didnt create cascade image!")
        else:
            #img = cv2.GaussianBlur(img, (5, 5), 0)
            # th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,111,3)
            gray = cv2.cvtColor(self.native, cv2.COLOR_BGR2GRAY)

            #img = cv2.blur(img, (5, 5))
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            self.tresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            if flag:
                Operation.save_picture(self,self.tresh,self.path_tresh,self.img_name)
                print('Your picture-thresholding save in THresh dir')
            else:
                print("Your picture is thresholding")

    def thretholding_tr(self):
        gray = cv2.cvtColor(self.native, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)

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
        plt.savefig(self.img_name)

    def Cascade(self):
        print('Hi')

        if os.path.exists("{0}//{1}".format(self.current_path,'haarcascade_russian_plate_number.xml')):
            plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
            # open_picture(gray)
            try:
                self.native = self.original_image
                gray = cv2.cvtColor(self.native, cv2.COLOR_BGR2GRAY)
                plaques = plate_cascade.detectMultiScale(gray, 1.3, 5)
                for i, (x, y, w, h) in enumerate(plaques):
                    roi_color = self.native[y:y + h, x:x + w]
                    r = 300.0 / roi_color.shape[1]
                    dim = (300, int(roi_color.shape[0] * r))
                    resized = cv2.resize(roi_color, dim, interpolation=cv2.INTER_AREA)
                    self.native = resized

            except:
                flag = False
                print("Sorry,Error")
        else:
            self.download()
            self.Cascade()
    def Creates_pictures(self,flag = False):
            self.Cascade()
            a = Operation(self.native,self.img_name)
            if flag:

                a.save_picture(self.path_native)
            self.thretholding()
            b = Operation(self.tresh,self.img_name)
            if flag:
                b.save_picture(self.path_tresh)
    # def Creates_pictures(self):
    #         self.Cascade()
    #         a = Operation(self.native,self.img_name)
    #         a.save_picture(self.path_native)
    #         self.thretholding()
    #         b = Operation(self.tresh,self.img_name)
    #         b.save_picture(self.path_tresh)

class Localisation(Detection):
    current_path = os.getcwd()
    path_rorate_local = "{0}\\{1}".format(current_path,'Rorate_and_Local')
    def __init__(self,img,i):
            self.tr_img = img
            self.img_name = i
            self.rorate_rect = None
            self.rorate = None
            self.points = None
            self.local = None
            self.ang = None

            if os.path.exists(self.path_rorate_local) == False:
                os.mkdir(self.path_rorate_local)
    def detect_lines(self):
        edges = cv2.Canny(self.tr_img, 50, 100, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=20, maxLineGap=100)
        angles = []
        if lines is None:
            print('miss '+self.img_name)
            self.ang  = 0.0
        else:
            for i in range(0, len(lines)):
                for x, y, h, w in lines[i]:
                    #print(x, y, h, w)
                    tg = (w - y) / (h - x)
                    #cv2.line(img, (x, y), (h, w), (255, 255, 0), 2)
                    tg1 = math.degrees(math.atan(tg))
                    angles.append(tg1)
            a = [i for i in angles if i != 0.0]
            try:
                self.ang = sum(angles) / len(a)
                # return (av, img)
            except:
                self.ang  = 0.0
                # return (0,img)
    def rotate(self,flag = False):
        self.detect_lines()
        rows, cols= self.tr_img.shape

        print(self.ang)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.ang, 1)
        self.rorate = cv2.warpAffine(self.tr_img, M, (cols, rows))
        print('Picture rorated')
        if flag:
            Operation.open_picture(self,self.rorate,'lol')
    def roi_area(self,p1,p2,img = None):
    #roi_color = img[y:y + h, x:x + w]
        if img is None:
            self.local = self.rorate[p2[1]:p1[1], p1[0]:p2[0]]
        else:
            img = img[p2[1]:p1[1], p1[0]:p2[0]]

            return img
    def max_min(self,points = None):
        if points is not None:
            a = np.array(points)
        else:
            a = np.array(self.points)
        x = a[:, 0][:, 0]
        y = a[:, 0][:, 1]
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        return (min_x, max_y), (max_x, min_y)
    def find_license(self,flag = False):
        # imgray = cv2.cvtColor(self.rorate, cv2.COLOR_BGR2GRAY)
        im2, contours, hierarchy =cv2.findContours(self.rorate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        i = 0
        for c in contours:
            peri = cv2.arcLength(c, True)
            self.points = cv2.approxPolyDP(c, 0.02 * peri, True)
            if i ==1:
                pointt1, pointt2 = self.max_min()
                max_x = max((max_x,pointt2[0]))
                if flag == True:
                    # cv2.rectangle(self.rorate, pointt1, pointt2, (255, 22, 0), 2)
                    # cv2.rectangle(self.rorate, point1, (max_x,point2[1]), (255, 22, 0), 2)
                     Operation.open_picture(self,self.rorate,'lol')
                    # Operation.save_picture(self,self.local,self.path_rorate_local,self.img_name)
                else:
                    self.roi_area(point1,(max_x,point2[1]))
                    print('Picture Localisation')

            if i == 0:
                # res = ap
                #open_picture(im)

                #point1,point2 = tuple(a[0][0]),tuple(a[2][0])
                point1, point2 = self.max_min()
                max_x = point2[0]
                # cv2.rectangle(self.rorate_rect, point1, point2, (255, 22, 0), 2)
                 # = roi_area(im,point1,point2)
                #open_picture(im)
            i += 1
    def Rorate_Locat(self):
        Detection.Creates_pictures()
        self.rorate()
        self.find_license()
    def Already_tresh(self):
        self.rotate(flag=False)
        self.find_license()


class Recognition(Localisation):
    height = 0
    width = 0

    def __init__(self,img,i):
        self.img_name = i
        self.reduce_img = img
        self.loc_img = img
        self.img_red = img
        self.characters = []
        self.plate_chars = ""
        self.digits_chars = ''
        self.result = ''

    def take_character(self,H, K, H_s, K_s):
    #  (min_x, max_y), (max_x, min_y)
        H_s1 = abs(K_s[0] - H_s[0])  # height
        K_s1 = abs(H_s[1] - K_s[1])
        if H[0] <= H_s1 <= H[1] and K_s1 >= K[0] and K_s1 <= K[1]:
            return True
        else:
            return False


    def seg_in_seg(self,point1, point2, point):
        x_min, max_y = point1
        max_x, min_y = point2
        a = self.inPolygon(point[0], point[1], (x_min, max_x, max_x, x_min), (min_y, min_y, max_y, max_y))
        return a


    def inPolygon(self,x, y, xp, yp):
        c = 0
        for i in range(len(xp)):
            if (((yp[i] <= y and y < yp[i - 1]) or (yp[i - 1] <= y and y < yp[i])) and \
                        (x > (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) + xp[i])): c = 1 - c
        return c


    def Squre(self,point1, point2):
        height = abs(point2[0] - point1[0])
        width = abs(point2[1] - point1[1])
        return height * width


    def delete_crosses(self,arr):
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
                if self.seg_in_seg(point1, point2, temppoint1) \
                        or self.seg_in_seg(point1, point2, temppoint2):
                    sq = self.Squre(point1, point2) > self.Squre(temppoint1, temppoint2)
                    if sq:
                        banned.append(j)
                    else:
                        flag = False
                        break
            if flag:
                result.append((point1, point2))
        return result


    def clean_image_1(self):
        # gray_img = cv2.cvtColor(self.reduce_img, cv2.COLOR_BGR2GRAY)

        # resized_img = cv2.resize(gray_img
        #                          , None
        #                          , fx=5.0
        #                          , fy=5.0
        #                          , interpolation=cv2.INTER_CUBIC)
        #
        # resized_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
        # # cv2.imwrite('licence_plate_large.png', resized_img)
        #
        # equalized_img = cv2.equalizeHist(resized_img)
        # cv2.imwrite('licence_plate_equ.png', equalized_img)

        # reduced = cv2.cvtColor(self.reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8), cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('licence_plate_red.png', reduced)

        # ret, mask = cv2.threshold(gray_img, 127, 255, 0)
        # cv2.imwrite('licence_plate_mask.png', mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.reduce_img = cv2.erode(self.reduce_img, kernel, iterations=1)
        # cv2.imwrite('licence_plate_mask2.png', mask)

        # return mask

    def extract_characters(self,flag = False):

        # bw_image = cv2.bitwise_not(img)
        # imgray = cv2.cvtColor(self.loc_img, cv2.COLOR_BGR2GRAY)

        # self.reduce_img = self.clean_image_1()
        # imgray = cv2.cvtColor(self.reduce_img, cv2.COLOR_BGR2GRAY)
        contours = cv2.findContours(self.reduce_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        height = np.size(self.loc_img, 0)
        width = np.size(self.loc_img, 1)
        h_coef_min = 12
        h_coef_max = 70

        w_coef_min = 10
        w_coef_max =70

        h_max = height * h_coef_max / 100
        h_min = height * h_coef_min / 100

        w_max = width * w_coef_max / 100
        w_min = width * w_coef_min / 100

        char_mask = np.zeros_like(self.reduce_img)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        bounding_boxes = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            ap = cv2.approxPolyDP(c, 0.02 * peri, True)
            oh = Localisation.max_min(self,points=ap)
            point1, point2 = Localisation.max_min(self,points=ap)
            t = self.take_character((h_min, h_max), (w_min, w_max), point1, point2)
            i= 0
            if t:
                bounding_boxes.append(oh)
                # cv2.rectangle(self.reduce_img, point1, point2, (255, 45,60), 3 )
                # Operation.open_picture(self,self.loc_img,'lol')
            # Operation.save_picture(self,self.reduce_img,r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\Segmentation_1',self.img_name)
            if flag == True:
                cv2.rectangle(self.loc_img, point1, point2, (0, 0,0), 3 )
                Operation.open_picture(self,self.loc_img,'lol')

        bounding_boxes.sort()
        ter = self.delete_crosses(bounding_boxes)
        # for point1, point2 in ter:
        #     cv2.rectangle(self.reduce_img, point1, point2, (255, 45,60), 3 )
        # Operation.save_picture(self,self.reduce_img,r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\Segmentation_2',self.img_name)

        for point1, point2 in ter:
            x, y = point1
            w, h = point2
            char_image = Localisation.roi_area(self,point1, point2,img = self.reduce_img)
            # Operation.open_picture(self,char_image,'lol')
            # Operation.save_picture(self,char_image,r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\Segmentation_1',self.img_name)

            # dec.open_picture(char_image,'t')
            self.characters.append(((point1, point2), char_image))
        # Operation.save_picture(self,self.reduce_img,r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\Segmentation_2',self.img_name)
        # return characters
    def model_knn(self,samples,responses):
        samples = np.loadtxt(samples, np.float32)
        # print(samples)
        responses = np.loadtxt(responses, np.float32)
        # print(len(responses))
        # print(responses.size)
        responses = responses.reshape((responses.size, 1))
        # print(responses)
        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)
        return model
    def Recorgnition_KNN(self,Flag = False):
        self.clean_image_1()
        self.extract_characters(flag= False)
        charss = self.model_knn(r'chars_samples.data',r'chars_responses.data')
        digits = self.model_knn(r'digits_samples.data',r'digits_responses.data')

        for bbox, char_img in self.characters:
            # print(char_img.size)
            small_img = cv2.resize(char_img, (100,120))
            # dec.open_picture(small_img)
            small_img = small_img.reshape((1, 12000))
            small_img = np.float32(small_img)
            retval, dig, neigh_resp, dists = digits.findNearest(small_img, k=1)
            # print(str(chr(dig)))
            self.digits_chars += str(chr((dig[0][0])))

            retval, chars, neigh_resp, dists = charss.findNearest(small_img, k=1)
            self.plate_chars += str(chr((chars[0][0])))
        # return (self.plate_chars)
        self.result = (self.plate_chars[0]+self.digits_chars[1:4] + self.plate_chars[4:6]+self.digits_chars[6:])
        if Flag:
            Operation.save_picture(self,self.reduce_img,r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\Segmentation_2','{0}.png'.format(a))
        return self.result


def Prepare_Img(path):
    im  = cv2.imread(r"{0}".format(path))
    img_name =os.path.split(path)
    cas = Detection(im,img_name)
    cas.Creates_pictures()
    l =  Localisation(cas.tresh,img_name)
    l.Already_tresh()
    r = Recognition(l.local,img_name)
    a = r.Recorgnition_KNN()
    print(a)

def Open_Img_Seg(path):
    im  = cv2.imread(r"{0}".format(path))
    im  = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_name =os.path.split(path)
    l =  Localisation(im,img_name)
    l.Already_tresh()
    r = Recognition(l.local,img_name)
    a = r.Recorgnition_KNN()
    print(a)


def Prepare_all_Img(path):
    pass
    # lst =os.listdir()
    




if __name__ == '__main__':
    path = r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\Video\THres1030.jpg'
    Open_Img_Seg(path)
    # Open_Img_Seg(r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\Tresholding_tr\THres0032.jpg')
    # Prepare_Img(r'D:\Github_project\VKR\CARS_ANOTHER\1.png')
    # path = r'D:\Github_project\VKR\CARS_ANOTHER'
    # a = Detection.path_tresh
    # b = Localisation.path_rorate_local
    # # os.listdir(b) ['213.png','412.png','407.png'] ['1.png','213.png','412.png','407.png']
    # lst =os.listdir(r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\Segmentation_1')
    # # lst = ['1.png','213.png','412.png','407.png','109.png']
    # for i in lst:
    #     print(r"{0}\{1}".format(b,i))
    #     img = cv2.imread(r"{0}\{1}".format(b,i))
    #     # s = Detection(img,i)
    #     # s.tresh = s
    #     l = Localisation(img,i)
    #     l.img_name = i
    #     # l = Localisation(img,i)
    #     # l.Already_tresh()
    #     r = Recognition(img,i)
    #
    #     a = r.Recorgnition_KNN()
    #     print(a)


