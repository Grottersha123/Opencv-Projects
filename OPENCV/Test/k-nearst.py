import cv2
import numpy as np
import Detect_lines as dec
import Segmentation as s
import os

# ============================================================================

def reduce_colors(img, n):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2


# ============================================================================

def clean_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resized_img = cv2.resize(gray_img
    #                          , None
    #                          , fx=5.0
    #                          , fy=5.0
    #                          , interpolation=cv2.INTER_CUBIC)

    # resized_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
    # cv2.imwrite('licence_plate_large.png', resized_img)
    #
    # equalized_img = cv2.equalizeHist(resized_img)
    # cv2.imwrite('licence_plate_equ.png', equalized_img)

    # reduced = cv2.cvtColor(reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8), cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('licence_plate_red.png', reduced)

    # ret, mask = cv2.threshold(gray_img, 127, 255, 0)
    # cv2.imwrite('licence_plate_mask.png', mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(gray_img, kernel, iterations=1)
    cv2.imwrite('licence_plate_mask2.png', mask)

    return mask


# ============================================================================

def extract_characters(img):
    # bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    height = np.size(img, 0)
    width = np.size(img, 1)
    h_coef_min = 12
    h_coef_max = 90

    w_coef_min = 10
    w_coef_max = 90

    h_max = height * h_coef_max / 100
    h_min = height * h_coef_min / 100

    w_max = width * w_coef_max / 100
    w_min = width * w_coef_min / 100

    char_mask = np.zeros_like(img)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    bounding_boxes = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.02 * peri, True)
        oh = s.max_min(ap)
        point1, point2 = s.max_min(ap)
        t = s.take_character((h_min, h_max), (w_min, w_max), point1, point2)
        if t:
            bounding_boxes.append(oh)
            cv2.rectangle(char_mask, point1, point2, 255, -1)

    bounding_boxes.sort()
    ter = s.delete_crosses(bounding_boxes)
    for point1, point2 in ter:
        cv2.rectangle(char_mask , point1, point2, 255, 1)
    characters = []
    for point1, point2 in ter:
        x, y, = point1
        w, h = point2
        char_image = dec.roi_area(img,point1, point2)
        # dec.open_picture(char_image,'t')
        characters.append(((point1, point2), char_image))
    return characters


def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        point1, point2 = bbox
        cv2.rectangle(output_img, point1, point2, 255, 1)

    return output_img

def model_knn(samples,responses):
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
# ============================================================================os.listdir(r'D:\Github_project\VKR\ROI_PICTURE') ['106.png']
dir = ['353.png']
charss = model_knn(r'D:\Github_project\OPENCV_Examples\OPENCV\Test\chars_samples.data',r'D:\Github_project\OPENCV_Examples\OPENCV\Test\chars_responses.data')
digits = model_knn(r'D:\Github_project\OPENCV_Examples\OPENCV\Test\digits_samples.data',r'D:\Github_project\OPENCV_Examples\OPENCV\Test\digits_responses.data')
for j in dir:
    img = cv2.imread(r'D:\Github_project\VKR\ROI_PICTURE\{0}'.format(j))

    img = clean_image(img)
    chars = extract_characters(img)

    # output_img = highlight_characters(clean_img, chars)
    # cv2.imwrite('licence_plate_out.png', output_img)

    # samples = np.loadtxt(r'D:\Github_project\OPENCV\Test\chars_samples.data', np.float32)
    # # print(samples)
    # responses = np.loadtxt(r'D:\Github_project\OPENCV\Test\chars_responses.data', np.float32)
    # # print(len(responses))
    # # print(responses.size)
    # responses = responses.reshape((responses.size, 1))
    # # print(responses)
    # model = cv2.ml.KNearest_create()
    #
    # model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    # charss = model_knn(r'D:\Github_project\OPENCV\Test\chars_samples.data',r'D:\Github_project\OPENCV\Test\chars_responses.data')
    i = 1
    plate_chars = ""
    digits_chars = ''
    for bbox, char_img in chars:
        # print(char_img.size)
        small_img = cv2.resize(char_img, (100,120))
        # dec.open_picture(small_img)
        small_img = small_img.reshape((1, 12000))
        small_img = np.float32(small_img)
        retval, results, neigh_resp, dists = digits.findNearest(small_img, k=1)
        print(str(chr(results)))
        # digits_chars += str(chr((results[0][0])))
        retval, results, neigh_resp, dists = charss.findNearest(small_img, k=1)
        # print(str(chr(results)))
        # plate_chars += str(chr((results[0][0])))
        # # retval, results, neigh_resp, dists = digits.findNearest(small_img, k=1)
        # # plate_chars += str(chr((results[0][0])))
        # if i < 2 and i >= 1:
        #     retval, results, neigh_resp, dists = charss.findNearest(small_img, k=1)
        #     plate_chars += str(chr((results[0][0])))
        # if i >= 2 and i < 5:
        #     retval, results, neigh_resp, dists = digits.findNearest(small_img, k=1)
        #     plate_chars += str(chr((results[0][0])))
        # if i > 5 and i < 7:
        #     retval, results, neigh_resp, dists = charss.findNearest(small_img, k=1)
        #     plate_chars += str(chr((results[0][0])))
        # if i > 7:
        #     retval, results, neigh_resp, dists = digits.findNearest(small_img, k=1)
        #     plate_chars += str(chr((results[0][0])))
        # i+= 1
    # print("Digits: %s" % digits_chars)
    # print("Chars: %s" % plate_chars)
    # print(plate_chars[0]+digits_chars[1:4] + plate_chars[4:6]+digits_chars[6:])
    # output = highlight_characters(img,chars)
    # dec.open_picture(output,j)
