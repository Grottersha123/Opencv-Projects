import os
import cv2
import numpy as np

# ============================================================================

def extract_chars(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        x,y,w,h = x-1, y-2, w+2, h+4
        bounding_boxes.append((x,y,w,h))


    characters = []
    bounding_boxes.sort()
    for bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = img[y:y+h,x:x+w]
        characters.append(char_image)

    return characters

# ============================================================================

def output_chars(chars, labels):
    for i, char in enumerate(chars):
        filename = "chars/%s.png" % labels[i]
        char = cv2.resize(char
            , None
            , fx=3
            , fy=3
            , interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(filename, char)

# ============================================================================

if not os.path.exists("chars"):
    os.makedirs("chars")

img_digits = cv2.imread(r"D:\Github_project\OPENCV\Test\shrift\digits.png", 0)
img_letters = cv2.imread(r"D:\Github_project\OPENCV\Test\shrift\characters.png", 0)

digits = extract_chars(img_digits)
letters = extract_chars(img_letters)

DIGITS = [1,2,3,4,5,6,7,8,9,0]
LETTERS = ['A', 'B', 'C', 'D', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']

output_chars(digits, DIGITS)
output_chars(letters, LETTERS)

# CHARS = [chr(ord('0') + i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
#
# # ============================================================================
#
# def load_char_images():
#     characters = {}
#     for char in CHARS:
#         char_img = cv2.imread("chars/%s.png" % char, 0)
#         characters[char] = char_img
#     return characters
#
# # ============================================================================
#
# characters = load_char_images()
#
# samples =  np.empty((0,100))
# for char in CHARS:
#     char_img = characters[char]
#     small_char = cv2.resize(char_img,(10,10))
#     sample = small_char.reshape((1,100))
#     samples = np.append(samples,sample,0)
#
# responses = np.array([ord(c) for c in CHARS],np.float32)
# responses = responses.reshape((responses.size,1))
#
# np.savetxt('char_samples.data',samples)
# np.savetxt('char_responses.data',responses)
