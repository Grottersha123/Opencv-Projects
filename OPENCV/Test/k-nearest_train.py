import os
import cv2
import numpy as np
import Detect_lines as dec

# ============================================================================
CHARS = ['A','B','C','E','H','K','M','O','P','T','X','Y']
DIGITS = [chr(ord('0') + i) for i in range(10)]

def extract_chars(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        x,y,w,h = x-2, y-2, w+4, h+4
        bounding_boxes.append((x,y,w,h))
    bounding_boxes.sort()
    characters = []
    for bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = img[y:y+h,x:x+w]
        #dec.open_picture(char_image)
        characters.append(char_image)

    return characters

# ============================================================================

def output_chars(chars, labels):
    for i, char in enumerate(chars):
        filename = "D:\Github_project\VKR\mine_character\%s.png" % labels[i]
        print(filename)
        char = cv2.resize(char
            , None
            , fx=3
            , fy=3
            , interpolation=cv2.INTER_CUBIC)
        #dec.open_picture(char)
        cv2.imwrite(filename, char)
        print('lol')

# ============================================================================
def load_char_images(CHARS,path):
    characters = {}
    for char in CHARS:
        char_img = cv2.imread(r'{0}\{1}.png'.format(path,char), 0)
        print(r'{0}\{1}.png'.format(path,char))
        characters[char] = char_img
    return characters
def train_text(w,h,CHARS,name,path):
    characters = load_char_images(CHARS,path)
    samples =  np.empty((0,h*w))
    for char in CHARS:
        char_img = characters[char]
        small_char = cv2.resize(char_img,(w,h))
        sample = small_char.reshape((1,h*w))
        samples = np.append(samples,sample,0)
    responses = np.array([ord(c) for c in CHARS],np.float32)
    responses = responses.reshape((responses.size,1))
    np.savetxt('{0}_samples.data'.format(name),samples)
    np.savetxt('{0}_responses.data'.format(name),responses)


# if not os.path.exists("chars"):
#     os.makedirs("chars")
#
# img_digits = cv2.imread(r'D:\Github_project\VKR\mine_character\digits.png', 0)
# img_letters = cv2.imread(r"D:\Github_project\VKR\mine_character\characters.png", 0)
#
# digits = extract_chars(img_digits)
# letters = extract_chars(img_letters)
#
# DIGITS = [1,2,3,4,5,6,7,8,9,0]
# LETTERS = ['A','B','C','D','E','H','K','M','O','P','T','X','Y']
#
# output_chars(digits, DIGITS)
# output_chars(letters, LETTERS)
# 100,120 - size of picture
if __name__ == '__main__':
    train_text(100,120,CHARS,'chars',r'D:\Github_project\OPENCV\Test\chars\chars')
    train_text(100,120,DIGITS,'digits',r'D:\Github_project\OPENCV\Test\chars\digits')
    # print(DIGITS)
    # characters = load_char_images(CHARS)
    # samples =  np.empty((0,12000))
    # for char in CHARS:
    #     char_img = characters[char]
    #     small_char = cv2.resize(char_img,(100,120))
    #     sample = small_char.reshape((1,12000))
    #     samples = np.append(samples,sample,0)
    # # np.savetxt('char_samples.data',samples)
    # np.savetxt('char_responses.data',responses)
    #
    # responses = np.array([ord(c) for c in CHARS],np.float32)
    # print(responses)
    # responses = responses.reshape((responses.size,1))
    # print(samples)
    # np.savetxt('char_samples.data',samples)
    # np.savetxt('char_responses.data',responses)

