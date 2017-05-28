import numpy as np
import cv2
from matplotlib import pyplot as plt
def open(img):
    cv2.imshow('d',img)
    cv2.waitKey(0 )
    cv2.destroyAllWindows()


img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
l = 0
for i in x:
    for j in i:
        cv2.imwrite(r'D:\Github_project\OPENCV\Test\lol\{0}.png'.format(str(l)),j)
        l+=1

# Now we prepare train_data and test_data.
train = x[:,:10].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:90].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# for i in test:
#     cv2.imwrite(r'D:\Github_project\OPENCV\Test\lol\rt.png', img)
# # Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,50)[:,np.newaxis]
test_labels = np.repeat(k,200)[:,np.newaxis]
im = cv2.imread(r'D:\Github_project\OPENCV\Test\lol\60.png')
arr = np.array(train[0])
ter = arr.reshape(-1,400).astype(np.float32)
open(ter)
# # Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=1)
print(ret)
# print(result)
# # Now we check the accuracy of classification
# # For that, compare the result with test_labels and check which are wrong

matches = result==test_labels
# print(matches)
correct = np.count_nonzero(matches)
# print(correct)
accuracy = correct*100.0/result.size
print (accuracy)
