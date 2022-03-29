# import matplotlib.pyplot as plt

# img = plt.imread('orig.png')
# rows,cols,colors = img.shape # gives dimensions for RGB array
# img_size = rows*cols*colors
# img_1D_vector = img.reshape(img_size)
# # you can recover the orginal image with:
# img2 = img_1D_vector.reshape(rows,cols,colors)

# plt.imshow(img) # followed by 
# plt.show() # to show the first image, then 
# plt.imshow(img2) # followed by
# plt.show() # to show you the second image.

from msilib.schema import Directory
from pickle import LONG4
from pickletools import long4
import pandas as pd
import os
import math
import numpy as np
import cv2
try:
    import Image
except ImportError:
    from PIL import Image

dayFolder = os.listdir('day')
nightFolder = os.listdir('night')

n_day_image = len(dayFolder)
n_night_image = len(nightFolder)

nightVector = []
dayVector = []

for _ in range(0, n_day_image):
    day = 'day/'+dayFolder[_]
    night = 'night/'+nightFolder[_]
    dayImg = Image.open(day)
    dayImg = dayImg.resize((200,200))
    dayImageSequence = dayImg.getdata()
    imageArray = np.array(dayImageSequence)
    #print(imageArray)
    imageVector = imageArray.flatten()
    #print(imageArray)
    dayVector.append(imageVector)

    nightImg = Image.open(night)
    nightImg = nightImg.resize((200,200))
    nightImgSequence = nightImg.getdata()
    imageArray = np.array(nightImgSequence)
    imageVector = imageArray.flatten()
    nightVector.append(imageVector)

testImg = Image.open('test1.png')
testImg = testImg.resize((200,200))
testImgSequence = testImg.getdata()
testImgArray = np.array(testImgSequence)
testImgVector = testImgArray.flatten()

# for _ in range (0, n_day_image):
#    print(len(dayVector[_]))

# for _ in range (0, n_day_image):
#     print(len(nightVector[_]))

distanceWithday =  []
distanceWithnight = []

for i in range (0, len(dayVector)):
    distance = 111111111111111111111111111111111111111111111111111111111111111111
    for j in range (min(len(testImgVector), len(dayVector[i]))):
        distance += (testImgVector[j] - dayVector[i][j])*(testImgVector[j] - dayVector[i][j])
        #if(j==0):
            #distance -= 111111111111111111111111111111111111111111111111111111111111111111
        #print(distance)
    distance -= 111111111111111111111111111111111111111111111111111111111111111111
    #print(distance)
    distance = math.sqrt(distance)
    distanceWithday.append(distance)

for i in range (0, len(nightVector)):
    distance = 111111111111111111111111111111111111111111111111111111111111111111
    for j in range (min(len(testImgVector), len(nightVector[i]))):
        distance += (testImgVector[j] - nightVector[i][j])*(testImgVector[j] - nightVector[i][j])
    distance -= 111111111111111111111111111111111111111111111111111111111111111111
    #print(distance)
    distance = math.sqrt(distance)
    distanceWithnight.append(distance)

distanceWithday.sort()
distanceWithnight.sort()

k = int(input("enter the value for k: "))
i=0
j=0
dayCount = 0
nightCount = 0

for itr in range(0,k):
    if(distanceWithday[i]< distanceWithnight[j]):
        dayCount += 1
        i += 1
    else:
        nightCount +=1
        j+=1

if nightCount >dayCount:
    print('input image is night')
else:
    print("input image is day")

    





