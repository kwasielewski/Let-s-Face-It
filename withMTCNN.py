from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import tensorflow as tf

plotSwitch = False

facesPath = "./faces"
objectsPath = "./objects"
faceSamples = 1000
objectSamples = 1000
facesCorrect = 0
objectsCorrect = 0

faceFiles = os.listdir(facesPath)
random.shuffle(faceFiles)

objectFiles = os.listdir(objectsPath)
random.shuffle(objectFiles)

detector = MTCNN()
for i, path in enumerate(faceFiles):
    if i%50 == 0:
        print(i, facesCorrect)
    if i== faceSamples:
        break
    img = cv2.imread(os.path.join(facesPath, path))

    img = cv2.resize(img, (64, 64))
    #print(img.shape)
    answer = detector.detect_faces(img)
    if(len(answer)==1):
        facesCorrect += 1

for i, path in enumerate(objectFiles):
    if i%50 == 0:
        print(i, objectsCorrect)
    if i== objectSamples:
        break
    img = cv2.imread(os.path.join(objectsPath, path))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    
    answer = detector.detect_faces(img)

    if(len(answer)==0):
        objectsCorrect += 1

print(facesCorrect/faceSamples)
print(objectsCorrect/objectSamples)