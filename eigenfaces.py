import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random

plotSwitch = False

path = './faces'
allPhotos = os.listdir(path)
random.shuffle(allPhotos)

def magic(image):
    mode = 0
    if mode == 0:
        return image
    elif mode == 1:
        return cv2.GaussianBlur(image, (1,1), 0)
    elif mode == 2:
        tmp = image.astype(np.uint8)
        return cv2.Canny(tmp, 100, 200)
    return  image

temp_photos = allPhotos[1:1001]
temp_images = [path+'/' + photo for photo in temp_photos]
images = np.array([magic(cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE),(64,64))) for image in temp_images], dtype=np.float64)


n_samples, h, w = images.shape
print(images.shape)
h = 64
w = 64

def pca(X, n_pc):
    n_samples, n_features = X.shape

    #plt.imshow(mean.reshape(64,64), cmap=plt.cm.gray)
    #plt.show()
    centered_data = X - mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:, :n_pc]*S[:n_pc]

    return projected, components, mean, centered_data

n_components = 25
X = images.reshape(n_samples, h*w)
print(X.shape)
mean = np.mean(X, axis=0)
meanColors = np.mean(X, axis=1)
avgColor = np.mean(meanColors)

#for i, x in enumerate(X):
#    X[i] = avgColor*(x/meanColors[i])
#X = avgColor*(X/meanColors)
#mean = np.mean(X, axis=0)

P, C, M, Y = pca(X, n_pc=n_components)
eigenfaces = C.reshape((n_components, h, w))
print(eigenfaces.shape)

distMat = np.eye(64*64) - np.dot(C.T, C)
test_img_vec = images[0].reshape(-1)

delta = np.dot(distMat, test_img_vec)
print(np.sqrt(np.dot(delta.T, delta)))

facesPath = "./faces"
objectsPath = "./objects"
faceSamples = 1000
objectSamples = 1000
facesCorrect = 0
objectsCorrect = 0
epsilon = 0.47

posEps = []
faceFiles = allPhotos[1001:1001+faceSamples]
for i, path in enumerate(faceFiles):
    if i%50 == 0:
        print(i)
    img = cv2.imread(os.path.join(facesPath, path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = magic(img)
    img_vec = img.reshape(-1)

    img_vec = img_vec - M
    img = img_vec.reshape(64,64)
    delta = np.dot(distMat, img_vec)
    delta = np.sqrt(np.dot(delta.T, delta))
    posEps.append(delta)
    if delta < epsilon:
        facesCorrect += 1

negEps=[]
objectFiles = os.listdir(objectsPath)
random.shuffle(objectFiles)
for i, path in enumerate(objectFiles):
    if i%50 == 0:
        print(i)
    if i== objectSamples:
        break
    img = cv2.imread(os.path.join(objectsPath, path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = magic(img)
    img_vec = img.reshape(-1)

    img_vec = img_vec - M
    img = img_vec.reshape(64,64)
    delta = np.dot(distMat, img_vec)
    delta = np.sqrt(np.dot(delta.T, delta))
    negEps.append(delta)
    if delta >= epsilon:
        objectsCorrect += 1


if plotSwitch:
    plt.scatter(posEps, posEps, c="blue")
    plt.show()
    plt.scatter(negEps, negEps,  c="red")
    plt.show()

optParam = 100
optVal = 0
param = 1000
difference = 100

for j in range(30):
    print(param)
    print(len([i for i in posEps if i < param])/faceSamples)
    print(len([i for i in negEps if i >= param])/objectSamples)
    print()
    curdif = abs(len([i for i in posEps if i < param])/faceSamples - len([i for i in negEps if i >= param])/objectSamples)
    if curdif < difference:
        optParam = param
        difference = curdif
    param += 100

print(optParam)

facesCorrect = 0
objectsCorrect = 0
epsilon = 0.47

posEps = []
faceFiles = allPhotos[2001:2001+faceSamples]
for i, path in enumerate(faceFiles):
    if i%50 == 0:
        print(i)
    img = cv2.imread(os.path.join(facesPath, path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = magic(img)
    img_vec = img.reshape(-1)
    #print(np.min(img_vec))

    img_vec = img_vec - M
    img = img_vec.reshape(64,64)
    delta = np.dot(distMat, img_vec)
    delta = np.sqrt(np.dot(delta.T, delta))
    if delta < optParam:
        facesCorrect += 1

negEps=[]
objectFiles = objectFiles[1001:1001+objectSamples]
random.shuffle(objectFiles)
for i, path in enumerate(objectFiles):
    if i%50 == 0:
        print(i)
    if i== objectSamples:
        break
    img = cv2.imread(os.path.join(objectsPath, path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (64, 64))
    
    img = magic(img)
    img_vec = img.reshape(-1)

    img_vec = img_vec - M
    img = img_vec.reshape(64,64)
    delta = np.dot(distMat, img_vec)
    delta = np.sqrt(np.dot(delta.T, delta))

    if delta >= optParam:
        objectsCorrect += 1

print("results")
print(facesCorrect/faceSamples)
print(objectsCorrect/objectSamples)