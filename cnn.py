import numpy as np
from keras.utils import np_utils
import random
import os
import cv2
import matplotlib.pyplot as plt
from scipy import signal

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

class Activation(Layer):
    def __init__(self, activation, activation_der):
        self.activation = activation
        self.activation_der = activation_der

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_der(self.input))

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_der(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_der)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_der(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)



def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_der, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_der(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

facesPath = "./faces"
objectsPath = "./objects"

facePhotos = os.listdir(facesPath)
random.shuffle(facePhotos)
facePhotos = facePhotos[1:1001]
facePhotos = [facesPath+'/' + photo for photo in facePhotos]
facePhotos = np.array([cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE),(64,64)) for image in facePhotos], dtype=np.float64)


objectsPhotos = os.listdir(objectsPath)
random.shuffle(objectsPhotos)
objectsPhotos = objectsPhotos[1:1001]
objectsPhotos = [objectsPath+'/' + photo for photo in objectsPhotos]
objectsPhotos= np.array([cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE),(64,64)) for image in objectsPhotos], dtype=np.float64)

print(facePhotos.shape)
print(objectsPhotos.shape)
facePhotos = facePhotos/255
objectsPhotos = objectsPhotos/255
meanFace = facePhotos.mean(axis=0)

facePhotos = facePhotos - meanFace
objectsPhotos = objectsPhotos - meanFace

facesLabels = np.full(facePhotos.shape[0], 1)
objectsLabels = np.full(objectsPhotos.shape[0], 0)

photos = np.concatenate((facePhotos, objectsPhotos))
labels = np.concatenate((facesLabels, objectsLabels))
photos = photos.reshape(photos.shape[0], 1, 64, 64)
print(photos.shape)
print(labels.shape)
print(np.max(photos))

perm = np.random.permutation(photos.shape[0])

photos = photos[perm]
labels = labels[perm]
labels = np_utils.to_categorical(labels)
labels = labels.reshape(labels.shape[0], 2, 1)


x_train, y_train = photos[:1000], labels[:1000] 
x_test, y_test = photos[1001:2000], labels[1001:2000]


# neural network
network = [
    Convolutional((1, 64, 64), 3, 5),
    Sigmoid(),
    Convolutional((5, 62, 62), 3, 5),
    Sigmoid(),
    Reshape((5, 60, 60), (5 * 60 * 60, 1)),
    Dense(5 * 60 * 60, 100),
    Sigmoid(),
    Dense(100, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

# train

train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_der,
    x_train,
    y_train,
    epochs=10,
    learning_rate=0.1
)




# test
correct = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    if np.argmax(output) == np.argmax(y):
        correct += 1
    #print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

print(correct)
print(correct/x_train.shape[0])