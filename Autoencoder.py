import os
import keras
import cv2
import numpy as np
from keras.models import load_model
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from keras.preprocessing import image
import sys

# path ="C:/Users/Sergey/Desktop/Diplom/Preprocessing/Znaki1/Test"

"""Загрузка модели сверточного автоэнкодера"""
def Autoencoder():
    autoencoder = load_model('Preprocessing/Enc_CNN.h5', compile=False)
    autoencoder.compile()
    autoencoder.summary()
    return autoencoder.layers[1]

def KN(encoder):
    codes = np.load('C:/Users/Sergey/Desktop/Diplom/Preprocessing/codes')
    images = np.load('C:/Users/Sergey/Desktop/Diplom/Preprocessing/array2encoder')

    nei_clf = NearestNeighbors(metric="cosine")
    nei_clf.fit(codes)
    return nei_clf, images


def get_similar(encoder, nei_clf, images, image, n_neighbors = 3):
    assert image.ndim == 3, "image must be [batch,height,width,3]"
    code = encoder.predict(image[None])
    (distances,), (idx,) = nei_clf.kneighbors(code, n_neighbors=n_neighbors)
    return distances, images[idx]

def show_similar(encoder, nei_clf, images,image):

    distances, neighbors = get_similar(encoder, nei_clf, images, image, n_neighbors = 4)
    plt.figure(figsize=[8, 7])
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original image")

    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.imshow(neighbors[i])
        plt.title("Dist=%.3f" % distances[i])
    plt.show()

def Processing_img(img):
    img2 = cv2.resize(img, (224, 224))
    image_array = np.array(img2)
    image_array = np.array(image_array)
    image_array = image_array.astype('float32')
    image_array /= 255
    return image_array


def main(argv):
    # encoder = Autoencoder()
    # print(encoder.summary())
    #
    # nei_clf, images = KN(encoder)
    img = cv2.imread('C:\\Users\\Sergey\\Desktop\\Diplom\\Preprocessing\\Znaki1\\Test\\00105.png')
    img = cv2.resize(img, (12,12))
    cv2.namedWindow('aaa', cv2.WINDOW_NORMAL)
    cv2.imshow('aaa', img)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # image_array = np.array(img2)
    # image_array = np.array(image_array)
    # image_array = image_array.astype('float32')
    # image_array /= 255
    #
    # distances, neighbors = get_similar(encoder, nei_clf, images, image_array, n_neighbors=4)
    # print(distances[0])
    # show_similar(encoder, nei_clf, images,image_array)

    # # open a binary file in write mode
    # file = open("C:/Users/Sergey/Desktop/Diplom/Preprocessing/arr", "wb")
    # np.save(file, image_array)
    # file.close

if __name__=='__main__':
    main(sys.argv)