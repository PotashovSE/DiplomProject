import os
import sys

import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
import random

from CNN import Upload_model, Signs_Recognition
from Autoencoder import Autoencoder, KN, get_similar, Processing_img


def HSV_mask(img, blur = 5, minb = 3, ming = 20, minr = 20):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv, (blur,blur))
    mask = cv.inRange(hsv, (minb, ming, minr), (255, 255, 255))
    return mask


def CLAHE(img, contast = 3.):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=contast, tileGridSize=(12, 12))

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv.merge((l2, a, b))  # merge channels
    img2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    return img2

def Hist(img):
    buff = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    Y,Cr,Cb = cv.split(buff)
    new_img = cv.equalizeHist(Cr)
    return new_img

def Detection(img2, minV=110, maxV=110):
    detected_edges = cv.Canny(img2, minV, maxV * 4, 3)
    # cv.namedWindow('Canny', cv.WINDOW_NORMAL)
    # cv.imshow('Canny', detected_edges)
    countours, hierarchy = cv.findContours(detected_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(detected_edges.shape, dtype=np.uint8)
    for i in range(len(countours)):
        sq = cv.contourArea(countours[i])
        per = cv.arcLength(countours[i], True)**2
        c = sq / (per + 0.0000001)
        # print(c)
        if 200< sq < 1750:  # можно регулировать дальность
            if (0.060 < c < 0.10) or (0.05 < c < 0.065) or (0.02 < c < 0.055):
                drawing = cv.drawContours(drawing, countours, i, (255,255,255), 1, cv.LINE_AA, hierarchy, 0)
    return drawing

def Accuracy(cnt_sign, cnt_all):
    cnt1 = int(input('Сколько знаков в кадре: '))
    cnt_all += cnt1
    cnt2 = int(input('Сколько знаков нашел алгоритм: '))
    cnt_sign += cnt2
    return cnt_all, cnt_sign

def Path(katalog = 'C:\\Users\\Sergey\\Desktop\\Diplom\Detect\\TrainIJCNN2013\\TrainIJCNN2013\\'):
    files = os.listdir(katalog)
    spisok = []
    for file in files:
        if file[-4:] == '.png':
            spisok.append(katalog + file)
    return spisok

# Загрузка Сверточного автокодировщика и прогон через обученные данные для промежуточной проверки знака(Знак / Не знак)
encoder = Autoencoder()
print(encoder.summary())
nei_clf, images = KN(encoder)

# Загрузка обученной модели CNN для классификации дорожных знаков
model = Upload_model()


def main(argv):
    cnt_sign = 0 #Кол-во задетектированных знаков
    cnt_all = 0  # Общее кол-во задетектированных областей

    file = Path('C:\\Users\\Sergey\\Desktop\\Diplom\\Detect\\TestIJCNN2013\\TestIJCNN2013Download\\')
    print(file)
    for i in range(0,299,3):
        img = cv.imread(file[i])

        img2 = CLAHE(img, 3.)

        # img2 = Hist(img2)
        # cv.imshow('Hist',img2)

        result = Detection(img2, 110, 110)

        countours, hierarchy = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for cnt in countours:
            (x, y, w, h) = cv.boundingRect(cnt)
            crop_img = img[y-5:y + h+5, x - 5:x + w + 5] # вырезание найденных областей

            # Промежуточная проверка выделенной области на наличие в ней знака
            proverka_znaka = Processing_img(crop_img.copy())
            distances, neighbors = get_similar(encoder, nei_clf, images, proverka_znaka)
            print(distances[0])
            if distances[0] < 0.3:
                print('Good')
                # Распознавание знака
                sign_type = Signs_Recognition(crop_img, model)

                # Отрисовка Bounding Box-ов и подписей вида знака
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                if len(sign_type) <= 25:
                    cv.rectangle(img, (x + w, y), (x + w + 280, y - 30), (255, 0, 0), -1)
                    cv.putText(img, sign_type, (x + w, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                else:
                    cv.rectangle(img, (x + w, y), (x + w + 600, y - 30), (255, 0, 0), -1)
                    cv.putText(img, sign_type, (x + w, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

            # cv.namedWindow(f'{i}', cv.WINDOW_NORMAL)
            # cv.imshow(f'{i}', detect)  # вывод обработанного кадра в окно
            # cv.waitKey(0)
            cnt_all += 1
        print(cnt_all)
        cv.namedWindow(f'contours{i}', cv.WINDOW_NORMAL)
        cv.imshow(f'contours{i}', img) # вывод обработанного кадра в окно

        while True:
            # flag, img = cap.read()
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                # cnt_sign = Accuracy(cnt_sign)
                # print(cnt_sign)
                break
            # cv.imshow('2', mser(result2))
    # accuracy = cnt_sign / cnt_all
    # print(f'accuracy: {accuracy}')
if __name__=='__main__':
    main(sys.argv)