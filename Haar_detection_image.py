import os
from os import path
import sys

import cv2 as cv
import numpy as np
from PIL import Image,ImageDraw, ImageFont
from matplotlib import pyplot as plt
from Countours_detection_image import Path

import keras
from keras.preprocessing import image
from keras.models import load_model
from CNN import Upload_model, Signs_Recognition
from Autoencoder import Autoencoder, KN, get_similar, Processing_img

def Accuracy(cnt_all, TP, FN, FP, TN ):
    cnt1 = int(input('Сколько знаков в кадре: '))
    cnt_all += cnt1
    cnt2 = int(input('Сколько знаков нашел алгоритм: '))
    TP += cnt2
    cnt3 = int(input('FN Знак был, но не нашелся: '))
    FN += cnt3
    cnt4 = int(input('FP Знака не было, но нашелся: '))
    FP += cnt4
    cnt5 = int(input('FP Знака не было, не нашелся: '))
    TN += cnt5
    return cnt_all, TP, FN, FP, TN

cnt_sign, TP, FN, FP, TN = 0,0,0,0,0  # Общее кол-во задетектированных областей

file = Path('C:\\Users\\Sergey\\Desktop\\Diplom\\Detect\\TestIJCNN2013\\TestIJCNN2013Download\\')
print(file)

# Импорт обученных каскадов для детектирования знаков
sign_cascade_circle = cv.CascadeClassifier('Haar\\circle.xml')
sign_cascade_triangle = cv.CascadeClassifier('Haar\\triangle.xml')

# Загрузка Сверточного автокодировщика и прогон через обученные данные для промежуточной проверки знака(Знак / Не знак)
encoder = Autoencoder()
print(encoder.summary())
nei_clf, images = KN(encoder)

# Загрузка обученной модели CNN для классификации дорожных знаков
model = Upload_model()


for i in range(0,300,3):
    print(i)
    img = cv.imread(file[i])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Настройка каскадов для поиска знаков на изображении
    circle_signs = sign_cascade_circle.detectMultiScale(gray, 1.2, 1)
    triangle_signs = sign_cascade_triangle.detectMultiScale(gray, 1.2, 1)

    """Циклы распознавания знаков и отрисовка значимых областей"""
    for (x, y, w, h) in circle_signs:
        # Вырезание области нахождения знака и подача его на распознавание в CNN
        img2=img.copy()
        crop_img = img2[y-3:y + h+3, x - 3:x + w + 3]

        # Промежуточная проверка выделенной области на наличие в ней знака
        proverka_znaka = Processing_img(crop_img.copy())
        distances, neighbors = get_similar(encoder, nei_clf, images, proverka_znaka)
        print('circle: ', distances[0])
        if distances[0] < 0.390:
            print('Good')
            # Распознавание знака
            sign_type = Signs_Recognition(crop_img, model)

            # Отрисовка Bounding Box-ов и подписей вида знака
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if len(sign_type) <= 25:
                cv.rectangle(img, (x + w , y), (x + w + 280, y - 30), (255, 0, 0), -1)
                cv.putText(img, sign_type, (x + w, y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            else:
                cv.rectangle(img, (x + w , y), (x + w + 600, y - 30), (255, 0, 0), -1)
                cv.putText(img, sign_type, (x + w, y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        # cv.namedWindow(f'cropped', cv.WINDOW_NORMAL)
        # cv.imshow("cropped", crop_img)
        # cv.waitKey(0)


    for (x, y, w, h) in triangle_signs:
        # Вырезание области нахождения знака и подача его на распознавание в CNN
        img2=img.copy()
        crop_img = img2[y-3:y + h+3, x - 3:x + w + 3]

        # Промежуточная проверка выделенной области на наличие в ней знака
        proverka_znaka = Processing_img(crop_img.copy())
        distances, neighbors = get_similar(encoder, nei_clf, images, proverka_znaka)
        print('triangle: ', distances[0])
        if distances[0] < 0.35:
            print('Good')
            # Распознавание знака
            sign_type = Signs_Recognition(crop_img, model)

            # Отрисовка Bounding Box-ов и подписей вида знака
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if len(sign_type) <= 25:
                cv.rectangle(img, (x + w, y), (x + w + 280, y - 30), (255, 0, 0), -1)
                cv.putText(img, sign_type, (x + w, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            else:
                cv.rectangle(img, (x + w, y), (x + w + 600, y - 30), (255, 0, 0), -1)
                cv.putText(img, sign_type, (x + w, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        # cv.namedWindow(f'cropped', cv.WINDOW_NORMAL)
        # cv.imshow("cropped", crop_img)
        # cv.waitKey(0)
    cv.namedWindow(f'contours{i}', cv.WINDOW_NORMAL)
    cv.imshow(f'contours{i}', img)

    while True:
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            # cnt_sign, TP, FN, FP, TN = Accuracy(cnt_sign, TP, FN, FP, TN)
            # print(cnt_sign)
            break

# Accuracy = (TP+TN)/(cnt_sign + TN)
# Precision = TP/(TP+FP)
# Recall = TP/(TP+FN)
# F1_score = (2*Precision*Recall)/(Precision + Recall)
#
# print(f'TP:{TP}   FN{FN}   FP{FP}')
# print(f'Accracy: {Accuracy} \nPrecision: {Precision} \nRecall: {Recall} \nF1-score: {F1_score}')