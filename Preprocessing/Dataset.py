import os
import pandas as pd
import numpy as np
import csv
import shutil
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io

def Path(katalog = 'C:\\Users\\Sergey\\Desktop\\Diplom\Detect\\TrainIJCNN2013\\TrainIJCNN2013\\'):
    files = os.listdir(katalog)
    spisok = []
    for file in files:
        if file[-4:] == '.png':
            spisok.append(katalog + file)
    return spisok

def image2array(image):
    img = io.imread(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))

    image_array = np.array(img)
    print(image_array.shape)
    # image_array = image_array.reshape(image_array.shape[0], 224, 224, 3)
    print(image_array.shape)
    image_array = image_array.astype('float32')
    image_array /= 255
    return np.array(image_array)

#
# file = Path('C:\\Users\\Sergey\\Desktop\\Diplom\\Detect\\TrainIJCNN2013\\TrainIJCNN2013\\')
# print(file)
#
# # Импорт обученных каскадов для детектирования знаков
# sign_cascade_circle = cv.CascadeClassifier('C:\\Users\\Sergey\\Desktop\\Diplom\\Haar\\circle.xml')
# sign_cascade_triangle = cv.CascadeClassifier('C:\\Users\\Sergey\\Desktop\\Diplom\\Haar\\triangle.xml')
#
# for i in range(0,599):
#     img = cv.imread(file[i])
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
#     # Настройка каскадов для поиска знаков на изображении
#     circle_signs = sign_cascade_circle.detectMultiScale(gray, 1.2, 1)
#     triangle_signs = sign_cascade_triangle.detectMultiScale(gray, 1.2, 1)
#
#     """Циклы распознавания знаков и отрисовка значимых областей"""
#     for (x, y, w, h) in circle_signs:
#         # Вырезание области нахождения знака и подача его на распознавание в CNN
#         img2=img.copy()
#         crop_img = img2[y-3:y + h+3, x - 3:x + w + 3]
#
#         # Отрисовка Bounding Box-ов и подписей вида знака
#         cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#         print(f'{x},{y},{x+w},{y+w}')
#     for (x, y, w, h) in triangle_signs:
#         # Вырезание области нахождения знака и подача его на распознавание в CNN
#         img2=img.copy()
#         crop_img = img2[y-3:y + h+3, x - 3:x + w + 3]
#         # Отрисовка Bounding Box-ов и подписей вида знака
#         cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#         print(f'{x},{y},{x + w},{y + w}')
#
#     cv.namedWindow(f'contours{i}', cv.WINDOW_NORMAL)
#     cv.imshow(f'contours{i}', img)
#
#     while True:
#         k = cv.waitKey(1) & 0xFF
#         if k == 27:
#             # cnt_sign, TP, FN, FP, TN = Accuracy(cnt_sign, TP, FN, FP, TN)
#             # print(cnt_sign)
#             break

data = pd.read_csv('C://Users//Sergey//Desktop//Diplom//Znaki//Meta.csv')
# for i in data.ClassId:
#     if i in del_data:
#         os.remove(f'Znaki//Meta//{i}.png')
#         data = data[data.ClassId != i]
print(data)
data.to_csv('Znaki//Meta1.csv')
#
# """очистка лишних классов из файла Test.csv и файлов из папки Test(image)"""
# test = pd.read_csv('Znaki//Test.csv', delimiter=',')
# for i in del_data:
#     d = test[test.ClassId == i]
#     for j in d.index:
#         os.remove(f"Znaki//Test//{d['Path'][j][5:]}")
#     test = test[test.ClassId != i]
#
# test.to_csv('Znaki//Test1.csv', index= False)
#
# """очистка лишних классов из файла Train.csv и файлов из папки Train(папки клaссов с изображениями)"""
# train = pd.read_csv('Znaki//Train.csv', delimiter=',')
# for i in del_data:
#     shutil.rmtree(f'Znaki//Train//{i}')
#     train = train[train.ClassId != i]
# print(train)
# train.to_csv('Znaki//Train1.csv')

