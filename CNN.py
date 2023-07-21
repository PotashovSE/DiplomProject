"""Загрузка обученной CNN и написание функций для обработки входного изображения"""
import cv2 as cv
import os
import numpy as np
import tensorflow
import keras
from keras.models import load_model
from matplotlib import pyplot as plt
import sys
from PIL import Image,ImageDraw, ImageFont

class_names = [
    'Speed limit (20km/h)',#0
    'Speed limit (30km/h)',#1
    'Speed limit (50km/h)',#2
    'Speed limit (60km/h)',#3
    'Speed limit (70km/h)',#4
    'Speed limit (80km/h)',#5
    'End of speed limit (80km/h)',#6
    'Speed limit (100km/h)',#7
    'Speed limit (120km/h)',#8
    'No passing',#9
    'No passing for vechiles over 3.5 metric tons',#10
    'Right-of-way at the next intersection',#11
    'Priority road',  # 12
    'Yield',  # 13
    'Stop',  # 14
    'No vehicles',#15
    'Vechiles over 3.5 metric tons prohibited',#16
    'No entry',  # 17
    'General caution',#18
    'Dangerous curve to the left',#19
    'Dangerous curve to the right',#20
    'Double curve',#21
    'Bumpy road',#22
    'Slippery road',#23
    'Road narrows on the right',#24
    'Road work',#25
    'Traffic signals',#26
    'Pedestrians',#27
    'Children crossing',#28
    'Bicycles crossing',#29
    'Beware of ice/snow',#30
    'Wild animals crossing',#31
    'End of all speed and passing limits',  # 32
    'Turn right ahead',#33
    'Turn left ahead',#34
    'Ahead only',#35
    'Go straight or right',#36
    'Go straight or left',#37
    'Keep right',#38
    'Keep left',#39
    'Roundabout mandatory',#40
    'End of no passing',#41
    'End of no passing by vechiles over 3.5 metric tons'#42
    ]

classes = ['20', '30', '50', '60', '70', '80','Otmena 80', '100', '120', 'Obgon zapreshchen',
           'Obgon zapreshchen gruzovym', 'Perekrestok','Glavnaya doroga','Ustupi dorogu','Stop',
           'Dvizheniye zapreshcheno','Zapreshchayushchiy gruzovym', 'Tupik','!','Nalevo','Napravo','Izgib',
           'Iskustvennaya nerovnost'', '"Skol'zkaya doroga",'Suzheniye dorogi','Vedutsya remontnyye raboty',
           'Svetofor','Ne peshekhodnyy perekhod','Ostorozhno deti','Velosipedisty', '*','Zhivotnyye',
           'Otmena ogranicheniy','Povorot napravo','Povorot nalevo', 'Pryamo','Razvilka napravo','Razvilka nalevo',
           'S"yezd napravo','S"yezd nalevo', 'Krugovoye dvizheniye','Konets zony zapreshcheniya obgona',
           'Konets zony zapreshcheniya \n obgona gruzovym']


"""Загрузка обученной модели сети"""
def Upload_model():
    model = load_model('CNN_model_Signs.h5', compile=False)
    model.compile()
    model.summary()
    return model

"""Обработка входящего изображения перед подачей на распознавание"""
def Signs_Recognition(img, model = None):

    #изменение размера изображения для подачи на вход CNN
    img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img2 = cv.resize(img2, (30, 30))
    test_img = np.array(img2)

    # вывод собственного изображения
    plt.imshow(test_img, cmap=plt.get_cmap('gray'))
    plt.show()

    # предобработка
    test_img = np.reshape(test_img, (-1, 30, 30, 3))

    # распознавание
    result = (model.predict(test_img) > 0.5).astype("int32")
    print(f'I think it\'s , {class_names[np.argmax(result)]}')
    return class_names[np.argmax(result)]
"""Обработка входящего изображения перед подачей на распознавание"""
def Signs_Recognition1(img, model = None):

    #изменение размера изображения для подачи на вход CNN
    img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img2 = cv.resize(img2, (30, 30))
    test_img = np.array(img2)

    # вывод собственного изображения
    plt.imshow(test_img, cmap=plt.get_cmap('gray'))
    plt.show()

    # предобработка
    test_img = np.reshape(test_img, (-1, 30, 30, 3))

    # распознавание
    result = (model.predict(test_img) > 0.5).astype("int32")
    print(f'I think it\'s , {class_names[np.argmax(result)]}')
    return np.argmax(result)
def main(argv):
    # model = Upload_model()
    # res = Signs_Recognition('70.png',model)
    # print(res)
    img = cv.imread('C://Users//Sergey//Desktop//Diplom//Detect//TrainIJCNN2013//TrainIJCNN2013//00000.png')
    cv.rectangle(img, (774, 411), (815, 446), (255, 0, 0), 2)
    cv.imshow('orig', img)

    while True:
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            # cnt_sign = Accuracy(cnt_sign)
            # print(cnt_sign)
            break


if __name__=='__main__':
    main(sys.argv)