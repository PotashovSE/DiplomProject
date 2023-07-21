import cv2 as cv
import numpy as np
from CNN import Upload_model, Signs_Recognition
import time

# Импорт обученных каскадов для детектирования знаков
sign_cascade_circle = cv.CascadeClassifier('Haar\\circle.xml')
sign_cascade_triangle = cv.CascadeClassifier('Haar\\triangle.xml')

# Загрузка обученной модели CNN для классификации дорожных знаков
model = Upload_model()

# инициализировать видео
cap = cv.VideoCapture('Road_Trim.mp4')
width = cap.get(3)
height = cap.get(4)
print(width, height)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    flag, img = cap.read()
    square = cv.rectangle(img.copy(), (1800, 200), (700, 600), 0, -1)

    img2 = cv.bitwise_xor(img, square)

    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    circle_signs = sign_cascade_circle.detectMultiScale(gray, 1.3, 2)
    triangle_signs = sign_cascade_triangle.detectMultiScale(gray, 1.3, 2)
    """Циклы распознавания знаков и отрисовка значимых областей"""
    for (x, y, w, h) in circle_signs:
        # Вырезание области нахождения знака и подача его на распознавание в CNN
        img3 = img2.copy()
        crop_img = img3[y - 3:y + h + 3, x - 3:x + w + 3]

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

    for (x, y, w, h) in triangle_signs:
        # Вырезание области нахождения знака и подача его на распознавание в CNN
        img3 = img.copy()
        crop_img = img3[y - 3:y + h + 3, x - 3:x + w + 3]


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

    new_frame_time = time.time()
    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    cv.putText(img, 'FPS: '+fps, (15,900), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
    square = cv.rectangle(img, (1800, 200), (700, 600), (100,100,255), 2)
    cv.namedWindow('detect', cv.WINDOW_NORMAL)
    cv.imshow('detect',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        # cnt_all, TP, FN, FP = Acc(cnt_all, TP, FN, FP)
        # print(cnt_all, TP, FN, FP)
        break

cv.destroyAllWindows()

