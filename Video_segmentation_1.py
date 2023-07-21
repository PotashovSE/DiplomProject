import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from Countours_detection_image import CLAHE, Hist, Detection, HSV_mask
from CNN import Upload_model, Signs_Recognition
import time

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# Загрузка обученной модели CNN для классификации дорожных знаков
model = Upload_model()

# инициализировать видео
cap = cv.VideoCapture('Road_Trim.mp4')
width = cap.get(3)
height = cap.get(4)
print(width, height)
box = 0

while True:
    ret, frame = cap.read()
    # рисуем прямоугольник
    square = cv.rectangle(frame.copy(), (1700, 300), (1050, 700), 0, -1)
    frame1 = cv.bitwise_xor(frame, square)
    frame1 = CLAHE(frame1, 3.)

    result = Detection(frame1, 200, 150)

    countours, hierarchy = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in countours:
        (x, y, w, h) = cv.boundingRect(cnt)
        crop_img = frame1[y - 5:y + h + 5, x - 5:x + w + 5]  # вырезание найденных областей

        # Распознавание знака
        sign_type = Signs_Recognition(crop_img, model)


        # Отрисовка Bounding Box-ов и подписей вида знака
        cv.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
        box += 1

        if len(sign_type) <= 25:
            cv.rectangle(frame, (x + w, y), (x + w + 280, y - 30), (255, 0, 0), -1)
            cv.putText(frame, sign_type, (x + w, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        else:
            cv.rectangle(frame, (x + w, y), (x + w + 600, y - 30), (255, 0, 0), -1)
            cv.putText(frame, sign_type, (x + w, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

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
    cv.putText(frame, 'FPS: '+fps, (15,900), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)

    square = cv.rectangle(frame, (1800, 250), (1050, 700), (100, 100, 255), 2)

    cv.namedWindow('contours', cv.WINDOW_NORMAL)
    cv.imshow('contours', frame) # вывод обработанного кадра в окно


    if cv.waitKey(1) == 27:
        break
print(f'box: {box}')
cap.release()
cv.destroyAllWindows()