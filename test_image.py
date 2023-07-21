import cv2 as cv
import numpy as np
from Countours import CLAHE

def HSV_mask(img, blur = 5, minb = 3, ming = 20, minr = 20):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv, (blur,blur))
    mask = cv.inRange(hsv, (minb, ming, minr), (255, 255, 255))
    return mask

img = cv.imread('120.png')
cv.namedWindow('orig',cv.WINDOW_NORMAL)
cv.imshow('orig', img)
spisok = []

# cap = cv.VideoCapture('Road.mp4')
# width = cap.get(3)
# height = cap.get(4)
# print(width, height)
# spisok = []
while True:
    # flag, img = cap.read()

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # рисуем прямоугольник
    # square = cv.rectangle(img.copy(), (1920, 0), (700, 570), 0, -1)
    # img = cv.bitwise_xor(img, square)
    # img = cv.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    # img = CLAHE(img, 3.)
    mask = HSV_mask(img, 8, 45, 106, 28)
    result = cv.bitwise_and(img, img, mask=mask)
    cv.namedWindow('result', cv.WINDOW_NORMAL)
    cv.imshow('result', result)

    contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sort = spisok

    for i in range(len(contours)):
        sq = cv.contourArea(contours[i])
        per = cv.arcLength(contours[i], True) ** 2
        c = sq / (per + 0.0000001)
        if (0.040 < c < 0.10) or (0.058 < c < 0.065) or (0.010 < c < 0.0183):
    #     if 100 < area:  # можно регулировать дальность
            sort.append(contours[i])
    image = img.copy()
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.drawContours(image, sort, -1, (0, 232, 44), -1, cv.LINE_AA, hierarchy, 0)
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', image)

    # перебираем все найденные контуры в цикле
    for cnt in contours:
        rect = cv.minAreaRect(cnt) # пытаемся вписать прямоугольник
        print(rect)
        box = cv.boxPoints(rect) # поиск четырех вершин прямоугольника
        box = np.int0(box) # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        if area > 500:
            cv.drawContours(image, [box], 0, (255, 0, 0), 2)
    cv.namedWindow('contours', cv.WINDOW_NORMAL)
    cv.imshow('contours', image) # вывод обработанного кадра в окно

    # (x, y, w, h) = cv.boundingRect(sort[0])
    # cv.rectangle(image, (x,y), (x + w +20, y + h + 20),(255, 255, 0), 2)
    # detect = img[y:y + h, x:x + w]
    # cv.namedWindow('IMG', cv.WINDOW_NORMAL)
    # cv.imshow('IMG', detect)
    #
    # for (x, y, w, h) in img:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # cv.imshow('IMG', img)