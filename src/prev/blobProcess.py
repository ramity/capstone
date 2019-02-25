from PIL import Image
import pytesseract

import cv2
import numpy as np

import time
import sys
import math
import os

from matplotlib import pyplot as plt

def main():
    #folder = os.path.abspath("./input")
    #output = os.path.abspath("./output")
    #files = os.listdir(folder)

    #for file in files:
    #    print(file)

    lightFontCasePath = "./input/2018022200072NAVWHN20.TIF"

    verA = "./input/2018022200120NAVWHN10.TIF"
    verC = "./input/2018022200074NAVWHN30.TIF"

    worstCasePath = "./input/2017122800008FWR3SV44.TIF"
    loadingErrorCasePath = "./input/085a17f2-f5c4-411d-bbcf-8e7303498fa.tif"

    #print(os.path.exists(loadingErrorCasePath))
    #print(os.path.isfile(loadingErrorCasePath))

    image = cv2.imread(verC)

    if image is None:
        sys.exit("Image was not loaded properly")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./output/gray.jpg", gray)

    height, width = gray.shape
    blank = np.zeros((height, width, 3), np.uint8)
    final = blank.copy()
    dotMatrix = np.zeros((height, width), np.uint8)

    for z in range(0, 20):

        if z % 2 == 1:

            blur = cv2.medianBlur(gray, z)
            cv2.imwrite("./output/blur" + str(z) + ".jpg", blur)

            dots = blank.copy()
            boxes = blank.copy()
            fakeImage, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            id = 0
            hierarchy = hierarchy[0]

            for contour in contours:

                x,y,w,h = cv2.boundingRect(contour)

                padding = 5

                #if h <= avgHeight and w <= avgWidth:
                #    color = (255, 255, 0)
                #else:
                #    color = (0, 0, 255)

                color = (255, 255, 255)

                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00']) if M['m00'] else 0
                cy = int(M['m01']/M['m00']) if M['m00'] else 0

                if cx != 0 and cy != 0:
                    cv2.circle(dots, (cx, cy), 1, color, -1)
                    dotMatrix[cy][cx] += 1

                if hierarchy[id][3] < 5 and w < width * 0.9 and h < height * 0.9:
                    boxes = cv2.rectangle(boxes, (x - padding, y - padding), (x + w + padding, y + h + padding), color, -1)

                id += 1

            cv2.imwrite("./output/boxes" + str(z) + ".jpg", boxes)
            cv2.addWeighted(boxes, 0.2, final, 1 - 0.2, 0, final)
            cv2.imwrite("./output/dots" + str(z) + ".jpg", dots)

            contoursImage = blank.copy()
            contoursImage = cv2.drawContours(contoursImage, contours, -1, (255, 255, 255), 2)
            cv2.imwrite("./output/contours" + str(z) + ".jpg", contoursImage)

            fakeImage, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            boxesContours = blank.copy()

            id = 0
            hierarchy = hierarchy[0]

            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)

                padding = 5

                #if h <= avgHeight and w <= avgWidth:
                #    color = (255, 255, 0)
                #else:
                #    color = (0, 0, 255)

                color = (255, 255, 255)

                if hierarchy[id][3] < 5 and w < width * 0.9 and h < height * 0.9:
                    boxesContours = cv2.rectangle(boxesContours, (x - padding, y - padding), (x + w + padding, y + h + padding), color, -1)

                id += 1

            cv2.imwrite("./output/boxesContours" + str(z) + ".jpg", boxesContours)

    cv2.imwrite("./output/final.jpg", final)
    ret, threshold = cv2.threshold(final, 48, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./output/threshold.jpg", threshold)

    lotsOfDots = gray.copy()

    for cy, row in enumerate(dotMatrix):
        for cx, value in enumerate(row):
            if value > 0:
                if value == 1:
                    color = (25, 25, 25)
                elif value == 2:
                    color = (50, 50, 50)
                elif value == 3:
                    color = (75, 75, 75)
                elif value == 4:
                    color = (100, 100, 100)
                elif value == 5:
                    color = (125, 125, 125)
                elif value == 6:
                    color = (150, 150, 150)
                elif value == 7:
                    color = (175, 175, 175)
                elif value == 8:
                    color = (200, 200, 200)
                elif value == 9:
                    color = (225, 225, 225)
                elif value == 10:
                    color = (250, 250, 250)
                else:
                    color = (0, 0, 0)

                cv2.circle(lotsOfDots, (cx, cy), 2, color, -1)

    cv2.imwrite("./output/lotsOfDots.jpg", lotsOfDots)

main()
