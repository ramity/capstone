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
    worstCasePath = "./input/2017122800008FWR3SV44.TIF"
    loadingErrorCasePath = "./input/085a17f2-f5c4-411d-bbcf-8e7303498fa.tif"

    #print(os.path.exists(loadingErrorCasePath))
    #print(os.path.isfile(loadingErrorCasePath))

    image = cv2.imread(lightFontCasePath)

    if image is None:
        sys.exit("Image was not loaded properly")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./output/gray.jpg", gray)

    height, width = gray.shape

    median = cv2.medianBlur(gray, 9)
    cv2.imwrite("./output/median.jpg", median)

    invert = cv2.bitwise_not(median)
    cv2.imwrite("./output/invert.jpg", invert)

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(invert, kernel, iterations = 1)
    cv2.imwrite("./output/medianDilate.jpg", dilate)

    blank = np.zeros((height, width, 3), np.uint8)
    fakeImage, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursImage = cv2.drawContours(blank, contours, -1, (0, 255, 0), 1)
    cv2.imwrite("./output/medianDilateContours.jpg", contoursImage)

    heightSum = 0;
    widthSum = 0;

    for contour in contours:
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        heightSum += h
        widthSum += w

    avgHeight = heightSum / len(contours)
    avgWidth = widthSum / len(contours)

    h, w = gray.shape[:2]

    boxes = np.zeros((height, width, 3), np.uint8)
    dots = np.zeros((height, width, 3), np.uint8)
    mask = np.zeros((height, width), np.uint8)
    #boxes = cv2.cvtColor(boxes, cv2.COLOR_GRAY2BGR)
    lines = gray.copy()
    lines = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)

    id = 0
    hierarchy = hierarchy[0]

    for contour in contours:

        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x,y,w,h = cv2.boundingRect(approx)

        padding = 5

        #if h <= avgHeight and w <= avgWidth:
        #    color = (255, 255, 0)
        #else:
        #    color = (0, 0, 255)

        color = (255, 0, 0)

        #area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00']) if M['m00'] else 0
        cy = int(M['m01']/M['m00']) if M['m00'] else 0

        if hierarchy[id][3] < 0:

            if cx != 0 and cy != 0:
                cv2.circle(dots, (cx, cy), 2, color, -1)

            mask = cv2.rectangle(mask, (x - padding, y - padding), (x + w + padding, y + h + padding), 255, -1)
            boxes = cv2.rectangle(boxes,(x - padding, y - padding), (x + w + padding, y + h + padding), color, 1)
            lines = cv2.line(lines, (x - padding, y + h + padding), (x + w + padding, y + h + padding), color, 1)

        id += 1

    cv2.imwrite("./output/mask.jpg", mask)
    cv2.imwrite("./output/dots.jpg", dots)
    cv2.imwrite("./output/medianDilateContoursEdgesBoxes.jpg", boxes)
    cv2.imwrite("./output/medianDilateContoursEdgesLines.jpg", lines)

    masked = mask - gray
    cv2.imwrite("./output/masked.jpg", masked)

    blur = cv2.GaussianBlur(masked, (7, 7), 0)
    cv2.imwrite("./output/blur.jpg", blur)

    maskedInvert = 255 - masked
    cv2.imwrite("./output/maskedInvert.jpg", maskedInvert)

    data = []

    output = pytesseract.image_to_data(Image.open('./output/maskedInvert.jpg'), output_type = pytesseract.Output.DICT)
    rows = len(output["level"])
    final = np.zeros((height, width, 3), np.uint8)

    for z in range(0, rows):

        level = output["level"][z]
        pageNumber = output["page_num"][z]
        blockNumber = output["block_num"][z]
        paragraphNumber = output["par_num"][z]
        lineNumber = output["line_num"][z]
        wordNumber = output["word_num"][z]
        left = output["left"][z]
        top = output["top"][z]
        width = output["width"][z]
        height = output["height"][z]
        confidence = output["conf"][z]
        text = output["text"][z]

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)
        fontScale = 0.5
        lineType = 2

        cv2.putText(final, text, (left, top), font, fontScale, color, lineType)


    cv2.imwrite("./output/final.jpg", final)

    sys.exit(0)

    print(output)
    print(len(output["level"]))
    print(len(output["page_num"]))
    print(len(output["block_num"]))
    print(len(output["par_num"]))
    print(len(output["line_num"]))
    print(len(output["word_num"]))
    print(len(output["left"]))
    print(len(output["top"]))
    print(len(output["width"]))
    print(len(output["height"]))
    print(len(output["conf"]))
    print(len(output["text"]))

    sys.exit(0)

    outputSize = len(output)
    columns = 12

    print(outputSize)
    print(outputSize / columns)

    print(output.expandtabs())

    sys.exit(0)

    output = pytesseract.run_and_get_output(Image.open('./output/masked.jpg'), 'txt', lang='spa', config='', nice=0)
    output = output.split("\t")
    print(output)
    output = output.split("\t")

    sys.exit(0)

    image = cv2.imread("./input/2017122800008FWR3SV44.TIF")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.blur(gray, (5, 5))
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    ret3, otsu = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    dilate = cv2.dilate(gray, kernel, iterations = 1)
    erode = cv2.erode(dilate, kernel, iterations = 1)

    ret,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    ret,trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    ret,tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

    height, width = gray.shape
    contoursImage = np.zeros((height, width, 3), np.uint8)
    fakeImage, contours, hierarchy = cv2.findContours(median,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contoursImage = cv2.drawContours(contoursImage, contours, -1, (0,255,0), 1)

    cv2.imwrite("./output/gray.jpg", gray)
    cv2.imwrite("./output/median.jpg", median)
    cv2.imwrite("./output/gaussian.jpg", gaussian)
    cv2.imwrite("./output/blur.jpg", blur)
    cv2.imwrite("./output/bilateral.jpg", bilateral)
    cv2.imwrite("./output/otsu.jpg", otsu)
    cv2.imwrite("./output/binary.jpg", binary)
    cv2.imwrite("./output/trunc.jpg", trunc)
    cv2.imwrite("./output/tozero.jpg", tozero)
    cv2.imwrite("./output/dilate.jpg", dilate)
    cv2.imwrite("./output/erode.jpg", erode)
    cv2.imwrite("./output/contours.jpg", contoursImage)

    sys.exit(0)

    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

    #invert = cv2.bitwise_not(threshold)

    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #dilate = cv2.dilate(invert, kernel, iterations = 1)

    height, width = dilate.shape

    mask = np.zeros((height + 2, width + 2), np.uint8)

    flood = threshold.copy()

    maxPoint = (-1, -1)
    maxArea = -1

    for y in range(0, height):
        for x in range(0, width):
            if flood[y][x] >= 128:

                mask[:] = 0

                area = cv2.floodFill(flood, mask, (x, y), 64)[0]

                if area > maxArea:
                    maxArea = area
                    maxPoint = (x, y)

    mask[:] = 0

    cv2.floodFill(flood, mask, maxPoint, 255)

    boxes = flood.copy()

    for y in range(0, height):
        for x in range(0, width):
            if boxes[y][x] == 64:
                boxes[y][x] = 0

    fakeImage, contours, hierarchy = cv2.findContours(boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contoursImage = np.zeros((height, width, 1), np.uint8)
    contoursImage = cv2.drawContours(contoursImage, contours, -1, (255, 255, 255), 5)

    corners = cv2.cornerHarris(np.float32(contoursImage), 2, 3, 0.04)

    cv2.imwrite("2.jpg", gray)
    cv2.imwrite("3.jpg", blur)
    cv2.imwrite("4.jpg", threshold)
    cv2.imwrite("5.jpg", invert)
    cv2.imwrite("6.jpg", dilate)
    cv2.imwrite("7.jpg", flood)
    cv2.imwrite("8.jpg", boxes)
    cv2.imwrite("9.jpg", contoursImage)
    cv2.imwrite("10.jpg", corners)

    #print(pytesseract.image_to_string(image).encode("utf-8"))


main()
