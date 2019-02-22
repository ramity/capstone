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

    lightFontCasePath = "./input/2018022200072NAVWHN20.TIF"
    verA = "./input/2018022200120NAVWHN10.TIF"
    verC = "./input/2018022200074NAVWHN30.TIF"
    worstCasePath = "./input/2017122800008FWR3SV44.TIF"
    loadingErrorCasePath = "./input/085a17f2-f5c4-411d-bbcf-8e7303498fa.tif"

    image = cv2.imread(verC)

    if image is None:
        sys.exit("image was not loaded properly")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    blank = np.zeros((height, width, 3), np.uint8)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 5)
    sobel = np.uint8(np.absolute(sobelx) + np.absolute(sobely))

    cv2.imwrite("./output/laplacian.jpg", laplacian)
    cv2.imwrite("./output/sobelx.jpg", sobelx)
    cv2.imwrite("./output/sobely.jpg", sobely)
    cv2.imwrite("./output/sobel.jpg", sobel)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closing = cv2.morphologyEx(sobely, cv2.MORPH_CLOSE, element)
    cv2.imwrite("./output/closing.jpg", closing)

    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    fakeImage, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = blank.copy()

    hierarchy = hierarchy[0]

    for id, contour in enumerate(contours):

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)

        if area > 100:

            x,y,w,h = cv2.boundingRect(contour)
            output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("./output/output.jpg", output)
main()
