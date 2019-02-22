from PIL import Image
import pytesseract

import cv2
import numpy as np

import time
import sys
import math
import os

import scipy.cluster.hierarchy as hcluster
from sklearn import metrics

from matplotlib import pyplot as plt

def main():

    lightFontCasePath = "./input/2018022200072NAVWHN20.TIF"
    verA = "./input/2018022200120NAVWHN10.TIF"
    verC = "./input/2018022200074NAVWHN30.TIF"
    worstCasePath = "./input/2017122800008FWR3SV44.TIF"
    loadingErrorCasePath = "./input/085a17f2-f5c4-411d-bbcf-8e7303498fa.tif"

    image = cv2.imread(worstCasePath)

    if image is None:
        sys.exit("Image was not loaded properly")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./output/gray.jpg", gray)

    height, width = gray.shape
    dotMatrix = np.zeros((height, width), np.uint8)
    max = 0

    for z in range(0, 20):

        if z % 2 == 1:

            blur = cv2.medianBlur(gray, z)
            cv2.imwrite("./output/blur" + str(z) + ".jpg", blur)
            fakeImage, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                for coord in contour:
                    x = coord[0][0]
                    y = coord[0][1]
                    dotMatrix[y][x] += 1
                    temp = dotMatrix[y][x]

                    if temp > max:
                        max = temp

    lotsOfContours = np.zeros((height, width), np.uint8)
    graph = []

    for cy, row in enumerate(dotMatrix):
        for cx, value in enumerate(row):
            if value > 0:
                color = int(int(255 / max) * value)
                lotsOfContours[cy][cx] = color
                graph.append([cx, cy])

    cv2.imwrite("./output/lotsOfContours.jpg", lotsOfContours)

    thresh = 1.5
    clusters = hcluster.fclusterdata(graph, thresh, criterion="distance")

    plt.scatter(*np.transpose(graph), c=clusters)
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
    plt.title(title)
    plt.show()

    cv2.imwrite("./output/clusters.jpg", clusters)

    sys.exit(0)

    lotsOfContours = cv2.cvtColor(lotsOfContours, cv2.COLOR_BGR2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(lotsOfContours)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    keypointsImage = cv2.drawKeypoints(lotsOfContours, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite("./output/keypointsImage.jpg", keypointsImage)

    finalContours = np.zeros((height, width), np.uint8)
    fakeImage, contours, hierarchy = cv2.findContours(lotsOfContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = cv2.drawContours(finalContours, contours, -1, 255, 1)

    cv2.imwrite("./output/finalContours.jpg", finalContours)

    finalBoxes = np.zeros((height, width), np.uint8)

    for id, contour in enumerate(contours):
        if hierarchy[0][id][3] < 2:
            x,y,w,h = cv2.boundingRect(contour)
            finalBoxes = cv2.rectangle(finalBoxes, (x, y), (x + w, y + h), 255, 1)

    cv2.imwrite("./output/finalBoxes.jpg", finalBoxes)

main()
