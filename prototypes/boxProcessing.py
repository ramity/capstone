import numpy as np
import cv2
import sys
from PIL import Image

lightFontCasePath = "./input/2018022200072NAVWHN20.TIF"
verA = "./input/2018022200120NAVWHN10.TIF"
verC = "./input/2018022200074NAVWHN30.TIF"
worstCasePath = "./input/2017122800008FWR3SV44.TIF"
loadingErrorCasePath = "./input/085a17f2-f5c4-411d-bbcf-8e7303498fa.tif"

image = cv2.imread(worstCasePath)
height, width, channels = image.shape
blankGray = np.zeros((height, width), np.uint8)
blankColor = np.zeros((height, width, 3), np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

data = {}
z = 0

for i in range(3, 19, 2):
    data[z] = []

    blur = cv2.medianBlur(gray, i)
    cv2.imwrite("./output/blur-" + str(i) + ".jpg", blur)

    fakeImage, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursImage = cv2.drawContours(blankColor.copy(), contours, -1, (0, 255, 0), 1)
    cv2.imwrite("./output/contours-" + str(i) + ".jpg", contoursImage)

    boxes = blankColor.copy()
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
        data[z].append([x, y, w, h])
    cv2.imwrite("./output/boxes-" + str(i) + ".jpg", boxes)
    z += 1

for z in range(0, z - 1):
    data[z] = sorted(data[z], key=lambda l: (l[0], l[1]))

for z in range(0, z - 1):
    for contour in data[z]:
        x = contour[0]
        y = contour[1]
        w = contour[2]
        h = contour[3]

        potentialSiblings = []

        for searchContour in data[z]:
            sx = searchContour[0]
            sy = searchContour[1]
            sw = searchContour[2]
            sh = searchContour[3]

            if sx > x + w and x + 2*w > sx and sy > y and y + h > sy:
                potentialSiblings.append(searchContour)

        for potentialSibling in potentialSiblings:
            sx = potentialSibling[0]
            sy = potentialSibling[1]
            sw = potentialSibling[2]
            sh = potentialSibling[3]

            # if potentialSibling is close enough to be a sibling, merge the two contours into a single contour
