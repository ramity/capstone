import numpy as np
from PIL import Image
import cv2
import sys

lightFontCasePath = "./input/2018022200072NAVWHN20.TIF"
verA = "./input/2018022200120NAVWHN10.TIF"
verC = "./input/2018022200074NAVWHN30.TIF"
worstCasePath = "./input/2017122800008FWR3SV44.TIF"
loadingErrorCasePath = "./input/085a17f2-f5c4-411d-bbcf-8e7303498fa.tif"

image = cv2.imread(worstCasePath)

if image is None:
    sys.exit("Image was not loaded properly")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
blank = np.zeros((height, width, 3), np.uint8)
all = blank.copy()
data = []

for i in range(3, 15, 2):

    blur = cv2.medianBlur(gray, i)
    edges = cv2.Canny(blur, 50, 150, apertureSize = 3)

    minLineLength = 50
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, maxLineGap, minLineLength)

    yThreshold = 2
    xThreshold = 2

    for line in lines:
        line = line[0]
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        resolved = False

        for line in data:
            if line[0] == x1 and line[1] == y1 and line[2] == x2 and line[3] == y2:
                # lines are the same, ignore:
                resolved = True
                break
            elif line[0] == x1 and line[2] == x2:
                # lines on same vertical axis therefore update y coords:
                line[1] = min(line[1], y1, line[3], y2)
                line[3] = max(line[1], y1, line[3], y2)
                resolved = True
                break
            elif line[1] == y1 and line[3] == y2:
                # lines on same horizontal axis therefore update x coords:
                line[0] = min(line[0], x1, line[2], x2)
                line[2] = max(line[0], x1, line[2], x2)
                resolved = True
                break
            elif (line[0] - xThreshold < x1 or line[0] + xThreshold > x1) and (line[2] - xThreshold < x2 and line[2] + xThreshold > x2):
                # lines on same vertical axis within xThreshold therefore update y coords:
                line[1] = min(line[1], y1, line[3], y2)
                line[3] = max(line[1], y1, line[3], y2)
                resolved = True
                break
            elif (line[1] - yThreshold < y1 or line[1] + yThreshold > y1) and (line[3] - yThreshold < y2 and line[3] + yThreshold > y2):
                # lines on same horizontal axis within yThreshold therefore update x coords:
                line[0] = min(line[0], x1, line[2], x2)
                line[2] = max(line[0], x1, line[2], x2)
                resolved = True
                break

        if resolved == False:
            data.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])

for line in data:
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    all = cv2.line(all, (x1, y1), (x2, y2), (0, 255, 0), 2)

print(data)

cv2.imwrite("./output/all.jpg", all)
