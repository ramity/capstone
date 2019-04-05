import cv2
import numpy as np
import pytesseract

debug = True

def getPlayingField(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    playingFieldUpperBounds = (16, 255, 255)
    playingFieldLowerBounds = (0, 50, 50)

    playingFieldMask = cv2.inRange(hsv, playingFieldLowerBounds, playingFieldUpperBounds)
    playingFieldImage = cv2.bitwise_and(image, image, mask=playingFieldMask)

    # find the biggest contour (hopefully the board)
    grayPlayingFieldImage = cv2.cvtColor(playingFieldImage, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(playingFieldMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # get the subregion bounded by the contour
    contourMask = np.zeros(grayPlayingFieldImage.shape, np.uint8)
    cv2.fillPoly(contourMask, pts=[contour], color=(255,255,255))
    output = cv2.bitwise_and(image, image, mask=contourMask)

    if debug:
        cv2.imwrite("./output/playingFieldMask.jpg", playingFieldMask)
        cv2.imwrite("./output/playingFieldContourMask.jpg", contourMask)
        cv2.imwrite("./output/playingFieldImage.jpg", output)

    return output

def getObstacle(playingFieldImage):
    grayPlayingFieldImage = cv2.cvtColor(playingFieldImage, cv2.COLOR_BGR2GRAY)

    obstacleTemplate = cv2.imread("./input/obstacleTemplate.jpg", 0)
    w, h = obstacleTemplate.shape[::-1]

    obstacleImage = cv2.matchTemplate(grayPlayingFieldImage, obstacleTemplate, cv2.TM_CCOEFF_NORMED)
    threshold = 0.65
    loc = np.where(obstacleImage >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(playingFieldImage, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    if debug:
        cv2.imwrite("./output/obstacleImage.jpg", playingFieldImage)

    return obstacleImage

def getCubes(playingFieldImage):

    hsv = cv2.cvtColor(playingFieldImage, cv2.COLOR_BGR2HSV)

    playingFieldUpperBounds = (20, 255, 255)
    playingFieldLowerBounds = (0, 135, 145)

    playingFieldMaskInv = cv2.inRange(hsv, playingFieldLowerBounds, playingFieldUpperBounds)
    playingFieldMask = cv2.bitwise_not(playingFieldMaskInv)
    kernel = np.ones((5, 5), np.uint8)
    playingFieldMask = cv2.erode(playingFieldMask, kernel, iterations=10)
    playingFieldMask = cv2.dilate(playingFieldMask, kernel, iterations=10)
    playingFieldImage = cv2.bitwise_and(playingFieldImage, playingFieldImage, mask=playingFieldMask)
    grayPlayingFieldImage = cv2.cvtColor(playingFieldImage, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(grayPlayingFieldImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1], reverse=True)

    limit = 5

    for contour in contours:

        x,y,w,h = cv2.boundingRect(contour)
        face = playingFieldImage[y:y+h,x:x+w]
        cv2.rectangle(playingFieldImage,(x,y),(x+w,y+h),(0,255,0),10)

        area = w * h

        if area > 200:

            if h > w:
                if h > w * 1.25:
                    print("OBSTACLE!")
                else:
                    print("CUBE!")
                    print(pytesseract.image_to_string(face))

            if w > h:
                if h * 2 > w:
                    print("CUBE!")
                    print(pytesseract.image_to_string(face))
                else:
                    print("MOTHERSHIP!")


            if limit == 1:
                break;

            limit -= 1

    if debug:
        cv2.imwrite("./output/cubesPlayingFieldMask.jpg", playingFieldMask)
        cv2.imwrite("./output/cubesPlayingField.jpg", playingFieldImage)

    return playingFieldImage


image = cv2.imread("./input/scene3.jpg")
playingFieldImage = getPlayingField(image)
#obstacleImage = getObstacle(playingFieldImage)
cubeImage = getCubes(playingFieldImage)
