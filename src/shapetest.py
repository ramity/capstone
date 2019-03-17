import imutils
import numpy as np
import cv2

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread("./input/scene.JPG")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
height, width = image.shape[:2]

sensitivity = 40
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,255,255])
mask = cv2.inRange(hsv, lower_white, upper_white)
res = cv2.bitwise_and(image, image, mask=mask)
res = cv2.bilateralFilter(res, 9, 75, 75)


cv2.imwrite("./output/mask.jpg", mask)
cv2.imwrite("./output/res.jpg", res)

final = res.copy()
rgb = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
squares = np.zeros(image.shape)

cubes = 5
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:cubes]

for contour in contours:
    M = cv2.moments(contour)

    # prevent division by 0
    if(M['m00']):
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(contour)

        convexity = cv2.isContourConvex(contour)

        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(squares,(x,y),(x+w,y+h),(0,255,0),2)

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(final,[box],0,(0,0,255),2)

        final[y:y+h,x:x+w] = image[y:y+h,x:x+w]
        face = gray[y:y+h,x:x+w]
        rect, face = cv2.threshold(face, 0, 175, cv2.THRESH_BINARY)
        corners = cv2.goodFeaturesToTrack(face,7,0.01,10)

        if(corners is not None):
            corners = np.int0(corners)

            for i in corners:
                facex,facey = i.ravel()
                cv2.circle(final,(x+facex,y+facey),5,(0,0,255),-1)

        edges = cv2.Canny(face, 50, 150, apertureSize = 3)

cv2.imwrite("./output/edges.jpg", edges)
cv2.imwrite("./output/final.jpg", final)
cv2.imwrite("./output/squares.jpg", squares)
