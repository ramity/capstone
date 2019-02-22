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

    input = "./input/a.jpg"
    image = cv2.imread(input)

    if image is None:
        sys.exit("Image was unable to load")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./output/gray.jpg", gray)

    height, width = gray.shape
    final = np.zeros((height, width, 3), np.uint8)

    median = cv2.medianBlur(gray, 3)
    cv2.imwrite("./output/median.jpg", median)

    output = pytesseract.image_to_data(median, output_type = pytesseract.Output.DICT, config='--psm 10')
    rows = len(output["level"])

    print(output)

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
        color = (255, 255, 255)
        fontScale = 0.5
        lineType = 2

        if confidence != -1 and text != "":
            size = cv2.getTextSize(text, font, fontScale, lineType)
            cv2.putText(final, text, (left, top), font, fontScale, color, lineType)
            textWidth = size[0][0]
            textHeight = size[0][1]
            cv2.putText(final, str(confidence), (left + textWidth, top + textHeight), font, fontScale, (0, 255, 0), lineType)

    cv2.imwrite("./output/final.jpg", final)

main();
