from PIL import Image
import cv2
import numpy as np
import random
import sys
import time

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def main():

    image = cv2.imread("./input/e2.jpg")
    azoom = cv2.imread("./input/azoom.jpg", 0)
    bzoom = cv2.imread("./input/bzoom.jpg", 0)
    czoom = cv2.imread("./input/czoom.jpg", 0)
    dzoom = cv2.imread("./input/dzoom.jpg", 0)
    ezoom = cv2.imread("./input/ezoom.jpg", 0)
    fzoom = cv2.imread("./input/fzoom.jpg", 0)

    if image is None:
        sys.exit("Image was unable to load")
    if azoom is None:
        sys.exit("Subimage was unable to load")
    if bzoom is None:
        sys.exit("Subimage was unable to load")
    if czoom is None:
        sys.exit("Subimage was unable to load")
    if dzoom is None:
        sys.exit("Subimage was unable to load")
    if ezoom is None:
        sys.exit("Subimage was unable to load")
    if fzoom is None:
        sys.exit("Subimage was unable to load")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = sp_noise(gray, 0.25)
    cv2.imwrite("./output/start.jpg", gray)

    edges = cv2.Canny(gray, 150, 150, apertureSize = 3)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    cv2.imwrite("./output/canny.jpg", edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, 50, 50)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imwrite("./output/lines.jpg", image)

    amts = time.time()
    ares = cv2.matchTemplate(gray, azoom, cv2.TM_CCOEFF)
    adur = time.time() - amts

    bmts = time.time()
    bres = cv2.matchTemplate(gray, bzoom, cv2.TM_CCOEFF)
    bdur = time.time() - bmts

    cmts = time.time()
    cres = cv2.matchTemplate(gray, czoom, cv2.TM_CCOEFF)
    cdur = time.time() - cmts

    dmts = time.time()
    dres = cv2.matchTemplate(gray, dzoom, cv2.TM_CCOEFF)
    ddur = time.time() - dmts

    emts = time.time()
    eres = cv2.matchTemplate(gray, ezoom, cv2.TM_CCOEFF)
    edur = time.time() - emts

    fmts = time.time()
    fres = cv2.matchTemplate(gray, fzoom, cv2.TM_CCOEFF)
    fdur = time.time() - fmts

    a_min_val, a_max_val, a_min_loc, a_max_loc = cv2.minMaxLoc(ares)
    b_min_val, b_max_val, b_min_loc, b_max_loc = cv2.minMaxLoc(bres)
    c_min_val, c_max_val, c_min_loc, c_max_loc = cv2.minMaxLoc(cres)
    d_min_val, d_max_val, d_min_loc, d_max_loc = cv2.minMaxLoc(dres)
    e_min_val, e_max_val, e_min_loc, e_max_loc = cv2.minMaxLoc(eres)
    f_min_val, f_max_val, f_min_loc, f_max_loc = cv2.minMaxLoc(fres)

    max_val = max((a_max_val, b_max_val, c_max_val, d_max_val, e_max_val, f_max_val))

    a_is_max = (a_max_val == max_val)
    b_is_max = (b_max_val == max_val)
    c_is_max = (c_max_val == max_val)
    d_is_max = (d_max_val == max_val)
    e_is_max = (e_max_val == max_val)
    f_is_max = (f_max_val == max_val)

    data = {}
    data['a'] = {}
    data['a']['delta'] = max_val - a_max_val
    data['a']['duration'] = adur
    data['a']['is_max'] = a_is_max
    data['a']['min_val'] = a_min_val
    data['a']['max_val'] = a_max_val
    data['a']['min_loc'] = a_min_loc
    data['a']['max_val'] = a_max_val
    data['b'] = {}
    data['b']['delta'] = max_val - b_max_val
    data['b']['duration'] = bdur
    data['b']['is_max'] = b_is_max
    data['b']['min_val'] = b_min_val
    data['b']['max_val'] = b_max_val
    data['b']['min_loc'] = b_min_loc
    data['b']['max_val'] = b_max_val
    data['c'] = {}
    data['c']['delta'] = max_val - c_max_val
    data['c']['duration'] = cdur
    data['c']['is_max'] = c_is_max
    data['c']['min_val'] = c_min_val
    data['c']['max_val'] = c_max_val
    data['c']['min_loc'] = c_min_loc
    data['c']['max_val'] = c_max_val
    data['d'] = {}
    data['d']['delta'] = max_val - d_max_val
    data['d']['duration'] = ddur
    data['d']['is_max'] = d_is_max
    data['d']['min_val'] = d_min_val
    data['d']['max_val'] = d_max_val
    data['d']['min_loc'] = d_min_loc
    data['d']['max_val'] = d_max_val
    data['e'] = {}
    data['e']['delta'] = max_val - e_max_val
    data['e']['duration'] = edur
    data['e']['is_max'] = e_is_max
    data['e']['min_val'] = e_min_val
    data['e']['max_val'] = e_max_val
    data['e']['min_loc'] = e_min_loc
    data['e']['max_val'] = e_max_val
    data['f'] = {}
    data['f']['delta'] = max_val - f_max_val
    data['f']['duration'] = fdur
    data['f']['is_max'] = f_is_max
    data['f']['min_val'] = f_min_val
    data['f']['max_val'] = f_max_val
    data['f']['min_loc'] = f_min_loc
    data['f']['max_val'] = f_max_val

    for character, row in data.items():
        print(character)
        print("Delta: " + str(row['delta']))
        print("Duration: " + str(row['duration']))
        print("Match?: " + str(row['is_max']))

    if(a_is_max):
        top_left = a_max_loc
        bottom_right = (top_left[0] + azoom.shape[1], top_left[1] + azoom.shape[0])
        character = 'a'
    elif(b_is_max):
        top_left = b_max_loc
        bottom_right = (top_left[0] + bzoom.shape[1], top_left[1] + bzoom.shape[0])
        character = 'b'
    elif(c_is_max):
        top_left = c_max_loc
        bottom_right = (top_left[0] + czoom.shape[1], top_left[1] + czoom.shape[0])
        character = 'c'
    elif(d_is_max):
        top_left = d_max_loc
        bottom_right = (top_left[0] + dzoom.shape[1], top_left[1] + dzoom.shape[0])
        character = 'd'
    elif(e_is_max):
        top_left = e_max_loc
        bottom_right = (top_left[0] + ezoom.shape[1], top_left[1] + ezoom.shape[0])
        character = 'e'
    elif(f_is_max):
        top_left = f_max_loc
        bottom_right = (top_left[0] + fzoom.shape[1], top_left[1] + fzoom.shape[0])
        character = 'f'

    cv2.rectangle(image, top_left, bottom_right, 255, 2)
    cv2.putText(image, character, top_left, cv2.FONT_HERSHEY_SIMPLEX, 4, 0, 2)
    cv2.imwrite("./output/final.jpg", image)

main();
