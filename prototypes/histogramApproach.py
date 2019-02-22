import numpy as np
from PIL import Image
import cv2
import sys

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

lightFontCasePath = "./input/2018022200072NAVWHN20.TIF"
verA = "./input/2018022200120NAVWHN10.TIF"
verC = "./input/2018022200074NAVWHN30.TIF"
worstCasePath = "./input/2017122800008FWR3SV44.TIF"
loadingErrorCasePath = "./input/085a17f2-f5c4-411d-bbcf-8e7303498fa.tif"

image = cv2.imread(worstCasePath)

if image is None:
    sys.exit("Image was not loaded properly")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cuts = []
gaps = []

grayInv = 255 - gray
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 3)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 5)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 7)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 9)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 11)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 13)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 15)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 17)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

grayInv = cv2.medianBlur(grayInv, 19)
rowMeans = cv2.reduce(grayInv, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
rowGaps = zero_runs(rowMeans)
rowCutpoints = (rowGaps[:,0] + rowGaps[:,1] - 1) / 2
gaps.extend(rowGaps.toList())
cuts.extend(rowCutpoints)

cuts = set(cuts)
print(cuts)
print(gaps)

height, width = gray.shape
output = gray.copy()

for pair in gaps:
    output = cv2.rectangle(output, (0, int(pair[0])), (width, int(pair[1])), 126, -1)

for y in cuts:
    output = cv2.line(output, (0, int(y)), (width, int(y)), 0, 2)

cv2.imwrite("./output/cuts.jpg", output)

sys.exit(0)

cv2.imwrite("./output/rowCutpoints.jpg", rowCutpoints)
