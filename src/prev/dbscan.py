import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import cv2
from PIL import Image
import sys


# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                            random_state=0)

image = cv2.imread("./output/lotsOfContours.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
data = []

for cy, row in enumerate(gray):
    for cx, value in enumerate(row):
        if gray[cy][cx] > 0:
            data.append([cx, cy])
            #print("cx: " + str(cx) + " cy: " + str(cy), flush=True)

data = np.array(data)

#print(data)
#print(X)
#print(labels_true)

#X = StandardScaler().fit_transform(X)

#print(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=5.25, min_samples=65).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

height, width = gray.shape
dots = np.zeros((height, width, 4), np.uint8)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    color = (col[0] * 255, col[1] * 255, col[2] * 255)
    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples_mask]

    for value in xy:
        x = value[0]
        y = value[1]
        dots = cv2.circle(dots, (x, y), 3, color, -1)

    xy = data[class_member_mask & ~core_samples_mask]

    for value in xy:
        x = value[0]
        y = value[1]
        dots = cv2.circle(dots, (x, y), 1, color, -1)

    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #         markeredgecolor='k', markersize=14)

    #xy = data[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #         markeredgecolor='k', markersize=6)

#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()

cv2.imwrite("./output/dots.jpg", dots)
