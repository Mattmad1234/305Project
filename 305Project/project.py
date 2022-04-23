# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import math
import requests
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
from skimage import measure            # to find shape contour
from skimage.feature import corner_harris, corner_subpix, corner_peaks, CENSURE
from scipy.signal import argrelextrema

from PIL import Image
from pathlib import Path
from pylab import rcParams
rcParams['figure.figsize'] = (6, 6)

# +
# Randomize the face being processed


james = (r"C:\Users\Mattmad1234\Desktop\WORKING\305Project\faces\099946.jpg") # Need to localize the files to the programs directory.
mary = (r"C:\Users\Mattmad1234\Desktop\WORKING\305Project\faces\099947.jpg")
hank = (r"C:\Users\Mattmad1234\Desktop\WORKING\305Project\faces\099948.jpg")

image = random.choice([hank, mary, james])

print (image)

# -

# Define Distance formula between 2 points
def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]


# Use the Prewitt edge detection to create a new image defining the curves of the face
image = imread(image, as_gray=True)
#calculating horizontal edges using prewitt kernel
edges_prewitt_horizontal = prewitt_h(image)
#calculating vertical edges using prewitt kernel
edges_prewitt_vertical = prewitt_v(image)
imshow(edges_prewitt_vertical, cmap='gray')


cy, cx = ndi.center_of_mass(edges_prewitt_vertical)
plt.imshow(edges_prewitt_vertical)
plt.show()
#image1 = imread(image, as_gray=True)
#imshow(image1)

# +
contours = measure.find_contours(edges_prewitt_vertical, .05)

contour = max(contours, key=len)

plt.plot(contour[::,1], contour[::,0], linewidth=0.5)
plt.imshow(edges_prewitt_vertical, cmap='Set3')
plt.show()

# +
polar_contour = np.array([cart2pol(x, y) for x, y in contour])

plt.plot(polar_contour[::,1], polar_contour[::,0], linewidth=0.5)
plt.show()
# -

contour[::,1] -= cx
contour[::,0] -= cy

plt.plot(-contour[::,1], -contour[::,0], linewidth=0.5)
plt.grid()
plt.scatter(0, 0)
plt.show()

# +
polar_contour = np.array([cart2pol(x, y) for x, y in contour])

rcParams['figure.figsize'] = (12, 6)
plt.subplot(121)
plt.scatter(polar_contour[::,1], polar_contour[::,0], linewidth=0, s=.5, c=polar_contour[::,1])
plt.title('Polar Coordinates')
plt.grid()
plt.subplot(122)
plt.scatter(contour[::,1],
            contour[::,0],
            linewidth=0, s=2,
            c=range(len(contour)))
plt.title('Cartesian Coordinates')
plt.grid()
plt.show()

# +
detector = CENSURE()
detector.detect(image)

coords = corner_peaks(corner_harris(image), min_distance=5)
coords_subpix = corner_subpix(image, coords, window_size=13)

plt.subplot(121)
plt.title('CENSURE feature detection')
plt.imshow(image, cmap='Set3')
plt.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              3 ** detector.scales, facecolors='none', edgecolors='r')

print(detector.keypoints)

# +
c_max_index = argrelextrema(polar_contour[::,0], np.greater, order=20)
c_min_index = argrelextrema(polar_contour[::,0], np.less, order=20)

plt.subplot(121)
plt.scatter(polar_contour[::,1], polar_contour[::,0], 
            linewidth=0, s=2, c='k')
plt.scatter(polar_contour[::,1][c_max_index], 
            polar_contour[::,0][c_max_index], 
            linewidth=0, s=30, c='b')
plt.scatter(polar_contour[::,1][c_min_index], 
            polar_contour[::,0][c_min_index], 
            linewidth=0, s=30, c='r')

plt.subplot(122)
plt.scatter(contour[::,1], contour[::,0], 
            linewidth=0, s=2, c='k')
plt.scatter(contour[::,1][c_max_index], 
            contour[::,0][c_max_index], 
            linewidth=0, s=30, c='r')
plt.scatter(contour[::,1][c_min_index], 
            contour[::,0][c_min_index], 
            linewidth=0, s=30, c='b')

plt.show()

print(c_max_index)
print(c_min_index)
# -


