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
import os
import random
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from mtcnn import MTCNN


# imports
import numpy as np                     # numeric python lib

import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour
from matplotlib.patches import Rectangle
from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality


# matplotlib setup
# %matplotlib inline
from pylab import rcParams
#rcParams['figure.figsize'] = (6, 6)      # setting default size of plots

directory = r'faces\train\faces'
mylist = os.listdir(r'faces\train\faces')

n = len(mylist)
i = 0
print(n)
# -
for x in range(n):
    image = random.choice(mylist)
    newdir = os.path.join(directory, image)
    image2= plt.imread(newdir)
    i = i+1


def highlight_faces(image_path, faces):
  # display image
  image = plt.imread(image_path)
  plt.imshow(image)

  ax = plt.gca()

  # for each face, draw a rectangle based on coordinates
  for face in faces:
    results = detector.detect_faces(image)
    x, y, width, height = face['box']
    face_border = Rectangle((x, y), width, height,
                          fill=False, color='red')
    ax.add_patch(face_border)
    for key, value in face['keypoints'].items():
            # create and draw dot
            dot = plt.Circle(value, radius=5, color='red')
            ax.add_patch(dot)
            # show the plot
  plt.show()


# +
image = random.choice(mylist)
newdir = os.path.join(directory, image)
#newdir = r'C:\Users\Mattmad1234\Desktop\WORKING\305Project\unknown8.jpg'
image= plt.imread(newdir)
print(newdir)


detector = MTCNN()

faces = detector.detect_faces(image)
for face in faces:
  print(face)

highlight_faces(newdir, faces)
# -


