# -*- coding: utf-8 -*-
"""Pencil Sketch

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LvZAc-m2fUepb7RmcsArdCEdtDuaTrSG
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

from google.colab import drive
drive.mount('/content/drive')

img = cv2.imread("/content/drive/MyDrive/pencil/PicsArt_07-16-09.19.35.jpg")

cv2_imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2_imshow(img_gray)

img_invert = cv2.bitwise_not(img_gray)

cv2_imshow(img_invert)

img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)
cv2_imshow(img_smoothing)

def dodgeV2(x, y):
  return cv2.divide(x, 255 - y, scale=256)

final_img = dodgeV2(img_gray, img_smoothing)

cv2_imshow(final_img)
