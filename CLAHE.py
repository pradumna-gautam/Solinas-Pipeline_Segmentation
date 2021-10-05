# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 20:56:22 2021

@author: Pradumna
"""
# CLAHE
# Contrast Limited Adaptive Histogram Equalization and Thresholding using OpenCV in Python
import cv2
import numpy as np
from matplotlib import pyplot as plt 


#Method1
##############################################################################################################

#First reading the image as grayscale and assigned it to the variable img. 
#To perform histogram equalization we can run cv2.equalizeHist(img)
img = cv2.imread("From S11 towards S12.mp4_vframe_51.404038.png", 0)
img = cv2.imread("From S11 towards S12.mp4_vframe_2951.107363.png", 0)
img = cv2.imread("From S11 towards S12.mp4_vframe_2787.948811.png", 0)
img = cv2.imread("From S11 towards S12.mp4_vframe_2677.639854.png", 0)
img = cv2.imread("From S11 towards S12.mp4_vframe_814.531266.png", 0)
img = cv2.imread("violet light on camera.png", 0)
equ = cv2.equalizeHist(img)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.imshow("Equalized", equ)
cv2.waitKey(0)

#Test image’s histogram
plt.hist(img.flat, bins=100, range=(0, 255))

#Equalized image’s histogram
plt.hist(equ.flat, bins=100, range=(0, 255))


# Applying CLAHE
clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8)) #Define tile size and clip limit.
cl_img = clahe.apply(img)
cv2.imshow('CLAHE',cl_img)
cv2.waitKey(0)



##################################################
##################################################
#Image Thresholding
#Histogram of the CLAHE image
plt.hist(cl_img.flat, bins=100, range=(100, 255))


ret, thresh1 = cv2.threshold(cl_img, 220, 170, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(cl_img, 220, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('Binary thresholded',thresh1)
cv2.waitKey(0)
cv2.imshow('Inverted Binary thresholded',thresh2)
cv2.waitKey(0)

#Using OTSU we can automatically segment it.

#
# If working with noisy images
# Clean up noise for better thresholding
# Otsu's thresholding after Gaussian filtering. Canuse median or NLM for beteer edge preserving
#blur = cv2.GaussianBlur(cl_img,(5,5),0)

ret, thresh3 = cv2.threshold(cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('image',thresh3)
cv2.waitKey(0)





#Method2 (Geeksforgeeks) (Not Working Well)
###############################################################################################################

import cv2
import numpy as np

# Reading the image from the present directory
image = cv2.imread("From S11 towards S12.mp4_vframe_51.404038.png", 0)
image = cv2.imread("From S11 towards S12.mp4_vframe_2951.107363.png", 0)
image = cv2.imread("From S11 towards S12.mp4_vframe_2787.948811.png", 0)
image = cv2.imread("From S11 towards S12.mp4_vframe_2677.639854.png", 0)
image = cv2.imread("From S11 towards S12.mp4_vframe_814.531266.png", 0)
image = cv2.imread("violet light on camera.png", 0)
# Resizing the image for compatibility
image = cv2.resize(image, (500, 600))

# The initial processing of the image
image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit = 5)
final_img = clahe.apply(image) + 30

# Ordinary thresholding the same image
_, ordinary_img = cv2.threshold(image, 155, 255, cv2.THRESH_BINARY)

# Showing all the three images
cv2.imshow("ordinary threshold", ordinary_img)
cv2.waitKey(0)
cv2.imshow("CLAHE image", final_img)
cv2.waitKey(0)