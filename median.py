import cv2 
import numpy as np

image = cv2.imread('pene.png')

blur = cv2.blur(image,(5,5))

cv2.imshow('Pene',blur)
cv2.waitKey(0)
