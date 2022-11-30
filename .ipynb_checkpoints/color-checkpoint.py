import numpy as np
import cv2

# use numpy to create an array of color
color = (255,255,255)
pixel_array = np.full((120, 160, 3), color, dtype=np.uint8)
cv2.imshow('image',pixel_array)
cv2.waitKey(0)