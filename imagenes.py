import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((10,10), np.uint8)

img[0, 1] = 140

img2 = 100 * np.ones((10,10,3), np.uint8)

R = img2[:,:,0]
G = img2[:,:,1]
B = img2[:,:,2]

#R[:,:] = 0

#img2[:,:,0] = R


R[:,:] = 255
G[:,:] = 255
B[:,:] = 0


print(img2)
plt.imshow(img2)
plt.show()