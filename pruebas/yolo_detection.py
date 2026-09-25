import cv2
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('autito.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
 
contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

# Medidas de las patentes Argentina (400x130)
license_ratio = 3.07692307692
min_w = 80
max_w = 110
min_h = 25
max_h = 52

candidates = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h

    if (np.isclose(aspect_ratio, license_ratio, atol=0.7)) and (max_w > w > min_w) and (max_h > h > min_h):
        candidates.append(cnt)


canvas = np.zeros_like(img)
cv2.drawContours(canvas, candidates, -1, (0, 255, 0), 2)
plt.axis('off')
plt.imshow(canvas)
plt.show()
