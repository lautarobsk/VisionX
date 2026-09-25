import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('autito.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


blur = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(blur, 30, 200)

contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

license_ratio = 3.07
candidates = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    area = w * h

    
    if np.isclose(aspect_ratio, license_ratio, atol=0.8) and area > 1000:
        candidates.append(cnt)

canvas = np.zeros_like(img)
cv2.drawContours(canvas, candidates, -1, (0, 255, 0), 2)

plt.axis('off')
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)) 
plt.show()