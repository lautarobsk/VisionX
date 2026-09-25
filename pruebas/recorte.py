import matplotlib.pyplot as plt
import cv2

img = cv2.imread("bmwm5.jpg")
roi = img[100:1000, 150:1500]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(8,4))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("ROI")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gray, cmap="gray")
plt.title("Grises")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(thresh, cmap="gray")
plt.title("Umbralizada")
plt.axis("off")

plt.show()

