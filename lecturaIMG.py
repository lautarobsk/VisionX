import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('bmwm5.jpg')

tamaño = img.shape
tipo = img.dtype
print(tamaño, tipo)