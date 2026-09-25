import cv2
import easyocr
from ultralytics import YOLO

modelo_patentes = YOLO('yolo_patentes.pt') 
lector = easyocr.Reader(['en', 'es'])

img = cv2.imread('auto.jpg')
alto, ancho = img.shape[:2]
nuevo_ancho = 1280 
proporcion = nuevo_ancho / ancho
nuevo_alto = int(alto * proporcion)
img = cv2.resize(img, (nuevo_ancho, nuevo_alto))


resultado_patente = modelo_patentes(img)
deteccion = resultado_patente[0].boxes
conf = float(deteccion.conf[0])
    
if conf > 0.4:
    px1, py1, px2, py2 = map(int, deteccion.xyxy[0])
    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 255), 2)

    patente = img[py1:py2, px1:px2]

    grises = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(grises, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    resultado_ocr = lector.readtext(bin)
    texto_detectado = ""

    _, texto, conf_ocr = resultado_ocr[0]

    if conf_ocr > 0.4:
        texto_limpio = texto.replace(" ", "").upper()
        texto_detectado += texto_limpio

    if len(texto_detectado) >= 6:
        print(f"Patente: {texto_detectado}")
    else:
        pass

    

cv2.imshow('Sistema ALPR', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


    