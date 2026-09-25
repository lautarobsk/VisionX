import cv2
import easyocr

lector = easyocr.Reader(['en', 'es'])

recorte_patente = cv2.imread('Mat.png')

resultados = lector.readtext(recorte_patente)


# EasyOCR retorna una tupla de 3 parametros Coordenadas, el Texto, y la Confianza
# ([[10, 20], [200, 20], [200, 80], [10, 80]],  
#    "AB123CD",                                   
#    0.87)
# En cambio YOLO retorna un objeto 
# # coordenadas = deteccion.xyxy[0]
#c onfianza = float(deteccion.conf[0])
# clase = int(deteccion.cls[0])

for deteccion in resultados:
    coordenadas, texto, confianza = deteccion
    if confianza > 0.50:
        patente_limpia = texto.replace(" ", "").upper()
        print(f"Patente leída: {patente_limpia} | Seguridad: {confianza:.2f}")       
