import cv2
import easyocr
from ultralytics import YOLO
from utils_patentes import autocorregir_patente

#Archivo para prueba con fotos 

modelo_patentes = YOLO('../yolo_patentes.pt') 
modelo_auto = YOLO('../yolov8n.pt')
lector = easyocr.Reader(['en', 'es'])

img = cv2.imread(r'C:\\Users\\lauta\\OneDrive\\Desktop\\sandbox-vision\\media\\auto.jpg')
alto, ancho = img.shape[:2]
nuevo_ancho = 1280 
proporcion = nuevo_ancho / ancho
nuevo_alto = int(alto * proporcion)
img = cv2.resize(img, (nuevo_ancho, nuevo_alto))


resultado_auto = modelo_auto(img, verbose=False)

auto = resultado_auto[0].boxes

if auto is not None:
    
    for det in auto:

        clase_auto = int(det.cls[0])
        conf_auto = float(det.conf[0])

        if clase_auto == 2 and conf_auto >= 0.6:
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            auto_roi = img[y1:y2, x1:x2]

            resultado_patente = modelo_patentes(auto_roi)
            deteccion = resultado_patente[0].boxes
            if len(deteccion) == 0:
                continue
            conf = float(deteccion.conf[0])
                
            if conf > 0.4:
                px1, py1, px2, py2 = map(int, deteccion.xyxy[0])
                
                px1_global = x1 + px1
                py1_global = y1 + py1
                px2_global = x1 + px2
                py2_global = y1 + py2

                cv2.rectangle(img, (px1_global, py1_global), (px2_global, py2_global), (0, 255, 255), 2)

                patente = auto_roi[py1:py2, px1:px2]
                
                if patente.size == 0:
                    continue

                patente = cv2.resize(patente, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                grises = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(grises, (5, 5), 0)
                _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Intentar primero con imagen normal, si falla probar invertida
                for imagen_ocr in [binaria, cv2.bitwise_not(binaria)]:
                    resultado_ocr = lector.readtext(imagen_ocr, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    if resultado_ocr:
                        break

                texto_detectado = ""

                if len(resultado_ocr) == 0:
                    continue

                _, texto, conf_ocr = resultado_ocr[0]

                if conf_ocr > 0.3:
                    texto_limpio = texto.replace(" ", "").upper()
                    texto_detectado += texto_limpio

                texto_detectado = autocorregir_patente(texto_detectado)

                if len(texto_detectado) >= 6:
                    print(f"Patente: {texto_detectado}")

                cv2.putText(img, f'PATENTE: {texto_detectado}', (px1_global, py1_global - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    

cv2.imshow('Sistema ALPR', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
