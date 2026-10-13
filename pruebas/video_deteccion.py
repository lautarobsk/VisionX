import os
import cv2
from ultralytics import YOLO
import easyocr
import re 
import requests
from datetime import datetime

from utils_patentes import autocorregir_patente, GestorPatentes

API_URL = "http://127.0.0.1:8000/api/tracker/ingesta/"
CAMARA_ID = "CAM-CENTRO-01"
patentes_enviadas = set()

def enviar_al_backend(patente):
    if patente in patentes_enviadas:
        return
    payload = {
        "patente_leida": patente,
        "codigo_camara": CAMARA_ID,
        "timestamp": datetime.now().isoformat()
    }
    try:
        res = requests.post(API_URL, json=payload, timeout=3)
        if res.status_code == 201:
            patentes_enviadas.add(patente)
            print(f"[BACKEND OK] Deteccion enviada: {patente} en {CAMARA_ID}")
            data = res.json()
            if data.get("ALERTA SOSPECHOSO"):
                print(f"[ALERTA] El vehiculo {patente} es sospechoso. Plan de bloqueo activado.")
        else:
            print(f"[BACKEND ERROR] Codigo {res.status_code}: {res.text}")
    except requests.exceptions.RequestException:
        print(f"[BACKEND] No se pudo conectar a {API_URL}. Compruebe que Django este iniciado.")

base_dir = os.path.dirname(os.path.abspath(__file__))
modelo_vehiculos = YOLO(os.path.join(base_dir, '..', 'yolov8n.pt')) 
modelo_patentes = YOLO(os.path.join(base_dir, '..', 'yolo_patentes.pt')) 

ruta_video = os.path.join(base_dir, '..', 'media', 'Autos1.mov')
if not os.path.exists(ruta_video):
    ruta_video = os.path.join(base_dir, '..', 'media', 'HAVALH6.mp4')

cap = cv2.VideoCapture(ruta_video)

gestor = GestorPatentes(max_votos=3)


while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        print("Fin del video o error al leer la cámara.")
        break

    alto, ancho = frame.shape[:2]
    nuevo_ancho = 1280 
    proporcion = nuevo_ancho / ancho
    nuevo_alto = int(alto * proporcion)
    frame = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

    resultados_vehiculos = modelo_vehiculos.track(frame, persist=True, verbose=False)

    if resultados_vehiculos[0].boxes is not None:
        for det_vehiculo in resultados_vehiculos[0].boxes:
            clase_v = int(det_vehiculo.cls[0])
            conf_v = float(det_vehiculo.conf[0])

            if clase_v == 2 and conf_v > 0.5 and det_vehiculo.id is not None:
                id_auto = int(det_vehiculo.id[0])
                vx1, vy1, vx2, vy2 = map(int, det_vehiculo.xyxy[0])
                
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                cv2.putText(frame, f'Auto ID: {id_auto}', (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                patente_guardada = gestor.obtener_patente_definitiva(id_auto)
                if patente_guardada:
                    cv2.putText(frame, f'[{patente_guardada}]', (vx1, vy2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    continue
                
                progreso = gestor.obtener_progreso(id_auto)
                if progreso > 0:
                    cv2.putText(frame, f'[Analizando... {progreso}/{gestor.max_votos}]', (vx1, vy2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                

                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(nuevo_ancho, vx2), min(nuevo_alto, vy2)
                
                roi_auto = frame[vy1:vy2, vx1:vx2]
                if roi_auto.size == 0: continue

                resultados_patentes = modelo_patentes(roi_auto, verbose=False)

                for det_patente in resultados_patentes[0].boxes:
                    conf_p = float(det_patente.conf[0])

                    if conf_p > 0.4: 
                        px1_rel, py1_rel, px2_rel, py2_rel = map(int, det_patente.xyxy[0])
                        
                        px1_final = vx1 + px1_rel
                        py1_final = vy1 + py1_rel
                        px2_final = vx1 + px2_rel
                        py2_final = vy1 + py2_rel

                        roi_patente = roi_auto[py1_rel:py2_rel, px1_rel:px2_rel]
                        if roi_patente.size == 0: continue

                        # Agranda la imagen
                        roi_patente = cv2.resize(roi_patente, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
  
                        # Desenfoque para quitar ruido
                        roi_grises = cv2.cvtColor(roi_patente, cv2.COLOR_BGR2GRAY)
                        roi_blur = cv2.GaussianBlur(roi_grises, (5, 5), 0)
                        _, roi_bin = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        resultados_ocr = reader.readtext(roi_bin, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        texto_detectado = ""
                        ultima_conf_ocr_valida = 0.0

                        if resultados_ocr:
                            for deteccion in resultados_ocr:
                                _, texto, conf_ocr = deteccion

                                if conf_ocr > 0.4:
                                    texto_limpio = texto.replace(" ", "").upper()
                                    texto_detectado += texto_limpio
                                    ultima_conf_ocr_valida = conf_ocr

                        if texto_detectado:
                            texto_detectado = autocorregir_patente(texto_detectado)
                            # Veridicamos si cumple con el formato de patentes del mercosur o el formato viejo argentino (ej: OUV200 o AA000BB)
                            es_formato_valido = re.fullmatch(r'^[A-Z]{3}\d{3}$|^[A-Z]{2}\d{3}[A-Z]{2}$', texto_detectado)
                            
                            if es_formato_valido:
                                confirmada, ganadora = gestor.registrar_lectura(id_auto, texto_detectado)
                                
                                if confirmada:
                                    print(f"[OK] Auto {id_auto} confirmado con patente: {ganadora}")
                                    enviar_al_backend(ganadora)
                                else:
                                    progreso = gestor.obtener_progreso(id_auto)
                                    print(f"ID {id_auto} {progreso}/{gestor.max_votos}: {texto_detectado} (Confianza OCR: {ultima_conf_ocr_valida:.2f})")
                            else:
                                print(f"   [Descartado] '{texto_detectado}'no cumple el formato.")


                        cv2.rectangle(frame, (px1_final, py1_final), (px2_final, py2_final), (0, 255, 255), 2)
                        

    cv2.imshow('Sistema ALPR - Tiempo Real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
