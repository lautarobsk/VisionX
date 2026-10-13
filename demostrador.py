import os
import sys
import time
import requests
from datetime import datetime
import cv2
import re
from ultralytics import YOLO
import easyocr

from utils_patentes import autocorregir_patente, GestorPatentes

API_URL = "http://127.0.0.1:8000/api/tracker/ingesta/"
CAMARA_ID = "CAM-CENTRO-01"

def enviar_al_backend(patente, codigo_camara=CAMARA_ID):
    payload = {
        "patente_leida": patente,
        "codigo_camara": codigo_camara,
        "timestamp": datetime.now().isoformat()
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=3)
        if resp.status_code == 201:
            data = resp.json()
            print(f"[BACKEND OK] Deteccion registrada: {patente} en {codigo_camara}")
            if data.get("ALERTA SOSPECHOSO"):
                print(f"[ALERTA] Vehiculo sospechoso detectado.")
                for p in data.get("predicciones_escape", []):
                    print(f"  -> Bloqueo sugerido en {p['camara_destino']} por calle {p['calle']}")
            return True
        else:
            print(f"[BACKEND ERROR] Codigo {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException:
        print(f"[BACKEND] No se pudo conectar con el servidor en {API_URL}.")
    return False

def main():
    print("Iniciando detector ALPR para Camara 1 (CAM-CENTRO-01)...")
    
    reader = easyocr.Reader(['en', 'es'])
    modelo_vehiculos = YOLO('yolov8n.pt')
    modelo_patentes = YOLO('yolo_patentes.pt')
    gestor = GestorPatentes(max_votos=3)

    ruta_video = os.path.join('media', 'HAVALH6.mp4')
    
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video: {ruta_video}")
        return

    patentes_enviadas = set()
    print(f"Procesando video {ruta_video}. Presione 'q' para salir.")

    while cap.isOpened():
        exito, frame = cap.read()
        if not exito:
            print("Fin del video.")
            break

        alto, ancho = frame.shape[:2]
        nuevo_ancho = 1280
        proporcion = nuevo_ancho / ancho
        nuevo_alto = int(alto * proporcion)
        frame = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

        resultados_v = modelo_vehiculos.track(frame, persist=True, verbose=False)

        if resultados_v[0].boxes is not None:
            for det_v in resultados_v[0].boxes:
                clase_v = int(det_v.cls[0])
                conf_v = float(det_v.conf[0])

                if clase_v == 2 and conf_v > 0.5 and det_v.id is not None:
                    id_auto = int(det_v.id[0])
                    vx1, vy1, vx2, vy2 = map(int, det_v.xyxy[0])

                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Auto ID: {id_auto}', (vx1, vy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    patente_confirmada = gestor.obtener_patente_definitiva(id_auto)
                    if patente_confirmada:
                        cv2.putText(frame, f'[{patente_confirmada}]', (vx1, vy2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        continue

                    progreso = gestor.obtener_progreso(id_auto)
                    if progreso > 0:
                        cv2.putText(frame, f'[Analizando {progreso}/{gestor.max_votos}]',
                                    (vx1, vy2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                    vx1, vy1 = max(0, vx1), max(0, vy1)
                    vx2, vy2 = min(nuevo_ancho, vx2), min(nuevo_alto, vy2)
                    roi_auto = frame[vy1:vy2, vx1:vx2]
                    if roi_auto.size == 0:
                        continue

                    resultados_p = modelo_patentes(roi_auto, verbose=False)

                    for det_p in resultados_p[0].boxes:
                        conf_p = float(det_p.conf[0])
                        if conf_p > 0.4:
                            px1_rel, py1_rel, px2_rel, py2_rel = map(int, det_p.xyxy[0])
                            px1_final = vx1 + px1_rel
                            py1_final = vy1 + py1_rel
                            px2_final = vx1 + px2_rel
                            py2_final = vy1 + py2_rel

                            roi_patente = roi_auto[py1_rel:py2_rel, px1_rel:px2_rel]
                            if roi_patente.size == 0:
                                continue

                            roi_patente = cv2.resize(roi_patente, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                            roi_grises = cv2.cvtColor(roi_patente, cv2.COLOR_BGR2GRAY)
                            roi_blur = cv2.GaussianBlur(roi_grises, (5, 5), 0)
                            _, roi_bin = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                            resultados_ocr = reader.readtext(roi_bin, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            texto_detectado = ""

                            if resultados_ocr:
                                for det_ocr in resultados_ocr:
                                    _, txt, conf_o = det_ocr
                                    if conf_o > 0.35:
                                        texto_detectado += txt.replace(" ", "").upper()

                            if texto_detectado:
                                texto_corregido = autocorregir_patente(texto_detectado)
                                if re.fullmatch(r'^[A-Z]{3}\d{3}$|^[A-Z]{2}\d{3}[A-Z]{2}$', texto_corregido):
                                    conf, ganadora = gestor.registrar_lectura(id_auto, texto_corregido)
                                    if conf:
                                        print(f"[DETECCION] Auto {id_auto} confirmado con patente: {ganadora}")
                                        if ganadora not in patentes_enviadas:
                                            enviar_al_backend(ganadora, CAMARA_ID)
                                            patentes_enviadas.add(ganadora)
                                    else:
                                        pr = gestor.obtener_progreso(id_auto)
                                        print(f"Auto ID {id_auto} muestra {pr}/{gestor.max_votos}: {texto_corregido}")

                            cv2.rectangle(frame, (px1_final, py1_final), (px2_final, py2_final), (0, 255, 255), 2)

        cv2.imshow(f'ALPR Nodo: {CAMARA_ID}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
