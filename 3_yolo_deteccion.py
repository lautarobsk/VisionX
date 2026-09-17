import cv2
from ultralytics import YOLO

modelo = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('McP1.mp4')

while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        break
    resultados = modelo.track(frame, persist=True)

    if resultados[0].boxes is not None:
        for deteccion in resultados[0].boxes:
            clase = int(deteccion.cls[0])
            
            if clase == 2 and deteccion.id is not None:
                id_auto = int(deteccion.id[0])
                x1, y1, x2, y2 = map(int, deteccion.xyxy[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                texto = f'Auto ID: {id_auto}'
                cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Tracking de Vehiculos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()