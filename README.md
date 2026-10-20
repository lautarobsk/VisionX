# VisionX - Modulo ALPR (Automatic License Plate Recognition)

Este es un modulo de vision artificial para la deteccion, lectura y seguimiento automatico de patentes vehiculares en tiempo real. Este componente funciona como el nodo de captura dentro del sistema de seguridad y monitoreo policial (que estoy desarrollando), enviando las detecciones al backend para su posterior analisis.

---

## 1. Idea y Funcionamiento General

El sistema implementa un pipeline jerarquico de vision artificial por etapas.

1. **Deteccion y Seguimiento de Vehiculos:**
   Se utiliza **YOLOv8** (`yolov8n.pt`) para detectar automoviles en la escena y asignarles un identificador de seguimiento persistente (`track_id`) a lo largo de los cuadros.

2. **Localizacion de la Placa Patente:**
   Sobre la region de interes (ROI) de cada vehiculo detectado, un modelo YOLO entrenado especificamente para patentes (`yolo_patentes.pt`) ubica las coordenadas exactas de la chapa patente.

3. **Preprocesamiento Optico:**
   El recorte de la patente se normaliza mediante:
   - Escalado por interpolacion cubica (3x).
   - Conversion a escala de grises.
   - Filtro Gaussiano de suavizado para eliminar ruido digital.
   - Umbralizado y binarizacion adaptativa (Otsu) para maximizar el contraste de los caracteres.

4. **Reconocimiento Optico de Caracteres (OCR):**
   Se procesa la imagen con **EasyOCR**, restringiendo la deteccion exclusivamente a caracteres alfanumericos validos mediante listas permitidas (`allowlist`).

5. **Postprocesamiento y Autocorreccion Sintactica:**
   Dado que los motores OCR suelen confundir caracteres morfologicamente similares (como `0` con `O`, `8` con `B`, o `1` con `I`), el modulo aplica reglas sintacticas segun los formatos de patente de Argentina:
   - Formato Mercosur (`AA123BB`): Determina obligatoriamente letras en las posiciones 0, 1, 5 y 6, y numeros en las posiciones 2, 3 y 4.
   - Formato antiguo argentino (`AAA123`): Determina obligatoriamente letras en las posiciones 0, 1 y 2, y numeros en las posiciones 3, 4 y 5.
   Esto corrige errores comunes de forma automatica y determinista.

6. **Sistema de Consenso y Votacion Temporal:**
   Para videos, la clase `GestorPatentes` acumula lecturas consecutivas asociadas al mismo vehiculo. Una vez alcanzado el umbral de votos configurado (por ejemplo, 3 o 5 lecturas), define la patente por mayoria estadistica, descartando lecturas anomalas o cuadros borrosos.

7. **Notificacion al Backend:**
   --- En desarrollo ---

---

## 2. Requisitos Previos

- **Sistema Operativo:** Windows 10/11
- **Python:** Version 3.10 o 3.11 recomendada (por compatibilidad directa con PyTorch, Torchvision y EasyOCR).

---

## 3. Instalacion y Puesta en Marcha

### Paso 1: Clonar el repositorio

### Paso 2: Crear y activar un entorno virtual

### Paso 3: Instalar dependencias

pip install -r requirements.txt



## 4. Guia de Uso

### Opcion A: Procesamiento de Imagenes Estaticas

Para validar la deteccion en una fotografia puntual ejecutamos el archivo captura_deteccion.py

- Este script carga una imagen de prueba desde la carpeta `media/` (por ejemplo, `auto.jpg`), recorta el vehiculo y la patente, aplica el preprocesamiento, ejecuta el OCR y muestra en pantalla la imagen con los recuadros delimitadores y la matricula inferida.
- Para cambiar la imagen de prueba, editar la variable `img` en `pruebas/captura_deteccion.py` indicando la ruta del archivo deseado.

### Opcion B: Procesamiento de Video o Flujo Continuo

Para ejecutar el pipeline completo de seguimiento, votacion e integracion (uso en videos) ejecutamos el archivo video_deteccion.py


- **Fuente de entrada:** Por defecto busca un archivo de video en la carpeta `media/` (como `Autos1.mov` o `HAVALH6.mp4`).
- **Uso con camara web en vivo:** En `video_deteccion.py`, modificar la inicializacion de la captura:
  ```python
  cap = cv2.VideoCapture(0)
  ```

---
