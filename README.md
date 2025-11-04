# 🧠 Detección y Seguimiento de Personas y Vehículos con YOLO + Entrenamiento de Detector de Matrículas

Este proyecto desarrolla un **prototipo de visión artificial** capaz de **detectar y seguir personas y vehículos en vídeos reales**, y además **entrenar un modelo personalizado para la detección de matrículas**, usando modelos **YOLO** y aceleración con **GPU AMD (DirectML)**.

---

## Objetivos

1. Procesar varios vídeos propios detectando **personas y vehículos**.  
2. Realizar **seguimiento (tracking)** de los objetos entre fotogramas.  
3. Guardar los resultados en un **vídeo anotado** y un **CSV con datos de detección**.  
4. Entrenar un modelo YOLO para **detectar matrículas** personalizadas.

---

## Tecnologías utilizadas

- **Python 3.11+**  
- **OpenCV** → Lectura y escritura de vídeos, visualización de resultados.  
- **Ultralytics YOLO (v8 / v11)** → Detección y tracking de objetos.  
- **Torch / Torch DirectML** → Aceleración con GPU AMD o CPU.  
- **CSV / defaultdict** → Registro de resultados.  

---

## ⚙️ Instalación

1. **Clonar el repositorio o descargar el proyecto:**
   ```bash
   git clone https://github.com/NicolasReyAlonso/Entrega_Prac_4_VC.git
   cd yolo-tracking-matriculas

2. **Instalar dependencias:**

   ```bash
   pip install ultralytics opencv-python torch torchvision torchaudio torch-directml
   ```

3. **Descargar los modelos YOLO preentrenados:**

   * Por defecto el script usa `yolo11n.pt` o `yolov8n.pt`.
   * Estos se descargan automáticamente la primera vez que se ejecuta el código.

---

## 🎬 Uso

### 1. Detección y seguimiento en vídeos

Edita las variables del script principal (`video_path`, `output_video_path`, etc.) y ejecuta.

El script:

* Detecta **personas (class 0)** y **coches (class 2)**.
* Realiza **seguimiento de IDs** a lo largo de los fotogramas.
* Genera:

  * Un vídeo anotado: `salida_yolo_tracking_sinOCR.mp4`
  * Un CSV con las detecciones: `detecciones_yolo_tracking_sinOCR.csv`
* Muestra por pantalla los totales detectados por clase.

El resultado se encuentra en el video **VC_P4**

**Ejemplo de salida CSV:**

| fotograma | tipo_objeto | confianza | id_tracking | x1  | y1  | x2  | y2  |
| --------- | ----------- | --------- | ----------- | --- | --- | --- | --- |
| 1         | person      | 0.88      | 3           | 140 | 220 | 300 | 600 |
| 1         | car         | 0.93      | 5           | 420 | 250 | 680 | 580 |

---

### 2. Entrenamiento del modelo de matrículas

Se entrenó un modelo **YOLOv8 nano** (`yolov8n.pt`) para detectar matrículas de vehículos utilizando la librería **Ultralytics** y aceleración por GPU con **DirectML**.

### Configuración
- Imágenes: `416×416`
- Épocas: `10`
- Batch size: `4`
- Dispositivo: `DirectML` (`torch_directml`)
- Nombre del experimento: `matriculas_detector2`

### Dataset
Se utilizó un conjunto de datos de matrículas disponible en [Kaggle](https://www.kaggle.com/) con anotaciones en formato YOLO, descrito en el archivo `data.yaml`.

### Resultado
El modelo entrenado se guarda en `runs/detect/matriculas_detector2/` y está listo para realizar inferencias sobre imágenes o vídeos que contengan matrículas.






