# Detección de Anomalías en Paneles Solares (YOLOv9)

Este proyecto implementa un pipeline End-to-End para la detección de anomalías en paneles solares fotovoltaicos ("ThermoSolar-PV") usando el framework PyTorch y la arquitectura YOLOv9 a través de Ultralytics.
Está optimizado para ejecutarse en entornos con memoria VRAM limitada (ej. NVIDIA RTX 3050 de 4GB).

## Requisitos Previos

- Python 3.8+
- Una tarjeta gráfica NVIDIA (opcional pero altamente recomendada).

## 1. Configuración del Entorno

1. Abre una terminal (o Anaconda Prompt) y navega a la carpeta de este proyecto (`c:\Users\stive\Desktop\fotovoltaico`).
2. (Recomendado) Crea un entorno virtual:
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   ```
3. Instala PyTorch asegurando compatibilidad con tu versión de CUDA (Revisa https://pytorch.org/get-started/locally/). Por ejemplo, si tienes CUDA 11.8:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Instala el resto de las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Preparación de los Datos

El archivo `solar_data.yaml` ya está configurado para la siguiente estructura de carpetas (que debe estar en la raíz de este proyecto):
```text
fotovoltaico/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
Asegúrate de que tus imágenes y archivos de texto `.txt` correspondan correctamente antes de entrenar.

## 3. Entrenamiento

Para iniciar el entrenamiento, simplemente ejecuta:
```bash
python train.py
```
**Nota sobre VRAM (RTX 3050 4GB):** 
El script `train.py` verifica automáticamente la GPU disponible. Se ha ajustado con `batch=4` y `workers=2` para evitar desbordamientos de memoria (`CUDA Out of Memory`). Si el entrenamiento falla por memoria, intenta bajar el batch size a `batch=2`. Si es muy lento y tu GPU soporta más, puedes subirlo a `batch=8`.

El mejor modelo entrenado se guardará en `runs/detect/yolov9_solar_anomalies/weights/best.pt`.

## 4. Inferencia Local

Para probar el modelo en una sola imagen (después del entrenamiento):
```bash
python predict.py --image "test/images/alguna_imagen.jpg"
```
Esto mostrará una ventana con los bounding boxes dibujados y guardará un archivo `resultado_inferencia.jpg` en la raíz.

## 5. Levantar la API (FastAPI)

Para exponer el modelo entrenado como un servicio web:
```bash
python main.py
# o también: uvicorn main:app --reload
```
- La API estará disponible en `http://localhost:8000`.
- Puedes probar el endpoint accediendo a la documentación interactiva en `http://localhost:8000/docs`.
- Usa el endpoint `POST /predict` subiendo un archivo de imagen. Te devolverá un JSON con las coordenadas y las clases de las anomalías detectadas.
