import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Detección de Anomalías en Paneles Solares con YOLOv9\n",
                "Este notebook contiene el pipeline End-to-End para entrenar, predecir y desplegar un modelo de detección de anomalías usando la arquitectura YOLOv9 y el dataset ThermoSolar-PV."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Instalación de Dependencias\n",
                "Instalamos las librerías necesarias. Nota: Si estás en Windows y tienes una tarjeta NVIDIA, asegúrate de instalar la versión de PyTorch compatible con CUDA."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install ultralytics fastapi uvicorn python-multipart opencv-python\n",
                "# Para PyTorch con CUDA 11.8 (descomentar si es necesario):\n",
                "# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Configuración del Dataset\n",
                "Creamos el archivo `solar_data.yaml` necesario para que Ultralytics encuentre las carpetas de imágenes y etiquetas."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "yaml_content = \"\"\"# Dataset Path Configurations\n",
                "path: ImageSet  # Ruta base relativa al directorio actual\n",
                "train: train/images  # Ruta de imágenes de entrenamiento\n",
                "val: valid/images    # Ruta de imágenes de validación\n",
                "test: test/images    # Ruta de imágenes de prueba (opcional)\n",
                "\n",
                "# Classes\n",
                "names:\n",
                "  0: Single Hotspot\n",
                "  1: Multi Hotspots\n",
                "  2: Single Diode\n",
                "  3: Multi Diode\n",
                "  4: Single Bypassed Substring\n",
                "  5: Multi Bypassed Substring\n",
                "  6: String Open Circuit\n",
                "  7: String Reversed Polarity\n",
                "\"\"\"\n",
                "\n",
                "with open('solar_data.yaml', 'w') as f:\n",
                "    f.write(yaml_content)\n",
                "\n",
                "print(\"[INFO] solar_data.yaml creado exitosamente.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Entrenamiento del Modelo\n",
                "Usaremos el modelo `yolov9s.pt` por ser ligero y óptimo para gráficas con memoria limitada (ej. RTX 3050 de 4GB VRAM)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from ultralytics import YOLO\n",
                "\n",
                "# Verificar GPU\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "print(f\"[INFO] Entrenando en: {device}\")\n",
                "\n",
                "# Cargar modelo\n",
                "model = YOLO('yolov9s.pt')\n",
                "\n",
                "# Iniciar entrenamiento\n",
                "results = model.train(\n",
                "    data=\"solar_data.yaml\",\n",
                "    epochs=50,\n",
                "    imgsz=640,\n",
                "    batch=4,\n",
                "    device=device,\n",
                "    workers=2,\n",
                "    name=\"yolov9_solar_anomalies_nb\"\n",
                ")\n",
                "\n",
                "print(\"[INFO] Entrenamiento finalizado.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Inferencia y Visualización\n",
                "Probamos el modelo entrenado con una imagen de validación/prueba."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import cv2\n",
                "import matplotlib.pyplot as plt\n",
                "from ultralytics import YOLO\n",
                "\n",
                "# Ajusta esta ruta a una imagen real de tu dataset\n",
                "image_path = \"ImageSet/test/images/alguna_imagen.jpg\"\n",
                "model_path = \"runs/detect/yolov9_solar_anomalies_nb/weights/best.pt\"\n",
                "\n",
                "try:\n",
                "    # Cargar modelo entrenado\n",
                "    best_model = YOLO(model_path)\n",
                "    \n",
                "    # Predecir\n",
                "    results = best_model.predict(source=image_path, conf=0.25)\n",
                "    \n",
                "    # Obtener imagen con Bounding Boxes\n",
                "    annotated_img = results[0].plot()\n",
                "    \n",
                "    # Convertir BGR a RGB para matplotlib\n",
                "    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)\n",
                "    \n",
                "    # Mostrar\n",
                "    plt.figure(figsize=(10, 10))\n",
                "    plt.imshow(annotated_img_rgb)\n",
                "    plt.axis('off')\n",
                "    plt.show()\n",
                "except Exception as e:\n",
                "    print(f\"No se pudo ejecutar la inferencia. Asegúrate de que el modelo exista y la ruta de la imagen sea correcta. Error: {e}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Levantar API FastAPI (Opcional)\n",
                "Para desplegar la API desde Jupyter se necesita `nest_asyncio`. El siguiente código arranca el servidor en el puerto 8000."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install nest-asyncio\n",
                "\n",
                "import nest_asyncio\n",
                "import uvicorn\n",
                "from fastapi import FastAPI\n",
                "\n",
                "# Permitir bucles de eventos asíncronos anidados en Jupyter\n",
                "nest_asyncio.apply()\n",
                "\n",
                "# Aquí puedes cargar tu archivo main.py o iniciar la app que definimos allí\n",
                "from main import app\n",
                "\n",
                "print(\"Abriendo el servidor en http://localhost:8000 ...\")\n",
                "print(\"Abre http://localhost:8000/docs para ver el Swagger UI.\")\n",
                "\n",
                "try:\n",
                "    # uvicorn.run(app, host=\"0.0.0.0\", port=8000) # Descomentar para ejecutar\n",
                "    print(\"Servidor FastAPI listo para ser ejecutado.\")\n",
                "except KeyboardInterrupt:\n",
                "    print(\"Servidor detenido.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('c:/Users/stive/Desktop/fotovoltaico/pipeline_yolov9.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)
