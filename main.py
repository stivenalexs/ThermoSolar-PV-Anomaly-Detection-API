from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import cv2
import numpy as np
import io
import os

app = FastAPI(
    title="ThermoSolar-PV Anomaly Detection API",
    description="API para la detección de anomalías en paneles solares fotovoltaicos usando YOLOv9.",
    version="1.0.0"
)

# Variable global para el modelo
model = None
MODEL_PATH = "best.pt"

# Nombres de las clases para mayor claridad en la respuesta JSON
CLASS_NAMES = {
    0: "Single Hotspot",
    1: "Multi Hotspots",
    2: "Single Diode",
    3: "Multi Diode",
    4: "Single Bypassed Substring",
    5: "Multi Bypassed Substring",
    6: "String Open Circuit",
    7: "String Reversed Polarity"
}

@app.on_event("startup")
async def startup_event():
    """
    Se ejecuta al iniciar el servidor FastAPI.
    Carga el modelo en memoria para inferencia rápida.
    """
    global model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[INFO] Cargando el modelo YOLOv9 desde {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            print("[INFO] Modelo cargado exitosamente.")
        except Exception as e:
            print(f"[ERROR] Hubo un problema al cargar el modelo: {e}")
    else:
        print(f"[WARNING] Modelo no encontrado en {MODEL_PATH}. Asegúrate de entrenarlo primero.")

@app.post("/predict")
async def predict_anomaly(file: UploadFile = File(...)):
    """
    Endpoint de predicción que recibe una imagen y retorna las detecciones.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado o no existe aún. Por favor entrena el modelo primero.")
        
    # Verificar formato (básico)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida.")

    try:
        # Leer la imagen subida en un array de numpy
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

        # Realizar la inferencia
        # Usamos imgsz=640 por defecto, igual que en el entrenamiento
        results = model.predict(source=img, imgsz=640, conf=0.25)
        result = results[0]

        # Extraer detecciones
        detections = []
        for box in result.boxes:
            # box.xyxy: [x_min, y_min, x_max, y_max]
            # box.conf: confianza
            # box.cls: clase
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            
            # Ancho y alto para un formato de respuesta más amigable
            w = x2 - x1
            h = y2 - y1

            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, "Desconocido"),
                "confidence": conf,
                "bbox": {
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "width": w,
                    "height": h
                }
            })

        return JSONResponse(content={
            "status": "success",
            "message": f"Se detectaron {len(detections)} anomalías." if len(detections) > 0 else "No se detectaron anomalías.",
            "detections": detections
        })

    except Exception as e:
        # Manejo de cualquier error inesperado durante el procesamiento
        raise HTTPException(status_code=500, detail=f"Error interno procesando la imagen: {str(e)}")

if __name__ == "__main__":
    # Para ejecutar en desarrollo: python main.py
    # Opcional: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
