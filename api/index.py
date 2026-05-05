from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI()

# Variable global para el modelo
model = None
# En Vercel, los archivos deben estar en la misma carpeta o ruta relativa correcta
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best.pt")

CLASS_NAMES = {
    0: "Single Hotspot", 1: "Multi Hotspots", 2: "Single Diode",
    3: "Multi Diode", 4: "Single Bypassed Substring",
    5: "Multi Bypassed Substring", 6: "String Open Circuit",
    7: "String Reversed Polarity"
}

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            # Intentar buscar en raíz si falla el Join
            if os.path.exists("best.pt"):
                model = YOLO("best.pt")
    return model

@app.get("/api")
async def root():
    return {"status": "ok", "model_loaded": get_model() is not None}

@app.post("/api/predict")
async def predict_anomaly(file: UploadFile = File(...)):
    m = get_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Modelo no encontrado en el servidor.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen.")

    try:
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Error decodificando imagen.")

        results = m.predict(source=img, imgsz=640, conf=0.25)
        result = results[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, "Desconocido"),
                "confidence": conf,
                "bbox": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2, "width": x2-x1, "height": y2-y1}
            })

        return JSONResponse(content={"status": "success", "detections": detections})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
