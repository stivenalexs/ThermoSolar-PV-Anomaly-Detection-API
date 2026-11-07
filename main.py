from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import threading
import time
from typing import AsyncIterator

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from ultralytics import YOLO
import uvicorn

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Variable global para el modelo y lock de inferencia (YOLO no es thread-safe)
model = None
_predict_lock = threading.Lock()

def _resolve_model_path(raw_path: str | None) -> str:
    """
    Resuelve rutas relativas contra BASE_DIR (directorio de main.py) para que
    MODEL_PATH="best.pt" funcione con independencia del directorio de trabajo actual.
    """
    cleaned = (raw_path or "").strip() or "best.pt"
    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return str(candidate.resolve())


MODEL_PATH = _resolve_model_path(os.environ.get("MODEL_PATH"))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))

# Nombres de las clases (orden canónico del dataset ThermoSolar-PV)
CLASS_NAMES = {
    0: "Multi Bypassed Substring",
    1: "Multi Diode",
    2: "Multi Hotspots",
    3: "Single Bypassed Substring",
    4: "Single Diode",
    5: "Single Hotspot",
    6: "String Open Circuit",
    7: "String Reversed Polarity",
}

# Allowlist explícita de orígenes del frontend (CORS sin "*").
# En producción, configurar la variable de entorno ALLOWED_ORIGINS con el/los
# dominios reales del frontend, p. ej.: ALLOWED_ORIGINS="https://<dominio>.vercel.app"
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Frontend local (Vite)
]
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)).split(",")
    if origin.strip()
]

DOCS_PATHS = {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
DOCS_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "worker-src 'self' blob:; "
    "frame-ancestors 'none'"
)
STRICT_CSP = "default-src 'none'; frame-ancestors 'none'"


def _normalize_model_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {idx: str(v) for idx, v in enumerate(raw_names)}
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Se ejecuta al iniciar el servidor FastAPI (patrón lifespan, reemplazo de on_event).

    Carga el modelo en memoria para inferencia rápida y verifica que el mapa de
    clases del checkpoint coincida exactamente con CLASS_NAMES.
    """
    global model
    if os.path.exists(MODEL_PATH):
        try:
            logger.info("Cargando el modelo YOLOv9 desde %s", MODEL_PATH)
            loaded_model = YOLO(MODEL_PATH)
        except Exception:
            logger.error("Hubo un problema al cargar el modelo desde %s.", MODEL_PATH, exc_info=True)
            raise

        model_names = _normalize_model_names(getattr(loaded_model, "names", None))
        if len(model_names) != len(CLASS_NAMES) or model_names != CLASS_NAMES:
            logger.error(
                "Las clases del modelo (%s) no coinciden con CLASS_NAMES (%s). Abortando arranque.",
                model_names,
                CLASS_NAMES,
            )
            model = None
            raise RuntimeError(
                f"Incompatibilidad de clases en el modelo: se esperaban {CLASS_NAMES}, "
                f"pero el checkpoint contiene {model_names}."
            )

        model = loaded_model
        logger.info("Modelo cargado y verificado exitosamente (%d clases).", len(CLASS_NAMES))
    else:
        logger.warning("Modelo no encontrado en %s. Asegúrate de subirlo o entrenarlo.", MODEL_PATH)

    yield


app = FastAPI(
    title="ThermoSolar-PV Anomaly Detection API",
    description="API para la detección de anomalías en paneles solares fotovoltaicos usando YOLOv9.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configurar CORS para permitir peticiones solo desde la allowlist explícita del frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Middleware ASGI que añade cabeceras de seguridad recomendadas a las respuestas.
    Excluye las rutas de documentación OpenAPI del CSP estricto default-src 'none'
    para permitir la carga de Swagger UI y ReDoc.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path in DOCS_PATHS:
        response.headers["Content-Security-Policy"] = DOCS_CSP
    else:
        response.headers["Content-Security-Policy"] = STRICT_CSP
    return response


@app.get("/")
async def root():
    return {
        "message": (
            "ThermoSolar-PV Anomaly Detection API is running. "
            "Documentation available at /docs and /redoc; service status at /health."
        )
    }


@app.get("/health")
async def health():
    """
    Endpoint de salud del servicio: reporta estado operativo, carga del modelo y catálogo de clases.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "classes": CLASS_NAMES,
        "class_count": len(CLASS_NAMES),
    }


# Configuración de validación de archivos para POST /predict.
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _validate_image_upload(file: UploadFile) -> bytes:
    """
    Valida el archivo subido antes de decodificar/inferir.

    - Tamaño máximo permitido: 10 MB.
    - Content-type permitido: image/jpeg, image/png, image/gif, image/webp.
    - Extensión permitida: .jpg, .jpeg, .png, .gif, .webp.
    - Magic bytes: JPEG (FF D8 FF), PNG (89 50 4E 47 0D 0A 1A 0A),
      GIF (GIF87a/GIF89a) o WEBP (RIFF....WEBP).

    Retorna los bytes de la imagen ya validados.
    """
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida.")

    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="La extensión del archivo no es válida. Formatos permitidos: .jpg, .jpeg, .png, .gif, .webp.",
        )

    # Tamaño máximo permitido (10 MB). Se usa `file.size` de Starlette si está disponible.
    file_size = getattr(file, "size", None)
    if file_size is not None and file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="El archivo subido excede el tamaño máximo permitido de 10 MB.")

    image_bytes = file.file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="El archivo subido excede el tamaño máximo permitido de 10 MB.")

    magic_bytes = image_bytes[:12]
    is_jpeg = magic_bytes[0:3] == b"\xff\xd8\xff"
    is_png = magic_bytes[0:8] == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    is_gif = magic_bytes[0:6] in (b"GIF87a", b"GIF89a")
    is_webp = magic_bytes[0:4] == b"RIFF" and magic_bytes[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_gif or is_webp):
        raise HTTPException(
            status_code=400,
            detail="El archivo no tiene una firma de imagen válida o el formato no está soportado.",
        )

    return image_bytes


@app.post("/predict")
def predict_anomaly(file: UploadFile = File(...)):
    """
    Endpoint de predicción que recibe una imagen y retorna las detecciones junto con
    las dimensiones reales de la imagen decodificada y el tiempo de inferencia.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado. Asegúrate de que best.pt existe en el servidor.",
        )

    try:
        # Validar el archivo subido (content-type, extensión, tamaño y magic bytes) antes de decodificar/inferir.
        image_bytes = _validate_image_upload(file)

        # Leer la imagen subida en un array de numpy
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

        image_height, image_width = int(img.shape[0]), int(img.shape[1])

        # Realizar la inferencia protegida por lock (Ultralytics YOLO no es thread-safe)
        infer_start = time.perf_counter()
        with _predict_lock:
            results = model.predict(source=img, imgsz=IMGSZ, conf=CONF_THRESHOLD)
        inference_ms = round((time.perf_counter() - infer_start) * 1000.0, 2)

        result = results[0]

        # Extraer detecciones
        detections = []
        for box in result.boxes:
            # box.xyxy: [x_min, y_min, x_max, y_max] en coordenadas de la imagen original
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())

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
                    "height": h,
                },
            })

        return JSONResponse(
            content={
                "status": "success",
                "message": (
                    f"Se detectaron {len(detections)} anomalías."
                    if len(detections) > 0
                    else "No se detectaron anomalías."
                ),
                "image_width": image_width,
                "image_height": image_height,
                "inference_ms": inference_ms,
                "detections": detections,
            }
        )

    except HTTPException:
        # Las HTTPException son errores controlados del endpoint: se propagan tal cual.
        raise
    except cv2.error:
        logger.error("Error de OpenCV al decodificar o procesar la imagen.", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Error al decodificar o procesar los datos de la imagen.",
        )
    except Exception:
        # Error interno de inferencia: loguear server-side con exc_info sin exponer detalles al cliente.
        logger.error("Fallo interno durante la inferencia del modelo.", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno procesando la imagen.")


if __name__ == "__main__":
    # Importante: Bind a 0.0.0.0 y usar el puerto de la variable de entorno o 3000
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
