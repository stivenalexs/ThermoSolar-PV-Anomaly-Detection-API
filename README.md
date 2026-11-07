# ThermoSolar-PV Anomaly Detection

Plataforma full-stack para el diagnóstico automatizado de anomalías térmicas en paneles solares fotovoltaicos a partir de imágenes termográficas aéreas, impulsada por **YOLOv9-S** (Ultralytics), **FastAPI** y **React + TypeScript + Vite**.

---

## Arquitectura

El sistema está desacoplado en dos servicios independientes listos para despliegue en la nube:

```text
┌───────────────────────────────────────┐        HTTP POST /predict        ┌─────────────────────────────────────────┐
│       Frontend (Vercel / Vite)        │ ───────────────────────────────▶ │       Backend (Railway / Docker)        │
│  React 19 · TypeScript · Tailwind v4  │      multipart/form-data         │  FastAPI · OpenCV · Ultralytics YOLOv9  │
│  Puerto local: 5173                   │ ◀─────────────────────────────── │  Puerto local: 3000                     │
└───────────────────────────────────────┘  JSON: detections, image_width,  └─────────────────────────────────────────┘
                                           image_height, inference_ms
```

- **Backend ([`main.py`](./main.py)):** API REST sobre FastAPI y Uvicorn. Valida tamaño máximo (10 MB), tipo MIME, extensión y firma binaria (*magic bytes*) antes de decodificar la imagen con OpenCV. Ejecuta inferencia thread-safe (`threading.Lock`) con YOLOv9-S (`best.pt`), verifica en el arranque (`lifespan`) que el catálogo de clases del checkpoint coincida exactamente con `CLASS_NAMES` y devuelve las coordenadas en el espacio de píxeles original (`image_width` × `image_height`) junto con el tiempo de inferencia (`inference_ms`).
- **Frontend ([`src/App.tsx`](./src/App.tsx)):** Terminal interactiva de inspección termográfica (*SolarGuard*), con proyección exacta de *bounding boxes* sobre las dimensiones reales de la imagen sin *letterboxing*, retícula HUD dinámica, historial persistente en `localStorage` (acotado a las últimas 50 inspecciones) y panel analítico de incidencias con Recharts.
- **CLI local ([`predict.py`](./predict.py)):** Herramienta de línea de comandos para ejecutar inferencias puntuales sobre imágenes locales y exportar la termografía anotada.
- **Entrenamiento ([`notebooks/kaggle_yolov9s_640.ipynb`](./notebooks/kaggle_yolov9s_640.ipynb)):** Notebook oficial reproducible para entrenamiento y evaluación a `640×640` en Kaggle, documentado en [`notebooks/README.md`](./notebooks/README.md).

---

## Modelo y Catálogo de Clases (`best.pt`)

### Trazabilidad del checkpoint activo

| Propiedad | Valor |
| :--- | :--- |
| **Arquitectura** | `YOLOv9-S` (`yolov9s.pt`, Ultralytics `8.4.46`) |
| **Resolución de entrenamiento** | `640 × 640` px (`yolov9s_640_thermal`) |
| **Archivo** | `best.pt` (raíz del repositorio) |
| **Tamaño** | `15,254,665` bytes (~14.55 MB) |
| **SHA256** | `be6796b6f98abc3d9b46a706cacd40cf48cb114372e4898a9844067b83b6b538` |

Para verificar la integridad del checkpoint localmente:

```bash
# Linux / macOS
sha256sum best.pt

# Windows PowerShell
Get-FileHash best.pt -Algorithm SHA256
```

### Clases detectadas (orden canónico `solar_data.yaml`)

| ID | Nombre de clase | Descripción |
|---:|:---|:---|
| `0` | `Multi Bypassed Substring` | Múltiples subcadenas en bypass eléctrico |
| `1` | `Multi Diode` | Múltiples diodos de bypass activados/defectuosos |
| `2` | `Multi Hotspots` | Múltiples puntos calientes distribuidos en el módulo |
| `3` | `Single Bypassed Substring` | Una subcadena en bypass eléctrico |
| `4` | `Single Diode` | Un diodo de bypass activado/defectuoso |
| `5` | `Single Hotspot` | Punto caliente aislado en celda fotovoltaica |
| `6` | `String Open Circuit` | Cadena completa en circuito abierto |
| `7` | `String Reversed Polarity` | Cadena con polaridad invertida |

---

## Requisitos Previos

- **Backend:** Python **3.11** (recomendado y fijado en Docker y CI) y `pip`.
- **Frontend:** Node.js **20+** (probado con Node 22 LTS) y `npm`.

---

## Ejecución en Desarrollo Local

### 1. Configurar variables de entorno

Copia la plantilla `.env.example` a `.env` en la raíz del proyecto:

```bash
cp .env.example .env
```

El backend carga automáticamente `.env` mediante `python-dotenv` (`load_dotenv()`), y Vite expone al cliente únicamente las variables con prefijo `VITE_`.

### 2. Levantar el Backend (FastAPI — puerto `3000`)

```bash
# Crear y activar entorno virtual
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Instalar dependencias de producción (PyTorch CPU + Ultralytics 8.4.46 + FastAPI)
pip install -r requirements.txt

# Iniciar el servidor en http://localhost:3000
uvicorn main:app --reload --host 0.0.0.0 --port 3000
```

- Documentación interactiva Swagger UI: `http://localhost:3000/docs`
- Documentación ReDoc: `http://localhost:3000/redoc`
- Estado de salud: `http://localhost:3000/health`

### 3. Levantar el Frontend (Vite — puerto `5173`)

En otra terminal desde la raíz del proyecto:

```bash
npm install
npm run dev
```

Abre `http://localhost:5173` en el navegador.

### 4. Uso por CLI (`predict.py`)

```bash
python predict.py --image ruta/a/termografia.jpg --conf 0.25 --imgsz 640 --output resultado.jpg
```

Flags disponibles:
- `--image`: ruta a la imagen de entrada (requerido).
- `--model`: ruta al checkpoint `.pt` (por defecto `best.pt` en la raíz).
- `--conf`: umbral de confianza (por defecto `0.25`).
- `--imgsz`: resolución de inferencia (por defecto `640`).
- `--output`: ruta donde guardar la imagen anotada (por defecto `resultado_inferencia.jpg`).
- `--show`: abre una ventana gráfica con OpenCV (desactivado por defecto para entornos sin interfaz gráfica).

---

## Variables de Entorno

En local se leen desde `.env` (`python-dotenv` en backend, Vite en frontend). En producción **no se sube ningún archivo `.env`**: las variables deben configurarse directamente en el panel de **Railway** (backend) y **Vercel** (frontend).

| Variable | Ámbito | Valor por defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `ALLOWED_ORIGINS` | Backend | `http://localhost:5173` | Lista separada por comas de orígenes permitidos por CORS (ej. `https://tu-app.vercel.app`). |
| `PORT` | Backend | `3000` | Puerto de escucha HTTP (inyectado automáticamente por Railway). |
| `MODEL_PATH` | Backend | `best.pt` (resuelto contra `BASE_DIR`) | Ruta al archivo de pesos YOLOv9-S; si es relativa, se resuelve respecto al directorio de `main.py`. |
| `CONF_THRESHOLD` | Backend | `0.25` | Umbral mínimo de confianza para reportar una detección. |
| `IMGSZ` | Backend | `640` | Tamaño de entrada pasado a `model.predict`. |
| `VITE_API_URL` | Frontend | `http://localhost:3000` | URL pública base del backend (ej. `https://tu-api.up.railway.app`, sin barra final). |

---

## Especificación de la API

### `GET /health`

Verifica la disponibilidad del servicio, si el modelo está cargado en memoria y el catálogo activo de clases.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/best.pt",
  "classes": {
    "0": "Multi Bypassed Substring",
    "1": "Multi Diode",
    "2": "Multi Hotspots",
    "3": "Single Bypassed Substring",
    "4": "Single Diode",
    "5": "Single Hotspot",
    "6": "String Open Circuit",
    "7": "String Reversed Polarity"
  },
  "class_count": 8
}
```

### `POST /predict`

Recibe una imagen mediante `multipart/form-data` en el campo `file`.

**Restricciones de entrada:**
- Formatos admitidos: JPEG (`.jpg`, `.jpeg`), PNG (`.png`), GIF (`.gif`) y WebP (`.webp`), validados por cabecera `Content-Type`, extensión y *magic bytes*.
- Tamaño máximo: **10 MB**.

**Ejemplo de petición:**

```bash
curl -X POST "http://localhost:3000/predict" \
  -F "file=@termografia_panel.jpg;type=image/jpeg"
```

**Ejemplo de respuesta (`200 OK`):**

```json
{
  "status": "success",
  "message": "Se detectaron 1 anomalías.",
  "image_width": 1280,
  "image_height": 720,
  "inference_ms": 84.32,
  "detections": [
    {
      "class_id": 5,
      "class_name": "Single Hotspot",
      "confidence": 0.8941,
      "bbox": {
        "x_min": 412.5,
        "y_min": 198.0,
        "x_max": 468.5,
        "y_max": 254.0,
        "width": 56.0,
        "height": 56.0
      }
    }
  ]
}
```

> Las coordenadas `bbox` están expresadas en píxeles respecto a `image_width` e `image_height` de la imagen original subida.

**Códigos de error:**
- `400 Bad Request`: tipo de archivo no soportado, extensión inválida, firma binaria inválida, archivo corrupto (`cv2.error`) o tamaño superior a 10 MB.
- `503 Service Unavailable`: el checkpoint `best.pt` no está cargado en memoria.
- `500 Internal Server Error`: fallo interno durante la inferencia (registrado en logs del servidor sin filtrar trazas internas al cliente).

---

## Tests y Calidad de Código

```bash
# Instalar dependencias de desarrollo y ejecutar tests del backend
pip install -r requirements-dev.txt
pytest -v

# Verificación estática de tipos TypeScript y build de producción del frontend
npm run lint
npm run build
```

El flujo de integración continua en [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) ejecuta automáticamente `tsc --noEmit`, `npm run build` y `pytest` en cada `push` o `pull_request`.

---

## Despliegue en Producción

### Backend en Railway (Docker)

1. Conecta el repositorio en **Railway**. Railway detectará automáticamente el [`Dockerfile`](./Dockerfile) (`python:3.11-slim`, PyTorch CPU, usuario sin privilegios `app` y `HEALTHCHECK` contra `/health`) y respetará [`.dockerignore`](./.dockerignore) para excluir el frontend, datasets y notebooks.
2. Configura las variables de entorno en el servicio de Railway:
   - `ALLOWED_ORIGINS=https://<tu-dominio>.vercel.app`
   - `CONF_THRESHOLD=0.25`
   - `IMGSZ=640`
   *(Railway inyecta `PORT` automáticamente).*
3. Verifica tras el despliegue que `https://<tu-servicio>.up.railway.app/health` devuelva `"status": "ok"` y `"model_loaded": true`.

### Frontend en Vercel

1. Importa el repositorio en **Vercel** con el preset **Vite** (Build Command: `npm run build`, Output Directory: `dist`). El archivo [`.vercelignore`](./.vercelignore) excluye el backend Python y los pesos `.pt`, y [`vercel.json`](./vercel.json) aplica cabeceras `Cache-Control: public, max-age=31536000, immutable` en `/assets/*` y `X-Content-Type-Options: nosniff`.
2. En **Settings → Environment Variables**, añade:
   - `VITE_API_URL=https://<tu-servicio>.up.railway.app`
3. Despliega el proyecto y comprueba la comunicación CORS subiendo una imagen de prueba.

---

## Recursos Externos (Tipografías)

El archivo [`index.html`](./index.html) carga las familias tipográficas **Outfit** y **JetBrains Mono** desde **Google Fonts** (`https://fonts.googleapis.com` y `https://fonts.gstatic.com`). Si la aplicación debe operar en una red aislada (*air-gapped*) o con requisitos estrictos de privacidad sin conexiones a terceros, descarga los archivos `.woff2` localmente en `public/fonts/`, decláralos con `@font-face` en [`src/index.css`](./src/index.css) y elimina las etiquetas `<link>` a Google Fonts en `index.html` (la hoja de estilos ya incluye *fallbacks* a `sans-serif` y `monospace` del sistema).

---

## Licencia

El código propio de este repositorio se distribuye bajo la licencia **MIT** (consulta [`LICENSE`](./LICENSE)). Las dependencias de terceros y el conjunto de datos de entrenamiento mantienen sus respectivas licencias originales (p. ej. Ultralytics bajo AGPL-3.0; consulta las notas en [`notebooks/README.md`](./notebooks/README.md)).
