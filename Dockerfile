# Imagen oficial ligera de Python 3.11
FROM python:3.11-slim

# Directorio de trabajo de la aplicación
WORKDIR /app

# Variables de entorno base para Python, Ultralytics y puerto por defecto
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=3000

# Dependencias del sistema requeridas por PyTorch (libgomp1), OpenCV y healthcheck (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario sin privilegios de root para ejecutar el servicio
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /tmp/Ultralytics && \
    chown -R app:app /app /tmp/Ultralytics

# Instalar únicamente las dependencias de producción aprovechando el caché de capas
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar únicamente el servidor de la API y el checkpoint entrenado
COPY --chown=app:app main.py best.pt ./

# Ejecutar bajo usuario no privilegiado
USER app

# Puerto expuesto (Railway inyecta dinámicamente $PORT en tiempo de ejecución)
EXPOSE 3000

# Verificación periódica del endpoint de salud
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT:-3000}/health || exit 1"

# Arrancar servidor ASGI Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-3000}"]
