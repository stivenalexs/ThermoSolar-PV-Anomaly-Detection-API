# Usa una imagen oficial de Python ligera
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Evita que Python escriba archivos .pyc en el disco y asegura que los logs se muestren en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala dependencias del sistema necesarias para OpenCV y operaciones en general
# (Ultralytics y OpenCV requieren ciertas librerías del sistema compartidas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos de requerimientos primero para aprovechar el caché de capas de Docker
COPY requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el código fuente de la API y el modelo al contenedor
# Copiamos solo lo esencial para el backend para no saturar la imagen
COPY main.py .
COPY best.pt .

# Expone el puerto (Railway define dinámicamente la variable de entorno $PORT)
EXPOSE 3000

# Comando para ejecutar la aplicación usando uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-3000}"]
