import torch
from ultralytics import YOLO

def verify_gpu():
    """
    Verifica si CUDA (GPU) está disponible.
    """
    if torch.cuda.is_available():
        print(f"[INFO] GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Cantidad de memoria VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return "cuda"
    else:
        print("[WARNING] No se detectó GPU. El entrenamiento se ejecutará en la CPU, lo cual es muy lento.")
        return "cpu"

def main():
    # 1. Verificar dispositivo
    device = verify_gpu()
    
    # 2. Cargar el modelo base YOLOv9
    # Usamos yolov9s.pt (small) que es ideal para 4GB de VRAM y evita conflictos.
    print("[INFO] Cargando el modelo base YOLOv9...")
    model = YOLO("yolov9s.pt") 

    # 3. Configuración del entrenamiento
    # Optimizaciones para RTX 3050 (4GB VRAM):
    # - batch=4 o 8 (evitar CUDA Out Of Memory).
    # - imgsz=640 (tamaño estándar, redimensionar si es necesario puede liberar memoria).
    # - device=device (asegurar uso de GPU).
    # - workers=2 (reducir los procesos de carga de datos para ahorrar RAM/VRAM).
    
    print("[INFO] Iniciando el entrenamiento...")
    results = model.train(
        data="solar_data.yaml", # Archivo de configuración creado previamente
        epochs=50,              # Ajusta según el tiempo disponible
        imgsz=640,              # Resolución de las imágenes
        batch=4,                # Importante: Lote pequeño para 4GB VRAM
        device=device,          # Usa 'cuda' si está disponible
        workers=2,              # Menos workers previenen colapso de memoria
        name="yolov9_solar_anomalies", # Nombre de la carpeta de resultados
        optimizer="auto",       # El optimizador automático (suele usar AdamW o SGD)
        patience=10             # Early stopping
    )
    
    print("[INFO] Entrenamiento finalizado. El mejor modelo se encuentra en 'runs/detect/yolov9_solar_anomalies/weights/best.pt'")

if __name__ == "__main__":
    # Necesario en Windows para multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
