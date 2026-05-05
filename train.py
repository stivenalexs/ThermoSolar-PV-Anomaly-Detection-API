import torch
from ultralytics import YOLO
import multiprocessing

def verify_gpu():
    if torch.cuda.is_available():
        print(f"[INFO] GPU detectada: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("[WARNING] No se detectó GPU. El entrenamiento se ejecutará en la CPU.")
        return "cpu"

def main():
    device = verify_gpu()
    
    print("[INFO] Cargando el modelo base YOLOv9...")
    model = YOLO("yolov9s.pt") 

    print("[INFO] Iniciando el entrenamiento...")
    # Asegúrate de tener solar_data.yaml y las imágenes en la carpeta ImageSet
    try:
        results = model.train(
            data="solar_data.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            device=device,
            workers=2,
            name="yolov9_solar_anomalies",
            optimizer="auto",
            patience=10
        )
        print("[INFO] Entrenamiento finalizado.")
    except Exception as e:
        print(f"[ERROR] Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
