import cv2
import argparse
import os
from ultralytics import YOLO

def predict_image(image_path, model_path="best.pt", conf_thresh=0.25):
    if not os.path.exists(image_path):
        print(f"[ERROR] No se encontró la imagen: {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontró el modelo: {model_path}.")
        return

    print(f"[INFO] Cargando modelo desde {model_path}...")
    model = YOLO(model_path)
    
    print(f"[INFO] Procesando imagen: {image_path}...")
    results = model.predict(source=image_path, conf=conf_thresh)
    result = results[0]
    
    annotated_frame = result.plot()
    
    # En servidores headless cv2.imshow fallará, así que solo guardamos
    output_path = "resultado_inferencia.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"[INFO] Resultado guardado en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia de anomalías en paneles solares.")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen a procesar.")
    parser.add_argument("--model", type=str, default="best.pt", help="Ruta al modelo entrenado best.pt.")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza.")
    
    args = parser.parse_args()
    predict_image(args.image, args.model, args.conf)
