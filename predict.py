import cv2
import argparse
import os
from ultralytics import YOLO

def predict_image(image_path, model_path="runs/detect/yolov9_solar_anomalies/weights/best.pt", conf_thresh=0.25):
    """
    Realiza la inferencia en una imagen utilizando un modelo YOLOv9 entrenado
    y dibuja las bounding boxes.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] No se encontró la imagen: {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontró el modelo: {model_path}. Asegúrate de entrenarlo primero.")
        return

    # Cargar el modelo entrenado
    print(f"[INFO] Cargando modelo desde {model_path}...")
    model = YOLO(model_path)
    
    # Realizar inferencia
    print(f"[INFO] Procesando imagen: {image_path}...")
    results = model.predict(source=image_path, conf=conf_thresh)
    
    # El resultado es una lista (un elemento por imagen de entrada)
    result = results[0]
    
    # Dibujar las detecciones en la imagen original
    annotated_frame = result.plot()
    
    # Mostrar la imagen resultante
    cv2.imshow("Deteccion de Anomalias Solares", annotated_frame)
    print("[INFO] Presiona cualquier tecla para cerrar la ventana.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Opcional: Guardar la imagen de salida
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
