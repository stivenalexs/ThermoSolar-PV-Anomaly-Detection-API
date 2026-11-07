import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "best.pt")


def predict_image(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    conf_thresh: float = 0.25,
    imgsz: int = 640,
    output_path: str = "resultado_inferencia.jpg",
    show: bool = False,
) -> None:
    """
    Realiza la inferencia en una imagen utilizando un modelo YOLOv9 entrenado,
    imprime las detecciones por consola y guarda las bounding boxes en disco.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] No se encontró la imagen: {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontró el modelo: {model_path}. Asegúrate de entrenarlo o ubicar best.pt en la raíz.")
        return

    print(f"[INFO] Cargando modelo desde {model_path}...")
    model = YOLO(model_path)

    print(f"[INFO] Procesando imagen: {image_path} (imgsz={imgsz}, conf={conf_thresh})...")
    results = model.predict(source=image_path, imgsz=imgsz, conf=conf_thresh)
    result = results[0]

    boxes = result.boxes
    if len(boxes) == 0:
        print("[INFO] No se detectaron anomalías.")
    else:
        print(f"[INFO] Se detectaron {len(boxes)} anomalía(s):")
        for idx, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            class_name = result.names.get(cls_id, str(cls_id))
            print(
                f"  [{idx}] class_name='{class_name}' (id={cls_id}) | "
                f"conf={conf:.4f} | "
                f"bbox=[x_min={x1:.1f}, y_min={y1:.1f}, x_max={x2:.1f}, y_max={y2:.1f}]"
            )

    annotated_frame = result.plot()

    if output_path:
        cv2.imwrite(output_path, annotated_frame)
        print(f"[INFO] Resultado guardado en {output_path}")

    if show:
        cv2.imshow("Deteccion de Anomalias Solares", annotated_frame)
        print("[INFO] Presiona cualquier tecla para cerrar la ventana.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia CLI de anomalías en paneles solares (YOLOv9-S).")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen a procesar.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Ruta al modelo entrenado (por defecto best.pt en la raíz del proyecto).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral mínimo de confianza (default: 0.25).")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de entrada para inferencia (default: 640).")
    parser.add_argument(
        "--output",
        type=str,
        default="resultado_inferencia.jpg",
        help="Ruta del archivo de imagen anotada de salida (default: resultado_inferencia.jpg).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Mostrar ventana emergente con la imagen anotada (desactivado por defecto).",
    )

    args = parser.parse_args()
    predict_image(
        image_path=args.image,
        model_path=args.model,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        output_path=args.output,
        show=args.show,
    )
