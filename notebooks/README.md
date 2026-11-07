# Runbook — Kaggle YOLOv9-S 640 (ThermoSolar-PV)

Notebook principal: [`kaggle_yolov9s_640.ipynb`](./kaggle_yolov9s_640.ipynb)

Este runbook acompaña al notebook. La idea es que puedas ejecutarlo sin adivinar rutas, y que sepas qué hacer si Kaggle corta la sesión o si las métricas no cuadran.

---

## 0. Orden de clases: confirmado, manda el `data.yaml`

Hay dos órdenes en juego. **El correcto es el del `data.yaml` del dataset.** Se confirmó con `labels.jpg` (histograma de instancias + cajas etiquetadas sobre las imágenes):

| id | clase | instancias train |
|---:|---|---:|
| 0 | Multi Bypassed Substring | 357 |
| 1 | Multi Diode | 969 |
| 2 | Multi Hotspots | 5081 |
| 3 | Single Bypassed Substring | 2978 |
| 4 | Single Diode | 1844 |
| 5 | Single Hotspot | 7311 |
| 6 | String Open Circuit | 1197 |
| 7 | String Reversed Polarity | 108 |

Los hotspots son las clases dominantes (5081 + 7311 ≈ 62 %), como corresponde a un dataset térmico. Con el orden antiguo, “Single Hotspot” habría tenido solo 357 instancias y “Multi Bypassed Substring” 7311, algo que no cuadra con las cajas etiquetadas: `Single Hotspot` marca un punto caliente y `Multi Hotspots` varios.

### Orden del `best.pt` antiguo (nombres permutados)

El checkpoint antiguo conserva estos nombres dentro, pero sus **IDs de label son los mismos** que los del dataset (por eso las métricas por ID coinciden):

| id | nombre antiguo (incorrecto) |
|---:|---|
| 0 | Single Hotspot |
| 1 | Multi Hotspots |
| 2 | Single Diode |
| 3 | Multi Diode |
| 4 | Single Bypassed Substring |
| 5 | Multi Bypassed Substring |
| 6 | String Open Circuit |
| 7 | String Reversed Polarity |

**El error estaba en los nombres, no en los IDs.** El modelo entrenado sigue siendo válido; lo que estaba mal era la etiqueta que devolvía el API.

### Consecuencias

- `main.py`, `solar_data.yaml` y `src/App.tsx` ya usan el orden del dataset.
- Las métricas históricas por clase deben reinterpretarse **por ID**: la clase débil real es el **ID 1 = Multi Diode** (mAP50-95 0.346), no “Multi Hotspots” (ID 2, 0.615).
- El agregado 58.02 % sigue siendo comparable; el desglose por nombre anterior no.

### Política del notebook: `CLASS_ORDER_POLICY="dataset"`

- El notebook lee el `data.yaml`, comprueba que las **8 clases son las mismas** (normalizando camelCase, espacios, paréntesis, plurales y `Bypassed`/`bypass`) y **adopta su orden**.
- Genera `data_solar_640.yaml` con ese orden y entrena con él.
- **Descarta el `best.pt` antiguo**: no lo usa para baseline ni para inicializar. Entrena desde `yolov9s.pt`.
- Escribe `reports/class_order.json` con el orden activo, el del checkpoint y si coinciden.
- Si el orden coincide con el del checkpoint, puedes poner `RUN_BASELINE=True` para el baseline 320 vs 640.
- `CLASS_ORDER_POLICY="checkpoint"` aborta cuando hay desajuste; úsalo solo si quieres forzar la comprobación estricta.

No reordenes etiquetas “para que coincidan” sin confirmar con qué orden se generaron los labels: cambiar los nombres en el YAML no cambia los IDs de las cajas y puede empeorar todo silenciosamente.

> **Al desplegar** el `best.pt` nuevo, mantén el orden del dataset en `main.py` y `solar_data.yaml`. Si dejas el checkpoint viejo, el API ya devuelve nombres correctos porque mapea por ID.

---

## 1. Preparar el dataset en Kaggle

### Opción A — usar el dataset público

El dataset parece ser [`pkdarabi/solarpanel`](https://www.kaggle.com/datasets/pkdarabi/solarpanel) (“Thermal Solar PV Anomaly Detection Dataset”, ~7,339 imágenes, 8 clases). En el notebook:

1. **Add Data** → busca `pkdarabi/solarpanel` → añádelo.
2. Ejecuta la celda de localización. El notebook buscará automáticamente una raíz con `train/` y `valid/`, tanto en formato `ImageSet/train/images` como `train/images`.

### Opción B — subir tu propio `ImageSet`

Si tienes la carpeta completa del repo (`ImageSet/` con `train`, `valid`, `test`):

1. Kaggle → **Datasets** → **New Dataset**.
2. Sube la carpeta `ImageSet` (o un ZIP que la contenga).
3. Nombre sugerido: `thermosolar-pv-imageset`.
4. En el notebook, añade el dataset con **Add Data**.
5. Si hay varias carpetas candidatas, define en la configuración:

```python
DATASET_ROOT_OVERRIDE = Path("/kaggle/input/thermosolar-pv-imageset/ImageSet")
```

La estructura esperada es:

```text
ImageSet/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/      # opcional
  data.yaml      # opcional; el notebook lo regenera
```

También se acepta:

```text
dataset/
  images/train/
  labels/train/
  images/valid/
  labels/valid/
```

### Opción C — subir el `best.pt` antiguo (solo si el orden coincide)

El `best.pt` antiguo **no es necesario** y por defecto se descarta. Solo tiene sentido si el `data.yaml` del dataset usa el mismo orden que el checkpoint.

1. Kaggle → **Datasets** → **New Dataset**.
2. Sube `best.pt` (el del run 320).
3. Nombre sugerido: `thermosolar-best-320`.
4. Añádelo al notebook y pon `RUN_BASELINE=True`. El notebook lo evalúa a 320 y 640.

Si hay varios `.pt`, define:

```python
BASELINE_WEIGHTS = "/kaggle/input/thermosolar-best-320/best.pt"
```

Si el orden no coincide, el notebook desactiva el baseline y fuerza `INIT_WEIGHTS="yolov9s.pt"` automáticamente.

---

## 2. Configurar el notebook

En **Settings**:

- **Accelerator:** `GPU T4 x2`.
- **Internet:** `On` (necesario para descargar `yolov9s.pt` y dependencias).
- **Persistence:** si Kaggle te lo permite, activa la persistencia de `/kaggle/working` para el resume.

En la celda **1. Configuración**:

| Variable | Valor por defecto | Nota |
|---|---|---|
| `IMG_SIZE` | `640` | P0 |
| `EPOCHS` | `200` | Puede requerir 2 sesiones |
| `BATCH` | `16` | 1×T4. Para DDP: `32` |
| `DEVICE` | `"0"` | 1×T4. `"0,1"` para DDP |
| `PATIENCE` | `50` | Early stopping |
| `TIME_LIMIT_H` | `8.5` | Corte limpio antes del límite de Kaggle |
| `SAVE_PERIOD` | `10` | Checkpoints numerados |
| `CLOSE_MOSAIC` | `10` | Cierra mosaic al final |
| `CLS_PW` | `0.0` | `0.25–0.5` para el experimento B |
| `RUN_NAME` | `yolov9s_640_thermal` | Cambia por experimento |
| `INIT_WEIGHTS` | `yolov9s.pt` | **No** uses el `best.pt` antiguo si el orden no coincide |
| `RUN_BASELINE` | `False` | `True` solo si el orden del `data.yaml` coincide con el checkpoint |
| `CLASS_ORDER_POLICY` | `"dataset"` | Adopta el orden del `data.yaml`; `"checkpoint"` aborta si no coincide |
| `AUTO_RESUME` | `True` | Reanuda `last.pt` |
| `RESUME_WEIGHTS` | `None` | Ruta a un `last.pt` subido como dataset |
| `REBUILD_GROUP_SAFE_SPLIT` | `False` | Actívalo si hay fuga alta |

### Sobre `batch=16` y las dos T4

- Una T4 de 16 GB debería aguantar YOLOv9-S a 640 con batch 16 y AMP.
- Si hay OOM: baja a `BATCH=8` o `BATCH=4`. No cambies a la vez resolución y batch.
- DDP (`DEVICE="0,1"`) puede funcionar, pero en notebooks de Kaggle es más frágil. Si lo usas, `BATCH=32` (16 por GPU) y ejecuta la celda de entrenamiento en una sola celda, sin `!python` externo.

---

## 3. Orden de ejecución

1. **Entorno** (`%pip install ultralytics==8.4.46`).
2. **Configuración**.
3. **Verificación de GPU**.
4. **Localización del dataset**: revisa que imprima `train`, `val` y, si existe, `test`.
5. **Orden de clases y data.yaml**: revisa el orden activo, `class_order.json` y que `Compatible con el best.pt antiguo` sea el esperado. Si es `False`, el notebook ya desactivó el baseline y el init desde `best.pt`.
6. **Conteo y QA de labels**: revisa `class_counts.csv`, `label_qa.csv`, `class_distribution.png`.
7. **Auditoría de fuga**: revisa `leakage_summary.json` y `leakage_examples.png`.
8. **Split group-safe** (opcional): solo si la fuga es alta.
9. **Baseline 320 vs 640** (opcional, solo si el orden coincide).
10. **Entrenamiento**.
11. **Evaluación final**.
12. **Test / TTA** (opcional).
13. **ONNX** (opcional).
14. **Artefactos y descarga**.

Las celdas de análisis están pensadas para ejecutarse en orden. Si no adjuntas `best.pt` o el orden no coincide, ejecuta igualmente la celda de baseline: se saltará sola y no rompe nada. Las funciones `find_attached_pt` y `per_class_table` se definen en la celda de verificación de GPU (sección 2), así que la evaluación posterior funciona aunque esa celda no encuentre pesos.

---

## 4. Checkpoint y resume

### Si la sesión se corta o se alcanza `TIME_LIMIT_H`

- `last.pt` se guarda cada época.
- Además, cada 10 épocas se copia a `/kaggle/working/checkpoints/last.pt`.
- Vuelve a ejecutar la celda de entrenamiento: con `AUTO_RESUME=True` retoma desde `last.pt`.

### Si `/kaggle/working` se reinició

1. Abre la versión anterior del notebook y descarga `last.pt` desde **Output** (o el ZIP de artefactos).
2. Crea un Kaggle Dataset privado con `last.pt`.
3. Añádelo al notebook.
4. En configuración:

```python
AUTO_RESUME = True
RESUME_WEIGHTS = "/kaggle/input/thermosolar-resume/last.pt"
```

5. Ejecuta de nuevo la celda de entrenamiento. El notebook copia `last.pt` al run y llama `model.train(resume=True)`.

> `resume=True` restaura época, optimizer, scheduler y estado de AMP. No cambies `imgsz`, `data` ni arquitectura al reanudar: el checkpoint manda.

> **Orden de clases:** si el `last.pt` existente se entrenó con otro orden de clases, el notebook lanza `ValueError` en vez de reanudar. Borra/renombra ese run o cambia `RUN_NAME`; no mezcles órdenes.

### Reanudar un run cortado por `TIME_LIMIT_H`

Cuidado: Ultralytics guarda en `last.pt` un `args.epochs` igual a las épocas que **cupieron** en el tiempo, y un `args.time > 0`. Al reanudar, recalcula `epochs` desde `time` en cada época y lo compara con la época **absoluta**, así que el run se detendría tras una sola época (o fallaría el assert de `resume`).

El notebook ya lo corrige:

- `prepare_resume_checkpoint()` reescribe `train_args["epochs"] = RESUME_TOTAL_EPOCHS` y `train_args["time"] = 0` (con backup `last_orig.pt`).
- Se reanuda con `resume=True, time=0`.
- La parada limpia por reloj se hace con el callback `stop_after_time_limit`.
- `RESUME_KEEP_MOSAIC_CLOSED=True` evita que mosaic se reactive al reanudar.

La cosine se recalcula sobre `RESUME_TOTAL_EPOCHS`, así que puede haber un pequeño salto de LR al reanudar. Es normal.

### Si el run ya terminó

Si `last.pt` ya alcanzó `EPOCHS`, la celda imprime que se omite el entrenamiento. Puedes pasar directamente a evaluación.

---

## 5. Qué mirar en los resultados

### Baseline 320 vs 640

Solo se ejecuta si adjuntaste el `best.pt` antiguo **y** su orden coincide con el `data.yaml`. Archivo: `reports/baseline_320_vs_640.csv`.

- Si **640 > 320**, se confirma el desajuste train/infer. El modelo actual ya mejora solo con inferir a 640, y el reentrenamiento a 640 debería ayudar más.
- Si **320 > 640**, el modelo está muy sesgado a baja resolución; el reentrenamiento a 640 es aún más necesario.
- Si el orden no coincide, el baseline se omite a propósito: comparar modelos con IDs de clase desalineados no significa nada.

### Run nuevo

Archivos:

- `reports/final_per_class.csv`
- `reports/comparison_vs_320.csv`
- `reports/confusion_matrix_normalized.csv`
- `reports/confusion_matrix_counts.csv`
- `reports/final_summary.json`
- `reports/class_order.json`

La comparación `comparison_vs_320.csv` compara la referencia histórica **por ID de label**, no por nombre: si el split es el mismo, los IDs son los mismos. La columna `checkpoint_name` muestra el nombre que el checkpoint antiguo le daba a ese ID, que puede estar permutado. El mAP agregado (58.02%) solo es comparable si el split de validación es el mismo.

Criterios:

| Señal | Acción |
|---|---|
| `mAP50-95` > 58.02% | P0 funcionó; sigue con P1/P2 |
| ID 1 (etiquetado `MultiDiode` en el data.yaml) sigue < 50% mAP50 | P1: revisar la confusión con Single Diode y reformular |
| ID 7 (`String Reversed Polarity`) sigue < 50% mAP50-95 | P2: oversampling + modelo string-level (solo 16 instancias en val) |
| `precision` alta, `recall` baja | revisar umbral de confianza y PR curve |
| Fuga > 5–10% de val | reentrenar con split group-safe y reportar ambos |

> La auditoría usa MD5 (copia exacta) y pHash (robusto a brillo/escala). Los flips y rotaciones de 90°/180° pueden cambiar el pHash lo suficiente como para no detectarse: el porcentaje de fuga es una **cota inferior**, no un máximo. Si sospechas, revisa `leakage_examples.png` y considera activar `REBUILD_GROUP_SAFE_SPLIT=True` de todos modos.

### Matriz de confusión

Ultralytics la dibuja con:

- **filas = predicho**
- **columnas = real**
- última fila/columna = `background`

La alerta del handoff probablemente venía de leerla transpuesta o de comparar umbrales distintos (`conf=0.25` vs `conf=0.001`). El notebook imprime la matriz con etiquetas y exporta ambos CSV. No tomes decisiones con la matriz vieja.

---

## 6. Matriz de experimentos

| Run | Cambio | Para qué |
|---|---|---|
| **A (default)** | 640, `cls_pw=0.0`, split original | P0 limpio y comparable con 58.02% |
| **B** | 640, `cls_pw=0.25–0.5` | P2: clases raras |
| **C** | 640, split group-safe | Medir generalización real si hay fuga |
| **D (futuro)** | 5 clases base + clasificador Single/Multi | P1: corregir formulación |

Cambia **una variable a la vez** y usa `RUN_NAME` distinto. No reutilices el mismo `RUN_NAME` para experimentos distintos.

---

## 7. Tiempo y compute

| GPU | 200 epochs a 640 | Nota |
|---|---:|---|
| T4 16 GB (Kaggle) | 8–15 h | Puede requerir 2 sesiones |
| RTX 4090 | 3–6 h | RunPod/Vast.ai |
| RTX 3050 4 GB | No recomendado | No cabe a 640 con batch 16 |

`TIME_LIMIT_H=8.5` deja margen para guardar y cerrar. Si tu sesión de Kaggle permite más, súbelo, pero no lo pongas por encima del límite real.

---

## 8. Problemas frecuentes

### “No se encontró el dataset”

- Comprueba que añadiste el dataset con **Add Data**.
- Revisa que exista `train/images` y `valid/images` (o `train/labels`).
- Define `DATASET_ROOT_OVERRIDE` con la ruta exacta.
- Si el dataset está en ZIP, el notebook intenta descomprimirlo en `/kaggle/working/dataset_unzipped`.

### “El data.yaml no contiene exactamente las mismas 8 clases”

- Significa que falta o sobra una clase, o que un nombre no se pudo normalizar (`unknown` / `missing` en el mensaje).
- El notebook **no** aborta por un orden distinto: lo adopta. Solo aborta si el conjunto de clases es distinto.
- Revisa la lista impresa y ajusta `CHECKPOINT_NAMES` solo si el dataset usa un sinónimo nuevo.

### El notebook adoptó el orden del dataset (`Compatible con el best.pt antiguo: False`)

Es el comportamiento esperado y correcto:

- Los labels mandan; el `best.pt` antiguo queda descartado para baseline e init.
- Se entrena desde `yolov9s.pt`.
- Al desplegar, actualiza `CLASS_NAMES` en `main.py` y el orden de `solar_data.yaml`.

### “El last.pt existente usa otro orden de clases”

- Hay un run previo entrenado con el orden del `best.pt` antiguo y ahora el `data.yaml` usa otro orden.
- No lo reanudes: `resume=True` restauraría el orden viejo y desalinearía las clases.
- Borra o renombra `runs/<RUN_NAME>`, o cambia `RUN_NAME`, y vuelve a entrenar desde `yolov9s.pt`.

### OOM en la T4

- `BATCH=8` o `BATCH=4`.
- Mantén `IMG_SIZE=640` si el objetivo es P0; no bajes a 320.
- `cache=False`.
- Cierra otras sesiones de Kaggle.

### DDP se cuelga

- Vuelve a `DEVICE="0"` y `BATCH=16`.
- DDP en notebooks es sensible a cómo se lanza el proceso. La opción de una GPU es más lenta pero más fiable.

### No hay `test/labels`

- El notebook omite la evaluación test. Usa `val` para el reporte principal.
- Si tienes test sin labels, no lo uses para métricas; solo para inferencia cualitativa.

### Los enlaces de descarga (`FileLink`) dan 404

Kaggle **no resuelve `FileLink` con rutas absolutas**. `FileLink('/kaggle/working/...')` genera un enlace a `kaggleusercontent.com/kaggle/working/...` que da 404.

- Usa rutas **relativas a `/kaggle/working`**: `os.chdir('/kaggle/working')` + `FileLink('runs/<RUN>/weights/best.pt')`.
- La celda 12 ya lo hace y además copia `best.pt` y el ZIP a la raíz de `/kaggle/working`.
- Fallback garantizado: panel derecho → **Output** → elige el archivo → **Download**.

```python
import os, glob
from IPython.display import FileLink, display

os.chdir('/kaggle/working')
for f in glob.glob('runs/*/weights/best.pt') + glob.glob('*_artifacts.zip'):
    print(f, f"{os.path.getsize(f) / 1024**2:.1f} MB")
    display(FileLink(f))
```

### El ZIP de artefactos es muy grande

Incluye `best.pt` y `last.pt` (~15 MB cada uno). Si solo quieres el modelo, descarga `best.pt` con el `FileLink` directo.

---

## 9. Licencias y uso comercial

- **Ultralytics:** AGPL-3.0. Si despliegas el modelo/API como servicio comercial, revisa las obligaciones de AGPL o la licencia comercial de Ultralytics.
- **Dataset de Kaggle:** los metadatos dicen **CC BY-NC 4.0** (no comercial), mientras que la descripción dice MIT. Es una contradicción. Antes de un uso comercial, aclara la licencia con el autor.
- **Este notebook** no cambia esas licencias.

---

## 10. Lo que NO hace este notebook

- No usa Soup. Soup es fine-tuning de LLMs/VLMs, no un framework de detección de objetos; no calcula pérdidas de detección ni mAP.
- No usa un VLM para devolver bounding boxes.
- No cambia el framework antes de agotar resolución, datos y formulación de clases.
- No pretende que el run a 640 en una RTX 3050 de 4 GB sea el run final.
- No reemplaza la revisión manual del protocolo de etiquetado (una caja por anomalía vs una caja por panel/string).
