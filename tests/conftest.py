"""
Configuración de pytest para la suite de tests del backend.

Si `ultralytics` está instalado (entorno estándar Python 3.11 / CI), se utiliza el paquete real.
En entornos donde PyTorch aún no publica ruedas precompiladas (p. ej. Python 3.15 local),
provee un stub mínimo de `ultralytics.YOLO` para ejercitar toda la capa HTTP, validaciones,
CORS, CSP, lifespan y OpenCV de `main.py`.
"""

import importlib.util
import sys
import types

if importlib.util.find_spec("ultralytics") is None:
    _stub = types.ModuleType("ultralytics")

    class _StubYOLO:
        def __init__(self, *args, **kwargs):
            self.names = {
                0: "Multi Bypassed Substring",
                1: "Multi Diode",
                2: "Multi Hotspots",
                3: "Single Bypassed Substring",
                4: "Single Diode",
                5: "Single Hotspot",
                6: "String Open Circuit",
                7: "String Reversed Polarity",
            }

    _stub.YOLO = _StubYOLO  # type: ignore[attr-defined]
    sys.modules["ultralytics"] = _stub

