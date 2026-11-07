"""
Suite de tests del backend FastAPI ThermoSolar-PV.

Cubre:
- Estado de salud (/health): status == "ok" y campos extendidos (model_loaded, model_path, classes, class_count).
- Catálogo canónico de 8 clases (IDs 0..7 y orden exacto).
- Verificación estricta de clases en lifespan al cargar el modelo.
- Cabeceras de seguridad y exención del CSP restrictivo en /docs, /redoc y /openapi.json.
- Política CORS explícita (http://localhost:5173 permitido; orígenes externos rechazados).
- Validación de imágenes, dimensiones reales (image_width, image_height), inference_ms y manejo diferenciado de cv2.error vs fallos internos de inferencia.
"""

from fastapi.testclient import TestClient
import numpy as np
import pytest

import main
from main import app

client = TestClient(app)

EXPECTED_CLASSES = {
    0: "Multi Bypassed Substring",
    1: "Multi Diode",
    2: "Multi Hotspots",
    3: "Single Bypassed Substring",
    4: "Single Diode",
    5: "Single Hotspot",
    6: "String Open Circuit",
    7: "String Reversed Polarity",
}


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_extended_model_metadata_when_unloaded(monkeypatch):
    monkeypatch.setattr(main, "model", None)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert isinstance(body["model_path"], str)
    assert body["model_path"].endswith("best.pt")
    assert body["class_count"] == 8
    assert body["classes"] == {str(k): v for k, v in EXPECTED_CLASSES.items()}


def test_health_reports_model_loaded_true_when_loaded(monkeypatch):
    monkeypatch.setattr(main, "model", object())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_class_names_has_8_ids_in_canonical_order():
    assert len(main.CLASS_NAMES) == 8
    assert list(main.CLASS_NAMES.keys()) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(main.CLASS_NAMES.items()) == list(EXPECTED_CLASSES.items())


def test_resolve_model_path_resolves_relative_against_base_dir(tmp_path):
    resolved_default = main._resolve_model_path(None)
    resolved_relative = main._resolve_model_path("best.pt")
    assert resolved_default == str((main.BASE_DIR / "best.pt").resolve())
    assert resolved_relative == str((main.BASE_DIR / "best.pt").resolve())

    abs_target = tmp_path / "custom.pt"
    assert main._resolve_model_path(str(abs_target)) == str(abs_target.resolve())


def test_lifespan_fails_on_class_names_mismatch(monkeypatch, tmp_path):
    fake_pt = tmp_path / "wrong_classes.pt"
    fake_pt.write_bytes(b"dummy")

    class MismatchedYOLO:
        def __init__(self, path):
            self.names = {
                0: "Single Hotspot",
                1: "Multi Hotspots",
                2: "Single Diode",
                3: "Multi Diode",
                4: "Single Bypassed Substring",
                5: "Multi Bypassed Substring",
                6: "String Open Circuit",
                7: "String Reversed Polarity",
            }

    monkeypatch.setattr(main, "MODEL_PATH", str(fake_pt))
    monkeypatch.setattr(main, "YOLO", MismatchedYOLO)
    monkeypatch.setattr(main, "model", None)

    with pytest.raises(RuntimeError, match="Incompatibilidad de clases"):
        with TestClient(app):
            pass

    assert main.model is None


def test_lifespan_succeeds_when_class_names_match(monkeypatch, tmp_path):
    fake_pt = tmp_path / "valid_classes.pt"
    fake_pt.write_bytes(b"dummy")

    class MatchingYOLO:
        def __init__(self, path):
            self.names = dict(EXPECTED_CLASSES)

    monkeypatch.setattr(main, "MODEL_PATH", str(fake_pt))
    monkeypatch.setattr(main, "YOLO", MatchingYOLO)
    monkeypatch.setattr(main, "model", None)

    with TestClient(app) as local_client:
        resp = local_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is True


def test_predict_rejects_non_image_content_type(monkeypatch):
    monkeypatch.setattr(main, "model", object())
    response = client.post(
        "/predict",
        files={"file": ("documento.txt", b"hola mundo", "text/plain")},
    )
    assert response.status_code == 400
    assert "imagen" in response.json()["detail"].lower()


def test_predict_http_exception_propagates_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "model", None)
    response = client.post(
        "/predict",
        files={"file": ("imagen.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
    )
    assert response.status_code == 503
    assert "modelo" in response.json()["detail"].lower()


def test_predict_http_exception_for_undecodable_image_not_converted_to_500(monkeypatch):
    monkeypatch.setattr(main, "model", object())
    response = client.post(
        "/predict",
        files={"file": ("imagen.jpg", b"esto-no-es-un-bytes-valido", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "firma" in response.json()["detail"].lower()


def test_health_includes_security_headers():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"


def test_docs_and_openapi_do_not_carry_restrictive_csp():
    for path in ("/docs", "/redoc", "/openapi.json"):
        response = client.get(path)
        assert response.status_code == 200
        csp = response.headers.get("Content-Security-Policy", "")
        assert "default-src 'none'" not in csp


def test_cors_preflight_rejects_disallowed_origin():
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_allows_configured_origin():
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_predict_returns_image_dimensions_and_inference_ms(monkeypatch):
    class FakeTensor:
        def __init__(self, value):
            self._value = value

        def tolist(self):
            return list(self._value)

        def item(self):
            return self._value

    class FakeBox:
        xyxy = [FakeTensor([100.0, 50.0, 300.0, 200.0])]
        conf = [FakeTensor(0.91)]
        cls = [FakeTensor(5)]

    class FakeResult:
        boxes = [FakeBox()]

    class FakeModel:
        def predict(self, source, imgsz, conf):
            return [FakeResult()]

    fake_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main.cv2, "imdecode", lambda arr, flags: fake_img)

    response = client.post(
        "/predict",
        files={"file": ("panel.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 16, "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["image_width"] == 1280
    assert payload["image_height"] == 720
    assert isinstance(payload["inference_ms"], (int, float))
    assert payload["inference_ms"] >= 0
    assert len(payload["detections"]) == 1
    assert payload["detections"][0]["class_id"] == 5
    assert payload["detections"][0]["class_name"] == "Single Hotspot"
    assert payload["detections"][0]["bbox"]["width"] == 200.0
    assert payload["detections"][0]["bbox"]["height"] == 150.0


def test_predict_handles_cv2_error_distinctly_from_inference_500(monkeypatch):
    def raise_cv2_error(arr, flags):
        raise main.cv2.error("OpenCV internal decode failure secret")

    monkeypatch.setattr(main, "model", object())
    monkeypatch.setattr(main.cv2, "imdecode", raise_cv2_error)

    response = client.post(
        "/predict",
        files={"file": ("corrupta.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 16, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "decodificar" in response.json()["detail"].lower()
    assert "secret" not in response.text


def test_predict_500_generic_error_does_not_leak_internal_details(monkeypatch):
    class FailingModel:
        def predict(self, source, imgsz, conf):
            raise RuntimeError("secreto-interno-xyz")

    fake_img = np.zeros((640, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(main, "model", FailingModel())
    monkeypatch.setattr(main.cv2, "imdecode", lambda arr, flags: fake_img)

    response = client.post(
        "/predict",
        files={"file": ("imagen.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 16, "image/jpeg")},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Error interno procesando la imagen."
    assert "secreto-interno-xyz" not in response.text

