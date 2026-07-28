import json
import os
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

from PIL import Image, ImageOps


class ONNXImageClassifier:
    def __init__(self, model_path: str | None = None, metadata_path: str | None = None):
        base_dir = Path(__file__).resolve().parents[1]
        default_model = base_dir / "models" / "ocunet.onnx"
        default_metadata = base_dir / "models" / "ocunet_metadata.json"
        self.model_path = Path(model_path or os.getenv("ONNX_MODEL_PATH", str(default_model))).resolve()
        self.metadata_path = Path(metadata_path or os.getenv("ONNX_METADATA_PATH", str(default_metadata))).resolve()
        self.session = None
        self.input_name = None
        self.output_name = None
        self.labels: List[str] = []
        self.ready = False
        self.error_message = ""
        self._load()

    def _load(self) -> None:
        if ort is None:
            self.error_message = "onnxruntime 未安装"
            return
        if np is None:
            self.error_message = "numpy 未安装"
            return
        if not self.model_path.exists():
            self.error_message = f"ONNX 模型文件不存在: {self.model_path}"
            return

        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if not inputs or not outputs:
                self.error_message = "模型输入/输出信息无效"
                return

            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
            self.labels = self._load_labels()
            self.ready = True
        except Exception as exc:  # pragma: no cover
            self.error_message = f"加载 ONNX 模型失败: {exc}"

    def _load_labels(self) -> List[str]:
        if self.metadata_path.exists():
            try:
                data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                for key in ["labels", "class_names", "classes"]:
                    value = data.get(key)
                    if isinstance(value, list) and value:
                        return [str(item) for item in value]
            except Exception:
                pass

        return [
            "NORMAL",
            "DR",
            "GLAUCOMA",
            "ARMD",
            "BRVO",
            "CRVO",
            "AION",
            "HYPEREMIA",
            "MYOPIA",
            "OTHER",
        ]

    def _prepare_input(self, image_path: str) -> Dict[str, np.ndarray]:
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.resize(img, (224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4:
            if shape[1] in (1, 3):
                arr = np.transpose(arr, (2, 0, 1))
            else:
                arr = np.transpose(arr, (1, 2, 0))
            arr = np.expand_dims(arr, axis=0)
        else:
            arr = np.expand_dims(arr, axis=0)

        return {self.input_name: arr}

    def _postprocess(self, raw_output) -> List[float]:
        arr = np.asarray(raw_output)
        if arr.ndim == 0:
            probs = np.array([float(arr)], dtype=np.float32)
        elif arr.ndim == 1:
            probs = arr.astype(np.float32)
        elif arr.ndim == 2:
            if arr.shape[0] == 1:
                probs = arr[0].astype(np.float32)
            elif arr.shape[1] == 1:
                probs = arr[:, 0].astype(np.float32)
            else:
                probs = arr[0].astype(np.float32)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            probs = arr[0].astype(np.float32)
        else:
            probs = arr.reshape(-1).astype(np.float32)

        if probs.ndim > 1:
            probs = probs.reshape(-1)

        if probs.size == 0:
            return []

        if np.all((probs >= 0) & (probs <= 1)):
            normalized = probs
        else:
            normalized = 1.0 / (1.0 + np.exp(-probs))

        if len(self.labels) != len(normalized):
            if len(normalized) < len(self.labels):
                padding = np.zeros(len(self.labels) - len(normalized), dtype=np.float32)
                normalized = np.concatenate([normalized, padding])
            else:
                normalized = normalized[: len(self.labels)]

        return [float(item) for item in normalized.tolist()]

    def predict(self, image_path: str) -> Dict[str, object]:
        if not self.ready:
            raise RuntimeError(self.error_message)

        inputs = self._prepare_input(image_path)
        outputs = self.session.run(None, inputs)
        probabilities = self._postprocess(outputs[0])

        scored = {}
        for idx, label in enumerate(self.labels):
            scored[str(label).upper()] = probabilities[idx] if idx < len(probabilities) else 0.0

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        detected = [name for name, score in ranked if score >= 0.5 and name != "NORMAL"]
        if not detected and ranked:
            top_name, top_score = ranked[0]
            if top_name == "NORMAL" and top_score >= 0.5:
                detected = ["NORMAL"]
            else:
                detected = [top_name]

        return {
            "provider": "onnx",
            "model_path": str(self.model_path),
            "detected_diseases": detected,
            "num_diseases": len(detected),
            "probabilities": scored,
            "all_predictions": [
                {
                    "label": label,
                    "probability": score,
                    "full_name": label,
                    "description": f"{label} 置信度 {score:.2f}",
                }
                for label, score in ranked
            ],
        }


_runner: ONNXImageClassifier | None = None


def get_model_runner() -> ONNXImageClassifier:
    global _runner
    if _runner is None:
        _runner = ONNXImageClassifier()
    return _runner


def predict_image(image_path: str) -> Dict[str, object]:
    runner = get_model_runner()
    if not runner.ready:
        raise RuntimeError(runner.error_message or "ONNX 模型不可用")
    return runner.predict(image_path)
