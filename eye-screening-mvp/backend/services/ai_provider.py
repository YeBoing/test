import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import httpx

from services.onnx_predictor import predict_image


def _deterministic_score(seed: str, salt: str) -> float:
    h = hashlib.md5(f"{seed}-{salt}".encode("utf-8")).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    return round(float(value), 4)


def _label_to_dr_severity(label: str) -> float:
    normalized = label.lower().replace("_", " ").replace("-", " ").strip()

    if normalized in {"0", "class 0", "no dr", "no diabetic retinopathy", "normal", "healthy"}:
        return 0.0
    if normalized in {"1", "class 1", "mild", "mild dr"}:
        return 0.35
    if normalized in {"2", "class 2", "moderate", "moderate dr"}:
        return 0.6
    if normalized in {"3", "class 3", "severe", "severe dr"}:
        return 0.82
    if normalized in {"4", "class 4", "proliferative", "proliferative dr", "pdr"}:
        return 0.95

    if "proliferative" in normalized:
        return 0.95
    if "severe" in normalized:
        return 0.82
    if "moderate" in normalized:
        return 0.6
    if "mild" in normalized:
        return 0.35
    if "no" in normalized and ("dr" in normalized or "retinopathy" in normalized):
        return 0.0
    if "diabetic" in normalized or "retinopathy" in normalized or normalized == "dr":
        return 0.75

    return 0.0


def _parse_hf_dr_score(predictions: list[dict]) -> float:
    if not predictions:
        raise RuntimeError("糖网模型未返回预测结果")

    weighted_score = 0.0
    total_probability = 0.0
    best_severity_score = 0.0

    for item in predictions:
        label = str(item.get("label", ""))
        probability = float(item.get("score", 0.0))
        severity = _label_to_dr_severity(label)
        weighted_score += severity * probability
        total_probability += probability
        best_severity_score = max(best_severity_score, severity * probability)

    if total_probability > 0:
        return round(max(weighted_score / total_probability, best_severity_score), 4)
    return round(best_severity_score, 4)


async def _call_hf_image_classification(image_path: str, model_id: str, token: str) -> list[dict]:
    endpoint = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Content-Type": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(endpoint, headers=headers, content=image_bytes)

    if resp.status_code == 503:
        raise RuntimeError("糖网模型正在加载，请稍后重试")
    if resp.status_code >= 400:
        raise RuntimeError(f"糖网模型调用失败: HTTP {resp.status_code} {resp.text[:200]}")

    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"糖网模型返回错误: {data['error']}")
    if not isinstance(data, list):
        raise RuntimeError("糖网模型返回格式不是分类列表")

    return data


async def analyze_fundus(
    left_image_path: Optional[str],
    right_image_path: Optional[str],
    patient_meta: dict,
) -> dict:
    provider = os.getenv("AI_PROVIDER", "mock").lower()
    seed = f"{left_image_path}-{right_image_path}-{patient_meta.get('patient_code', '')}"

    if provider == "mock":
        dr_score = _deterministic_score(seed, "dr")
        htn_score = _deterministic_score(seed, "htn")
        return {
            "provider": "mock",
            "dr_score": dr_score,
            "htn_score": htn_score,
            "raw": json.dumps({"dr_score": dr_score, "htn_score": htn_score}, ensure_ascii=False),
        }

    if provider == "onnx":
        image_paths = [p for p in [left_image_path, right_image_path] if p and Path(p).exists()]
        if not image_paths:
            raise RuntimeError("缺少眼底图像")

        try:
            result = predict_image(image_paths[0])
        except Exception as exc:
            raise RuntimeError(f"ONNX 推理失败: {exc}") from exc

        probabilities = result.get("probabilities", {})
        detected = [str(item).upper() for item in result.get("detected_diseases", [])]
        dr_score = 0.0
        if detected:
            for label in detected:
                dr_score = max(dr_score, float(probabilities.get(label, 0.0)))
        else:
            dr_score = float(probabilities.get("NORMAL", 0.0))

        htn_score = _deterministic_score(seed, "htn")
        return {
            "provider": "onnx",
            "dr_score": round(dr_score, 4),
            "htn_score": htn_score,
            "raw": json.dumps(
                {
                    "provider": "onnx",
                    "dr_score": round(dr_score, 4),
                    "htn_score": htn_score,
                    "result": result,
                },
                ensure_ascii=False,
            ),
        }

    if provider == "hf_dr":
        model_id = os.getenv("HF_DR_MODEL", "dima806/diabetic_retinopathy_detection").strip()
        token = os.getenv("HF_API_TOKEN", "").strip()

        image_paths = [p for p in [left_image_path, right_image_path] if p]
        if not image_paths:
            raise RuntimeError("缺少眼底图像")

        raw_predictions = []
        dr_scores = []
        for path in image_paths:
            predictions = await _call_hf_image_classification(path, model_id, token)
            raw_predictions.append({"image_path": path, "predictions": predictions})
            dr_scores.append(_parse_hf_dr_score(predictions))

        dr_score = round(max(dr_scores), 4)
        htn_score = _deterministic_score(seed, "htn")

        return {
            "provider": "hf_dr",
            "model_id": model_id,
            "dr_score": dr_score,
            "htn_score": htn_score,
            "raw": json.dumps(
                {
                    "provider": "hf_dr",
                    "model_id": model_id,
                    "dr_score": dr_score,
                    "htn_score": htn_score,
                    "predictions": raw_predictions,
                    "note": "糖网风险来自 Hugging Face 图像分类模型；高血压相关风险仍为MVP演示字段。",
                },
                ensure_ascii=False,
            ),
        }

    api_url = os.getenv("AI_API_URL", "").strip()
    api_key = os.getenv("AI_API_KEY", "").strip()

    if not api_url:
        raise RuntimeError("AI_API_URL 未配置")

    payload = {
        "left_image_path": left_image_path,
        "right_image_path": right_image_path,
        "patient_meta": patient_meta,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(api_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # 期望托管 API 返回 dr_score/htn_score（0~1）
    if "dr_score" not in data or "htn_score" not in data:
        raise RuntimeError("托管 API 返回缺少 dr_score 或 htn_score")

    return {
        "provider": "http",
        "dr_score": float(data["dr_score"]),
        "htn_score": float(data["htn_score"]),
        "raw": json.dumps(data, ensure_ascii=False),
    }
