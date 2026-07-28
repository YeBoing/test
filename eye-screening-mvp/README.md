# 眼底筛查 MVP（3-5 天演示版）

体检中心场景：托管 API（可替换）+ 糖网/高血压相关风险提示。

## 1. 项目结构

- `backend/` FastAPI + SQLite
- `frontend/` Streamlit

## 2. 快速启动

### 后端

```bash
cd eye-screening-mvp/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 前端

```bash
cd eye-screening-mvp/frontend
pip install -r requirements.txt
streamlit run app.py
```

默认后端地址：`http://localhost:8000`

## 3. 环境变量

复制根目录 `.env.example` 为 `.env`（可选）。

- `AI_PROVIDER=mock`（默认）
- `AI_PROVIDER=onnx`（启用本地 ONNX 模型）
- `AI_PROVIDER=hf_dr`（启用 Hugging Face 糖网图像分类模型）
- `AI_API_URL=`（接真实托管 API 时填）
- `AI_API_KEY=`（接真实托管 API 时填）
- `HF_DR_MODEL=dima806/diabetic_retinopathy_detection`（糖网模型 ID，可替换）
- `HF_API_TOKEN=`（Hugging Face Token，建议填写）
- `ONNX_MODEL_PATH=models/ocunet.onnx`（本地 ONNX 模型路径）
- `ONNX_METADATA_PATH=models/ocunet_metadata.json`（模型标签元数据路径）

### 启用真实糖网模型

`.env` 示例：

```dotenv
AI_PROVIDER=hf_dr
HF_DR_MODEL=dima806/diabetic_retinopathy_detection
HF_API_TOKEN=你的_HUGGINGFACE_TOKEN
BACKEND_URL=http://localhost:8000
```

说明：

- 现在默认支持本地 ONNX 模型推理。将 `ocunet.onnx` 和 `ocunet_metadata.json` 放在后端的 `models/` 目录下，即可通过 `/predict` 或筛查流程调用。
- 糖网风险来自 Hugging Face 图像分类模型。
- 高血压相关风险目前仍为 MVP 演示字段，暂不作为真实医学模型结论。
- 不同 Hugging Face 模型标签可能不同，后端已兼容 `0-4`、`mild/moderate/severe/proliferative`、`no DR` 等常见标签。
- 如果模型首次调用返回“正在加载”，稍后重试即可。

验证当前模式：

```bash
curl http://localhost:8000/health
```

## 4. 免责声明

本系统仅用于辅助筛查演示，不构成诊断结论，最终请以专科医生意见为准。

## 5. MVP 已覆盖

- 患者建档
- 筛查任务创建
- 眼底图片上传
- 基础质控（模糊/过暗提醒）
- 风险分级（无/低/中/高）
- 建议动作生成
- 记录列表与详情
- 随访状态记录
