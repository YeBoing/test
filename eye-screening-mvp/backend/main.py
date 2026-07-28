import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PIL import Image, ImageStat

from database import Base, engine, get_db
from models import Patient, Screening
from schemas import FollowupUpdate, PatientCreate, PatientOut, ScreeningCreate, ScreeningOut
from services.ai_provider import analyze_fundus
from services.onnx_predictor import predict_image
from services.risk_mapper import level_from_score, recommendation

load_dotenv()
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Eye Screening MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat(),
        "ai_provider": os.getenv("AI_PROVIDER", "mock"),
        "hf_dr_model": os.getenv("HF_DR_MODEL", ""),
    }


def generate_patient_code() -> str:
    return f"P{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"


def image_qc(path: Path) -> tuple[bool, str]:
    try:
        img = Image.open(path).convert("L")
        stat = ImageStat.Stat(img)
        mean = stat.mean[0]
        stddev = stat.stddev[0]

        if mean < 25:
            return False, "图像过暗"
        if mean > 230:
            return False, "图像过曝"
        if stddev < 12:
            return False, "图像可能模糊"
        return True, "通过"
    except Exception:
        return False, "图像读取失败"


@app.post("/patients", response_model=PatientOut)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    patient = Patient(
        patient_code=generate_patient_code(),
        name=payload.name,
        gender=payload.gender,
        age=payload.age,
        phone_masked=payload.phone_masked,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@app.post("/screenings", response_model=ScreeningOut)
def create_screening(payload: ScreeningCreate, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="患者不存在")

    screening = Screening(
        patient_id=payload.patient_id,
        checkup_date=payload.checkup_date,
        status="created",
        qc_status="pending",
    )
    db.add(screening)
    db.commit()
    db.refresh(screening)
    return screening


@app.post("/screenings/{screening_id}/images")
async def upload_images(
    screening_id: int,
    left_image: UploadFile | None = File(default=None),
    right_image: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="筛查任务不存在")

    if not left_image and not right_image:
        raise HTTPException(status_code=400, detail="至少上传一张图片")

    folder = UPLOAD_DIR / str(screening_id)
    folder.mkdir(parents=True, exist_ok=True)

    qc_msgs = []
    qc_pass = True

    if left_image:
        left_path = folder / f"left_{left_image.filename}"
        left_path.write_bytes(await left_image.read())
        screening.left_image_path = str(left_path)
        ok, msg = image_qc(left_path)
        qc_msgs.append(f"左眼:{msg}")
        qc_pass = qc_pass and ok

    if right_image:
        right_path = folder / f"right_{right_image.filename}"
        right_path.write_bytes(await right_image.read())
        screening.right_image_path = str(right_path)
        ok, msg = image_qc(right_path)
        qc_msgs.append(f"右眼:{msg}")
        qc_pass = qc_pass and ok

    screening.qc_status = "pass" if qc_pass else "fail"
    screening.status = "uploaded"
    db.commit()

    return {
        "screening_id": screening.id,
        "qc_status": screening.qc_status,
        "qc_message": "；".join(qc_msgs),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    folder = UPLOAD_DIR / "predict"
    folder.mkdir(parents=True, exist_ok=True)
    save_path = folder / f"{uuid.uuid4().hex}_{Path(file.filename or 'upload').name}"
    save_path.write_bytes(await file.read())

    try:
        result = predict_image(str(save_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    return {
        "ok": True,
        "image_path": str(save_path),
        "result": result,
        "detected_diseases": result.get("detected_diseases", []),
        "num_diseases": result.get("num_diseases", 0),
        "probabilities": result.get("probabilities", {}),
    }


@app.post("/screenings/{screening_id}/analyze")
async def analyze_screening(screening_id: int, db: Session = Depends(get_db)):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="筛查任务不存在")

    if screening.qc_status != "pass":
        raise HTTPException(status_code=400, detail="图像质控未通过，请重新上传")

    if not screening.left_image_path and not screening.right_image_path:
        raise HTTPException(status_code=400, detail="缺少图像")

    patient = db.query(Patient).filter(Patient.id == screening.patient_id).first()

    screening.status = "analyzing"
    db.commit()

    try:
        result = await analyze_fundus(
            screening.left_image_path,
            screening.right_image_path,
            {
                "patient_code": patient.patient_code,
                "age": patient.age,
                "gender": patient.gender,
            },
        )

        dr_level = level_from_score(float(result["dr_score"]))
        htn_level = level_from_score(float(result["htn_score"]))

        screening.dr_score = f"{float(result['dr_score']):.4f}"
        screening.htn_score = f"{float(result['htn_score']):.4f}"
        screening.dr_risk_level = dr_level
        screening.htn_risk_level = htn_level
        screening.recommendation_text = recommendation(dr_level, htn_level)
        screening.followup_needed = "Y" if (dr_level in ["medium", "high"] or htn_level in ["medium", "high"]) else "N"
        screening.status = "done"
        screening.api_raw_result = result["raw"]
        db.commit()

        return {
            "screening_id": screening.id,
            "dr_risk_level": screening.dr_risk_level,
            "htn_risk_level": screening.htn_risk_level,
            "dr_score": screening.dr_score,
            "htn_score": screening.htn_score,
            "recommendation_text": screening.recommendation_text,
            "followup_needed": screening.followup_needed,
            "disclaimer": "本系统仅用于辅助筛查演示，不构成诊断结论，请以专科医生意见为准。",
        }
    except Exception as e:
        screening.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.get("/screenings")
def list_screenings(db: Session = Depends(get_db)):
    rows = db.query(Screening).order_by(Screening.id.desc()).all()
    output = []
    for r in rows:
        patient = db.query(Patient).filter(Patient.id == r.patient_id).first()
        output.append(
            {
                "screening_id": r.id,
                "patient_code": patient.patient_code if patient else None,
                "patient_name": patient.name if patient else None,
                "checkup_date": r.checkup_date,
                "status": r.status,
                "qc_status": r.qc_status,
                "dr_risk_level": r.dr_risk_level,
                "htn_risk_level": r.htn_risk_level,
                "followup_needed": r.followup_needed,
            }
        )
    return output


@app.get("/screenings/{screening_id}/result")
def get_result(screening_id: int, db: Session = Depends(get_db)):
    r = db.query(Screening).filter(Screening.id == screening_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="筛查任务不存在")

    return {
        "screening_id": r.id,
        "status": r.status,
        "qc_status": r.qc_status,
        "dr_risk_level": r.dr_risk_level,
        "htn_risk_level": r.htn_risk_level,
        "dr_score": r.dr_score,
        "htn_score": r.htn_score,
        "recommendation_text": r.recommendation_text,
        "followup_needed": r.followup_needed,
        "followup_status": r.followup_status,
        "followup_note": r.followup_note,
        "disclaimer": "本系统仅用于辅助筛查演示，不构成诊断结论，请以专科医生意见为准。",
    }


@app.post("/screenings/{screening_id}/followup")
def update_followup(screening_id: int, payload: FollowupUpdate, db: Session = Depends(get_db)):
    r = db.query(Screening).filter(Screening.id == screening_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="筛查任务不存在")

    r.followup_needed = payload.followup_needed
    r.followup_status = payload.followup_status
    r.followup_note = payload.followup_note
    db.commit()

    return {"ok": True, "screening_id": r.id}
