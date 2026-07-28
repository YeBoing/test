from pydantic import BaseModel, Field
from typing import Optional


class PatientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    gender: str
    age: int
    phone_masked: Optional[str] = None


class PatientOut(BaseModel):
    id: int
    patient_code: str
    name: str
    gender: str
    age: int
    phone_masked: Optional[str] = None

    class Config:
        from_attributes = True


class ScreeningCreate(BaseModel):
    patient_id: int
    checkup_date: Optional[str] = None


class ScreeningOut(BaseModel):
    id: int
    patient_id: int
    checkup_date: Optional[str] = None
    status: str
    qc_status: str
    dr_risk_level: Optional[str] = None
    htn_risk_level: Optional[str] = None
    recommendation_text: Optional[str] = None

    class Config:
        from_attributes = True


class FollowupUpdate(BaseModel):
    followup_needed: str
    followup_status: str
    followup_note: Optional[str] = None
