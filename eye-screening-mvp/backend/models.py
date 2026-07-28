from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(32), unique=True, index=True)
    name = Column(String(64), nullable=False)
    gender = Column(String(16), nullable=False)
    age = Column(Integer, nullable=False)
    phone_masked = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    screenings = relationship("Screening", back_populates="patient")


class Screening(Base):
    __tablename__ = "screenings"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    checkup_date = Column(String(32), nullable=True)
    status = Column(String(32), default="created")
    qc_status = Column(String(32), default="pending")
    left_image_path = Column(String(255), nullable=True)
    right_image_path = Column(String(255), nullable=True)

    dr_risk_level = Column(String(16), nullable=True)
    htn_risk_level = Column(String(16), nullable=True)
    dr_score = Column(String(16), nullable=True)
    htn_score = Column(String(16), nullable=True)
    recommendation_text = Column(Text, nullable=True)

    followup_needed = Column(String(4), default="N")
    followup_status = Column(String(32), default="pending")
    followup_note = Column(Text, nullable=True)

    api_raw_result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("Patient", back_populates="screenings")
