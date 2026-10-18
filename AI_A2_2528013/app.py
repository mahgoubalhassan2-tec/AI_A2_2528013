"""FastAPI endpoints for the Musanze Harvest Intelligence Service."""
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from predict import predict

VERIFICATION_CODE = "AI-A2-8013"
app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=("Classroom decision-support API for synthetic crop-yield estimation, "
                 "high-risk classification, and farm clustering."),
    version="1.0.0",
)


class FarmInput(BaseModel):
    """Validated farm features accepted by POST /predict."""
    model_config = ConfigDict(extra="forbid")
    rainfall_mm: float = Field(ge=600, le=1400)
    avg_temp_c: float = Field(ge=12, le=26)
    soil_moisture_pct: float = Field(ge=20, le=80)
    fertilizer_kg: float = Field(ge=0, le=250)
    field_size_ha: float = Field(ge=0.2, le=4)
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]


@app.get("/health")
def health():
    return {"status": "healthy", "verification_code": VERIFICATION_CODE}


@app.get("/model-info")
def model_info():
    return {
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": VERIFICATION_CODE,
        "data_status": "synthetic classroom dataset",
        "tasks": ["yield prediction", "risk classification", "field clustering"],
        "selected_models": {
            "regression": "linear_regression",
            "classification": "logistic_regression",
            "clustering": "KMeans; K selected by silhouette score",
        },
    }


@app.post("/predict")
def predict_farm(farm: FarmInput):
    return predict(farm.model_dump())
