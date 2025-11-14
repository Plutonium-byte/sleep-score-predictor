import pickle

from pydantic import BaseModel, conint, confloat
from typing import Literal, Optional

from fastapi import FastAPI
import uvicorn

class SleepRecord(BaseModel):
    gender: Literal["male", "female"]
    age: conint(ge=0)
    occupation: Literal[
        "Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
        "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher"
    ]
    sleep_duration: confloat(ge=0)
    physical_activity_level: conint(ge=0)
    stress_level: conint(ge=0)
    bmi_category: Literal["Normal", "Norsmal Weight", "Overweight", "Obese"]
    heart_rate: conint(ge=0)
    daily_steps: conint(ge=0)
    sleep_disorder: Literal["Insomnia", "None", "Sleep Apnea"]
    systolic_bp: conint(ge=0)
    diastolic_bp: conint(ge=0)

class PredictResponse(BaseModel):
    sleep_score: int

app = FastAPI(title="Sleep Quality Score Prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(user):
    result = pipeline.predict(user)
    return result

@app.post("/predict")
def predict(user: SleepRecord) -> PredictResponse:
    pred = predict_single(user.model_dump())

    return PredictResponse(
        sleep_score = pred
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)