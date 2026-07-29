from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import re

# -------------------------------------------------------
# Load Trained Pipeline
# -------------------------------------------------------

model = joblib.load("model.pkl")

# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Mini Project B - FastAPI + Scikit-Learn Pipeline",
    version="1.0.0"
)

# -------------------------------------------------------
# Input Schema
# -------------------------------------------------------

class Passenger(BaseModel):
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


# -------------------------------------------------------
# Helper Function
# -------------------------------------------------------

def extract_title(name: str) -> str:
    """
    Extract passenger title from name.
    """

    match = re.search(r",\s*([^\.]+)\.", name)

    if match:
        title = match.group(1).strip()
    else:
        title = "Rare"

    # Same mapping used during training

    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don",
        "Dr", "Major", "Rev", "Sir",
        "Jonkheer", "Dona"
    ]

    if title in rare_titles:
        title = "Rare"

    elif title == "Mlle":
        title = "Miss"

    elif title == "Ms":
        title = "Miss"

    elif title == "Mme":
        title = "Mrs"

    return title


# -------------------------------------------------------
# Root Endpoint
# -------------------------------------------------------

@app.get("/")
def home():

    return {
        "Project": "Mini Project B",
        "Title": "Titanic Survival Prediction API",
        "Status": "Running"
    }


# -------------------------------------------------------
# Health Check
# -------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# -------------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------------

@app.post("/predict")
def predict(passenger: Passenger):

    family_size = passenger.SibSp + passenger.Parch + 1

    is_alone = 1 if family_size == 1 else 0

    title = extract_title(passenger.Name)

    input_df = pd.DataFrame([{

        "Pclass": passenger.Pclass,
        "Sex": passenger.Sex,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Embarked": passenger.Embarked,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Title": title

    }])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    survival_probability = probability[1]

    return {

        "prediction": int(prediction),

        "result": (
            "Survived"
            if prediction == 1
            else "Did Not Survive"
        ),

        "survival_probability": round(
            survival_probability * 100,
            2
        ),

        "engineered_features": {

            "FamilySize": family_size,
            "IsAlone": is_alone,
            "Title": title

        }

    }