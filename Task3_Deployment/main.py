from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load saved model
model = joblib.load("model.joblib")

# Create FastAPI app
app = FastAPI()

# Define input format using pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
