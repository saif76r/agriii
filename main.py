from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import predict

app = FastAPI(title="AgriNav ML API")

# Cross-Origin Resource Sharing (CORS) এনাবল করা
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# গ্লোবাল ভ্যারিয়েবলে মডেল সেভ করে রাখা
risk_model = None
crop_model = None

@app.on_event("startup")
def load_models_on_startup():
    global risk_model, crop_model
    try:
        risk_model, crop_model = predict.load_models()
        print("ML Models loaded successfully!")
    except Exception as e:
        print(f"Failed to load models: {e}")

class PredictionInput(BaseModel):
    district: str
    date: str  # ফরম্যাট: YYYYMMDD (উদাহরণ: "20260601")

@app.get("/")
def home():
    return {"status": "AgriNav ML API is online"}

@app.post("/predict")
def get_prediction(payload: PredictionInput):
    if risk_model is None or crop_model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded on server.")
    
    try:
        # predict.py এর predict_one ফাংশন কল
        result = predict.predict_one(payload.district, payload.date, risk_model, crop_model)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")