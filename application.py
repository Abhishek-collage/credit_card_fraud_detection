import os
import joblib
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Credit Card Fraud Detection System")

# Templates setup
templates = Jinja2Templates(directory="templates")
templates.env.cache = None

# Import config path safely
from config.config_path import MODEL_OUTPUT_PATH

# Global model variable
model = None


@app.on_event("startup")
def load_model():
  global model
  if os.path.exists(MODEL_OUTPUT_PATH):
    model = joblib.load(MODEL_OUTPUT_PATH)
    print("🤖 ML Model Loaded Successfully!")
  else:
    print(
        f"⚠️ Warning: Model not found at {MODEL_OUTPUT_PATH}. Make sure to"
        " train the model first."
    )


class TransactionData(BaseModel):
  features: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
  return templates.TemplateResponse(request, "index.html", {"prediction": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, features_input: str = Form(...)):
  global model

  if model is None:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "error": "ML model is not loaded yet. Please train the model first.",
            "prediction": None,
            "previous_input": features_input,
        },
    )

  try:
    # 🛠️ FIX: Newlines, spaces aur tabs ko clean karke comma se split karein
    cleaned_input = (
        features_input.replace("\n", ",")
        .replace("\r", ",")
        .replace("\t", ",")
    )

    feature_list = [
        float(x.strip()) for x in cleaned_input.split(",") if x.strip()
    ]

    if len(feature_list) != 30:
      return templates.TemplateResponse(
          request,
          "index.html",
          {
              "error": (
                  "Expected exactly 30 features (Time, V1-V28, Amount). Got"
                  f" {len(feature_list)}."
              ),
              "prediction": None,
              "previous_input": features_input,
          },
      )

    final_features = np.array(feature_list).reshape(1, -1)
    prediction_code = int(model.predict(final_features)[0])

    if prediction_code == 1:
      result = "🚨 FRAUDULENT TRANSACTION DETECTED!"
      alert_class = "danger"
    else:
      result = "✅ LEGITIMATE (NORMAL) TRANSACTION"
      alert_class = "success"

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "prediction": result,
            "alert_class": alert_class,
            "previous_input": features_input,
        },
    )

  except Exception as e:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "error": f"Invalid format or parsing error: {str(e)}",
            "prediction": None,
            "previous_input": features_input,
        },
    )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)