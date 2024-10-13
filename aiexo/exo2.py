from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from io import StringIO
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def get_homepage():
    with open("myenv/static/frontend3.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

# Allow CORS for all origins (not recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can also specify specific origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the model
model_path = Path("myenv/model/pipeline_with_catboost.joblib")
model = joblib.load(model_path)


@app.post("/features/predict")
async def create_items(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Check for required headers
    required_columns = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
                        'koi_depth', 'koi_teq', 'koi_steff', 'koi_slogg', 'koi_srad']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")

    # Assuming the CSV has the correct feature columns
    features = df[required_columns].values

    # Make predictions using the loaded model
    predictions = model.predict(features)

    # Add predictions to the DataFrame
    df['predictions'] = np.where(predictions == 1, 'confirmed', 'falsepositive')
    df['percentage'] = ''  # Add an empty column named 'percentage'

    # Create CSV response with predictions
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return Response(content=output.getvalue(),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"})
