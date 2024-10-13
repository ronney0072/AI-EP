from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from io import StringIO
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()
template = Jinja2Templates(directory="static/templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get('/', response_class=HTMLResponse)
def index(req: Request):
    return template.TemplateResponse(
        name="index.html",
        context={"request": req }
    )

# Path to the model
model_path = Path("model/xgb_model_grid .joblib")
model = joblib.load(model_path)

@app.post("/features/predict")
async def create_items(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Define a dictionary that maps the old column names to new column names
    new_column_names = {
        'tce_period': 'koi_period',
        'tce_time0': 'koi_time0bk',
        'tce_impact': 'koi_impact',
        'tce_duration': 'koi_duration',
        'tce_depth': 'koi_depth',
        'tce_eqt': 'koi_teq',
        'tce_steff': 'koi_steff',
        'tce_slogg': 'koi_slogg',
        'tce_sradius': 'koi_srad'
    }

    # Use the rename() method to rename the columns in the DataFrame
    df.rename(columns=new_column_names, inplace=True)

    # Check for required headers
    required_columns = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
                        'koi_depth', 'koi_teq', 'koi_steff', 'koi_slogg', 'koi_srad']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")

    # Select features for preprocessing
    features = df[required_columns]

    # Make predictions using the loaded model
    percentage = model.predict_proba(features)[:, 1]
    custom_threshold = 0.5
    predictions = (percentage >= custom_threshold).astype(int)

    # Add predictions to the DataFrame
    df['predictions'] = np.where(predictions == 1, 'confirmed', 'falsepositive')
    df['percentage'] = percentage * 100

    # Create CSV response with predictions
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return Response(content=output.getvalue(),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"})
