from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import joblib
from pathlib import Path

app = FastAPI()


results: List[Dict] = []

model_path = Path("myenv/model/pipeline_with_catboost.joblib")
model = joblib.load(model_path)


class Features(BaseModel):
    koi_period: float
    koi_time0bk: float
    koi_impact: float
    koi_duration: float
    koi_depth: float
    koi_teq: float
    koi_steff: float
    koi_slogg: float
    koi_srad: float


@app.get("/features", response_model=List[Dict])
async def get_items():
    return  results



@app.post("/features/create", response_model=Dict)
async def create_items(items: List[Features]):
    global results

    new_results = []

    features = np.array([[item.koi_period, item.koi_time0bk, item.koi_impact,
                          item.koi_duration, item.koi_depth, item.koi_teq,
                          item.koi_steff, item.koi_slogg, item.koi_srad] for item in items])


    predictions = model.predict(features)

    for item, prediction in zip(items, predictions):
        item_dict = item.dict()
        status = "confirmed" if prediction == 1 else "falsepositive"
        item_dict["status"] = status

        new_results.append(item_dict)

    results.extend(new_results)

    return {"message": "Items added successfully ", "items": new_results}
