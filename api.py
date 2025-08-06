#Interact with the model using the Vetiver API

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from pins import board_s3
from vetiver import VetiverModel

app = FastAPI()

board = board_s3("devakshah-stat468-models", allow_pickle_read=True)
v = VetiverModel.from_pin(board, "my_logit_model")
model = v.model

@app.get("/model_summary")
def model_summary():
    # Return model summary as string (or however you want to expose it)
    summary_str = str(model.summary())
    return {"model_summary": summary_str}

@app.post("/predict")
def predict(req):
    df = pd.DataFrame(req.data)
    df["prediction"] = model.predict(df)
    return df.to_dict(orient="records")