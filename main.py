from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from apis.all_plants import all_plants_predict
from apis.tomato_potato_pepper import plant_predict
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

origins = [
    os.getenv("FRONTEND_URL"),
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {
        "message": f"Welcome to the Plant Disease Detection API! {os.getenv('FRONTEND_URL')}"
    }


# Declare the function with the @app.post decorator
@app.post("/plant/other")
async def all_plant_predict(file: UploadFile = File(...)):
    prediction = await all_plants_predict(file)  # Call the function
    return prediction  # Return the prediction


@app.post("/plant/{plant_name}")
async def plant_predict(plant_name: str, file: UploadFile = File(...)):
    prediction = await plant_predict(file, plant_name)
    return prediction
