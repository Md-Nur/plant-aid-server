from fastapi import FastAPI, File, UploadFile
from apis.all_plants import all_plants_predict  # Import the function

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the API!"}


# Declare the function with the @app.post decorator
@app.post("/all_plants")
async def predict(file: UploadFile = File(...)):
    prediction = await all_plants_predict(file)  # Call the function
    return {
        "prediction": prediction[0],
        "confidence": prediction[1] * 100,
    }  # Call the function
