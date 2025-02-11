import numpy as np
from PIL import Image
import io
import onnxruntime as ort
from static_data.all_plants import all_plants


def isPata(image):
    IMG_SIZE = 256
    BATCH_SIZE = 32
    CHANNELS = 3

    onnx_model_path = "models/is_pata/sequential_2.onnx"
    session = ort.InferenceSession(onnx_model_path)

    image = image.resize((IMG_SIZE, IMG_SIZE))
    input_arr = np.array([image]).astype(np.float32)
    input_arr = np.tile(input_arr, (BATCH_SIZE, 1, 1, 1))
    input_arr = input_arr.reshape([BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    predictions = session.run([output_name], {input_name: input_arr})
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    if predicted_index == 0:
        return {
            "warn": "এটা কোনো পাতা নয়! অন্য কিছু হতে পারে।",
            "confidence": float(confidence) * 100,
        }
    else:
        return False


# Define the function
async def all_plants_predict(image_file):

    # Load the pickle model
    onnx_model_path = "models/all_plants/sequential.onnx"
    session = ort.InferenceSession(onnx_model_path)
    # Read the image file
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    pata = isPata(image)
    if pata:
        return pata
    image = image.resize((128, 128))
    input_arr = np.array([image]).astype(np.float32)

    input_arr = input_arr.reshape([1, 128, 128, 3])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_arr})
    plant_class = all_plants[np.argmax(result[0])]  # Get the class
    confidence = float(np.max(result[0]))  # Get the confidence

    return {"class": plant_class, "confidence": confidence * 100}
