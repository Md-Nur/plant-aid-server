import numpy as np
from PIL import Image
import io
import onnxruntime as ort
from static_data.tomato_potato_pepper import crop_diseases

IMG_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3

plants = {
    "pepper": 0,
    "potato": 1,
    "tomato": 2,
}

plnats_bangla = {
    "pepper": "মরিচ",
    "potato": "আলু",
    "tomato": "টমেটো",
}

pata_class = ["others", "pepper", "potato", "tomato"]


def isPata(image_batch, pata_name):
    onnx_model_path = "models/is_pata/sequential_2.onnx"
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: image_batch})
    predicted_class = pata_class[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    # print(predicted_class, confidence)
    if predicted_class == "others":
        return ["এটা (আলু/টমেটো/মরিচ)পাতা নয়! অন্য কিছু হতে পারে।", confidence]
    elif predicted_class != pata_name:
        return [
            f"এটা {plnats_bangla[pata_name]} পাতা নয়! এটা {plnats_bangla[predicted_class]} পাতা হতে পারে।",
            confidence,
        ]
    else:
        return [False, False]


# Define the function
async def plant_predict(image_file, plant_name):

    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))  # Convert to RGB if needed

    image = image.resize((IMG_SIZE, IMG_SIZE))
    input_arr = np.array(image).astype(np.float32)

    if input_arr.ndim == 3 and input_arr.shape[2] == 4:  # If RGBA, remove alpha channel
        input_arr = input_arr[:, :, :3]  # Keep only RGB channels

    # Ensure the array has 3 channels (RGB)
    if input_arr.ndim == 2:  # If grayscale, convert to RGB
        input_arr = np.stack((input_arr,) * 3, axis=-1)

    input_arr = np.tile(input_arr, (BATCH_SIZE, 1, 1, 1))
    input_arr = input_arr.reshape([BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS])
    pata = isPata(input_arr, plant_name)
    if pata[0]:
        return {"warn": pata[0], "confidence": float(pata[1] * 100)}
    else:
        onnx_model_path = f"models/{plant_name}/sequential_2.onnx"
        session = ort.InferenceSession(onnx_model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_arr})
        tomato_disesases = crop_diseases[plants[plant_name]]["diseases"]
        predicted_class = tomato_disesases[np.argmax(result[0])]
        confidence = float(np.max(result[0]))
        return {"class": predicted_class, "confidence": float(confidence * 100)}
