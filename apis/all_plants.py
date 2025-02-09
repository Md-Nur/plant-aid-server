import numpy as np
from PIL import Image
import io
import onnxruntime as ort

all_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Load the pickle model
onnx_model_path = "models/all_plants/sequential.onnx"
session = ort.InferenceSession(onnx_model_path)


# Define the function
async def all_plants_predict(image_file):
    """
    Make a prediction using the loaded model.

    Args:
        image_file: Uploaded image file.

    Returns:
        Prediction result.
    """
    # Read the image file
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")  # Convert to RGB if needed

    # Resize the image to the required input size (e.g., 128x128)
    image = image.resize((128, 128))

    # Convert the image to a NumPy array
    input_arr = np.array([image]).astype(np.float32)

    # Normalize the image (if required by the model)
    input_arr = input_arr / 255.0  # Example normalization to [0, 1]

    # Flatten the image or reshape it as required by the model

    # Make sure the input is properly shaped
    input_arr = input_arr.reshape([1, 128, 128, 3])
    # Make a prediction using the loaded model
    # Get model input name
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    result = session.run([output_name], {input_name: input_arr})

    return [all_classes[np.argmax(result[0])], float(np.max(result[0]))]
