import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================
# Load trained model
# =========================

MODEL_PATH = "plant_disease_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# Class names
# Same order used during training
# =========================

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


# =========================
# Prediction function
# =========================

def predict_disease(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((128, 128))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize pixels
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get top 3 predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]

    results = {}

    for index in top_3_indices:
        results[class_names[index]] = float(predictions[index])

    return results


# =========================
# Gradio Interface
# =========================

demo = gr.Interface(
    fn=predict_disease,

    inputs=gr.Image(
        type="pil",
        label="Upload a Plant Leaf Image"
    ),

    outputs=gr.Label(
        num_top_classes=3,
        label="Disease Prediction"
    ),

    title="🌿 Plant Disease Detection",

    description=(
        "Upload a plant leaf image to predict the disease "
        "using a CNN trained on the PlantVillage dataset."
    )
)


# =========================
# Run app
# =========================

if __name__ == "__main__":
    demo.launch()