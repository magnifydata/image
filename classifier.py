# classifier.py
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image # Use Pillow for loading image from bytes

# --- Model Loading (Can be loaded once) ---
# Wrap model loading in a function, potentially with caching in the Streamlit app
def load_model():
    """Loads the pre-trained MobileNetV2 model."""
    print("Loading pre-trained model (MobileNetV2)...")
    model = MobileNetV2(weights='imagenet')
    print("Model loaded.")
    return model

# --- Target Classes and Mapping (Keep these consistent) ---
target_classes = ["bicycle", "motor scooter", "moped",
                  "sports car", "convertible", "minivan", "pickup", "police van", "garbage truck", "ambulance", "fire engine", "cab", "limousine", "passenger car",
                  "trolleybus", "school bus", "minibus"]

category_map = {
    "bicycle": "Bicycle", "motor scooter": "Bicycle", "moped": "Bicycle",
    "sports car": "Car", "convertible": "Car", "minivan": "Car", "pickup": "Car",
    "police van": "Car", "ambulance": "Car", "fire engine": "Car", "cab": "Car",
    "limousine": "Car", "passenger car": "Car",
    "garbage truck": "Bus/Truck", "trolleybus": "Bus/Truck",
    "school bus": "Bus/Truck", "minibus": "Bus/Truck",
}

# --- Core Classification Function ---
def classify_image(model, image_data):
    """
    Loads an image from data (like uploaded file), preprocesses it,
    and returns the classification result.

    Args:
        model: The loaded Keras model.
        image_data: Image data (e.g., from Streamlit's file_uploader).

    Returns:
        A tuple: (final_category, highest_score, top_5_predictions_formatted)
                 or None if an error occurs.
    """
    try:
        # Load the image file using Pillow (handles bytes from upload)
        # Resize it to 224x224 pixels (required by MobileNetV2)
        img = Image.open(image_data).resize((224, 224))

        # Ensure image is RGB (MobileNetV2 expects 3 channels)
        if img.mode != 'RGB':
             img = img.convert('RGB')

        # Convert the image to a NumPy array
        img_array = image.img_to_array(img)

        # Expand dimensions to match the model's expected input shape
        img_array_expanded = np.expand_dims(img_array, axis=0)

        # Preprocess the image for MobileNetV2
        processed_img = preprocess_input(img_array_expanded)
        print("Image loaded and preprocessed.")

        # Make Prediction
        print("Classifying image...")
        predictions = model.predict(processed_img)

        # Decode the predictions
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        # Format top 5 predictions for display
        top_5_formatted = [f"{label} ({score:.2f})" for _, label, score in decoded_predictions]
        print(f"Raw predictions: {top_5_formatted}")


        # Filter for Target Classes and Determine Category
        found_category = "Other"
        highest_score = 0.0

        for imagenet_id, label, score in decoded_predictions:
            if label.lower() in target_classes:
                category = category_map.get(label.lower(), "Other")
                print(f"Found relevant class: '{label}' (Score: {score:.2f}), mapped to Category: '{category}'")
                if score > highest_score and category != "Other":
                    highest_score = score
                    found_category = category
                # Optional: break here if you only want the top hit
                # break

        print(f"Final category determined: {found_category} with score {highest_score:.2f}")
        return found_category, highest_score, top_5_formatted

    except Exception as e:
        print(f"Error during classification: {e}")
        return None, 0.0, [] # Return default values on error
