import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ================== CONFIGURATION ==================
MODEL_PATH = "<PATH/TO/>/steganography_detector.h5"

# ================== MODEL LOADING ==================
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ================== IMAGE PREDICTION ==================
def predict_image(image_path):
    """Predicts if an image contains steganography or is clean."""
    if not os.path.exists(image_path):
        return "❌ Error: Image file not found!"

    try:
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]  
        print(f"🔍 Raw Model Output: {prediction}")

        return "🛑 Steganography Detected" if prediction >= 0.5 else "✅ Clean Image"
    
    except Exception as e:
        return f"❌ Error processing image: {e}"

# ================== USER INPUT ==================
image_path = input("Enter image path: ").strip()
result = predict_image(image_path)
print(result)
