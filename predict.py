import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("plant_disease_model.keras")

# Load class names
with open("label_classes.pkl", "rb") as f:
    class_names = pickle.load(f)

# Path to test image
image_path = "test_leaf.jpg"

# Read and preprocess image
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# Convert BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to match training size
image = cv2.resize(image, (128,128), interpolation=cv2.INTER_AREA)

# Normalize
image = image / 255.0

# Reshape for model
image = np.reshape(image, (1, 128, 128, 3))

# Predict
prediction = model.predict(image)

# Get predicted class and confidence
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Predicted Disease:", predicted_class)
print("Confidence:", round(confidence, 2), "%")