import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('PestDetectionProject/models/mobilenetv2_20250605-091832.h5')

# Class labels from training
class_labels = ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Bell_pepper leaf', 'Bell_pepper leaf spot', 'Blueberry leaf', 'Blueberry___healthy', 'Cherry leaf', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach leaf', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato leaf early blight', 'Potato leaf late blight', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry leaf', 'Raspberry___healthy', 'Soyabean leaf', 'Soybean___healthy', 'Squash Powdery mildew leaf', 'Squash___Powdery_mildew', 'Strawberry leaf', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'grape leaf', 'grape leaf black rot']  # Paste full list here

# Load and preprocess image
img_path = 'images.jpeg'  # Update this
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_class = class_labels[predicted_index]

print(f"🔬 Predicted Class: {predicted_class}")