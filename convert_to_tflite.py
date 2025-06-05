# convert.py
import tensorflow as tf

model = tf.keras.models.load_model("PestDetectionProject/models/mobilenetv2_20250605-091832.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("PestDetectionProject/tflite_model/pest_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite!")