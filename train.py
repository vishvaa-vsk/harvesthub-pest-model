# train.py

import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ========= CONFIG =========
IMG_SIZE = (224, 224)
EPOCHS = 20
BASE_LR = 1e-4

# Adjust based on GPU capacity
BATCH_SIZE = 32

# ========= PATHS =========
base_dir = "PestDetectionProject"
train_dir = os.path.join(base_dir, "data/processed/train")
val_dir = os.path.join(base_dir, "data/processed/val")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

# ========= STRATEGY SETUP =========
strategy = tf.distribute.get_strategy()  # Uses GPU if available
print(f"✅ Running on strategy: {type(strategy).__name__} with {strategy.num_replicas_in_sync} replica(s)")

# ========= DATA AUGMENTATION =========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
class_labels = list(train_data.class_indices.keys())
print(f"\n📊 Detected {num_classes} classes: {class_labels}")

# ========= MODEL DEFINITION =========
with strategy.scope():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=BASE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# ========= CALLBACKS =========
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = os.path.join(model_dir, f"mobilenetv2_{timestamp}.h5")

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max"),
    EarlyStopping(patience=5, restore_best_weights=True)
]

# ========= TRAINING =========
print("\n🚀 Starting training...\n")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=callbacks
)

print(f"\n✅ Training complete. Best model saved to: {checkpoint_path}")