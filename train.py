import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime

# Paths
base_dir = "PestDetectionProject"
train_dir = os.path.join(base_dir, "data/processed/train")
val_dir = os.path.join(base_dir, "data/processed/val")
model_dir = os.path.join(base_dir, "models")

# Ensure model output directory exists
os.makedirs(model_dir, exist_ok=True)

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load data
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

# Detect number of classes dynamically
num_classes = len(train_data.class_indices)
print(f"\n✅ Detected {num_classes} classes in training set.")

# Check for class mismatch
train_classes = set(train_data.class_indices.keys())
val_classes = set(val_data.class_indices.keys())
missing_in_val = train_classes - val_classes
if missing_in_val:
    print(f"⚠️ Missing classes in validation set: {missing_in_val}\n")

# Build model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')  # <-- Match class count
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = os.path.join(model_dir, f"mobilenetv2_{timestamp}.h5")

callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(patience=5, restore_best_weights=True)
]

class_labels = list(train_data.class_indices.keys())
print(class_labels)


# Train model
print("\n🚀 Starting training...\n")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=callbacks
)

print(f"\n✅ Training complete. Best model saved to: {checkpoint_path}")