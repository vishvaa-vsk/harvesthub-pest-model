import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime

# =============================
# ⚙️ TPU Setup
# =============================
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("✅ TPU detected")
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("⚠️ TPU not found, using CPU/GPU strategy")
    strategy = tf.distribute.get_strategy()

print("🔁 Replicas in sync:", strategy.num_replicas_in_sync)

# =============================
# 📁 Paths
# =============================
base_dir = "PestDetectionProject"
train_dir = os.path.join(base_dir, "data/processed/train")
val_dir = os.path.join(base_dir, "data/processed/val")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

# =============================
# ⚙️ Hyperparams
# =============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
EPOCHS = 20
LEARNING_RATE = 0.0001
AUTOTUNE = tf.data.AUTOTUNE

# =============================
# 📊 TF Data Pipelines
# =============================
def prepare_dataset(dir_path, shuffle=True):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
    return ds

train_data = prepare_dataset(train_dir)
val_data = prepare_dataset(val_dir, shuffle=False)

# =============================
# 🔍 Class Info
# =============================
class_names = train_data.class_names
num_classes = len(class_names)
print(f"\n✅ Found {num_classes} classes: {class_names}")

# =============================
# 🧠 Build Model (in scope)
# =============================
with strategy.scope():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# =============================
# ⏱️ Callbacks
# =============================
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = os.path.join(model_dir, f"mobilenetv2_tpu_tfdata_{timestamp}.h5")

callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(patience=5, restore_best_weights=True)
]

# =============================
# 🚀 Train Model
# =============================
print("\n🚀 Starting training with tf.data + TPU...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n✅ Training complete. Best model saved to: {checkpoint_path}")
