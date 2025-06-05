import os

base_dir = "PestDetectionProject"

folders = [
    "data/raw/PlantDoc",
    "data/raw/PlantVillage",
    "data/processed/images",
    "data/processed/labels",
    "models",
    "notebooks",
    "scripts",
    "tflite_model",
    "reference/pesticide_mapping"
]

for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)
