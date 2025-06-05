import os
import shutil
import random
from tqdm import tqdm

random.seed(42)

BASE_DIR = "PestDetectionProject"
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
PROCESSED_TRAIN_DIR = os.path.join(BASE_DIR, "data/processed/train")
PROCESSED_VAL_DIR = os.path.join(BASE_DIR, "data/processed/val")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_images(src_dir, dest_dir):
    for pest in os.listdir(src_dir):
        pest_src_path = os.path.join(src_dir, pest)
        if os.path.isdir(pest_src_path):
            pest_dest_path = os.path.join(dest_dir, pest)
            ensure_dir(pest_dest_path)

            for img in os.listdir(pest_src_path):
                src_img = os.path.join(pest_src_path, img)
                dest_img = os.path.join(pest_dest_path, img)
                shutil.copy2(src_img, dest_img)

def split_and_copy_plantvillage(source_dir, train_dir, val_dir, split_ratio=0.8):
    for pest in os.listdir(source_dir):
        pest_path = os.path.join(source_dir, pest)
        if not os.path.isdir(pest_path):
            continue

        images = [img for img in os.listdir(pest_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create pest folders
        train_pest_dir = os.path.join(train_dir, pest)
        val_pest_dir = os.path.join(val_dir, pest)
        ensure_dir(train_pest_dir)
        ensure_dir(val_pest_dir)

        # Copy images
        for img in train_images:
            shutil.copy2(os.path.join(pest_path, img), os.path.join(train_pest_dir, img))
        for img in val_images:
            shutil.copy2(os.path.join(pest_path, img), os.path.join(val_pest_dir, img))

def main():
    print("Preparing dataset...")

    # Step 1: Clear previous processed data
    shutil.rmtree(PROCESSED_TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(PROCESSED_VAL_DIR, ignore_errors=True)
    ensure_dir(PROCESSED_TRAIN_DIR)
    ensure_dir(PROCESSED_VAL_DIR)

    # Step 2: Copy PlantDoc
    print("Copying PlantDoc data...")
    copy_images(os.path.join(RAW_DIR, "PlantDoc/train"), PROCESSED_TRAIN_DIR)
    copy_images(os.path.join(RAW_DIR, "PlantDoc/test"), PROCESSED_VAL_DIR)

    # Step 3: Process PlantVillage
    print("Splitting and copying PlantVillage data...")
    split_and_copy_plantvillage(
        os.path.join(RAW_DIR, "PlantVillage"),
        PROCESSED_TRAIN_DIR,
        PROCESSED_VAL_DIR,
    )

    print("✅ Dataset preparation complete!")

if __name__ == "__main__":
    main()