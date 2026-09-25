import os
import shutil
import random

SOURCE = "Tomato-leaf-disease-dataset"
DEST = "dataset"

CLASSES = {
    "Early_blight": "Early_Blight",
    "Late_blight": "Late_Blight",
    "healthy": "Healthy",
}

TRAIN_RATIO = 0.8

random.seed(42)


for source_class, target_class in CLASSES.items():

    source_folder = os.path.join(SOURCE, source_class)

    train_folder = os.path.join(
        DEST, "train", target_class
    )

    val_folder = os.path.join(
        DEST, "val", target_class
    )

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    if not os.path.exists(source_folder):
        print(f"ERROR: Folder not found: {source_folder}")
        continue

    images = []

    for file in os.listdir(source_folder):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            images.append(file)

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    print()
    print(f"{source_class}")
    print(f"Total: {len(images)}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")

    for file in train_images:

        shutil.copy2(
            os.path.join(source_folder, file),
            os.path.join(train_folder, file)
        )

    for file in val_images:

        shutil.copy2(
            os.path.join(source_folder, file),
            os.path.join(val_folder, file)
        )


print()
print("================================")
print("Dataset preparation completed!")
print("================================")