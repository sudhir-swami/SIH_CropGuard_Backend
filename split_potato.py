import os
import random
import shutil

classes = [
    "Potato_Healthy",
    "Potato_Early_Blight",
    "Potato_Late_Blight"
]

base = "dataset"
train = os.path.join(base, "train")
val = os.path.join(base, "val")

for class_name in classes:
    train_folder = os.path.join(train, class_name)
    val_folder = os.path.join(val, class_name)

    os.makedirs(val_folder, exist_ok=True)

    images = [
        f for f in os.listdir(train_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    # Move 20% of images from train to val
    number_to_move = max(1, int(len(images) * 0.20))

    for image in images[:number_to_move]:
        source = os.path.join(train_folder, image)
        destination = os.path.join(val_folder, image)
        shutil.move(source, destination)

    print(class_name, ":", number_to_move, "images moved to validation")

print("Done!")