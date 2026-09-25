import os
import random
import shutil

train = "dataset/train"
val = "dataset/val"

classes = [
    folder for folder in os.listdir(train)
    if folder.startswith("Rice")
]

print("Rice folders found:", classes)

for class_name in classes:
    train_folder = os.path.join(train, class_name)
    val_folder = os.path.join(val, class_name)

    os.makedirs(val_folder, exist_ok=True)

    images = [
        f for f in os.listdir(train_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    number_to_move = int(len(images) * 0.20)

    for image in images[:number_to_move]:
        shutil.move(
            os.path.join(train_folder, image),
            os.path.join(val_folder, image)
        )

    print(class_name, ":", number_to_move, "images moved")

print("Rice validation split completed!")