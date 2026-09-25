import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


# =========================
# SETTINGS
# =========================

DATA_DIR = "dataset"
MODEL_DIR = "models"

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================
# IMAGE TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# DATASET
# =========================

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


class_names = train_dataset.classes

print()
print("Classes:", class_names)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# PRETRAINED RESNET18
# =========================

print()
print("Loading pretrained ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze most pretrained layers
for param in model.parameters():
    param.requires_grad = False


# Replace final layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(class_names)
)

model = model.to(device)


# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TRAINING
# =========================

best_accuracy = 0.0


for epoch in range(EPOCHS):

    print()
    print("=" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 50)

    # ---------------------
    # TRAIN
    # ---------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    train_accuracy = 100 * correct / total

    train_loss = running_loss / len(train_loader)


    # ---------------------
    # VALIDATION
    # ---------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = 100 * val_correct / val_total

    val_loss = val_loss_total / len(val_loader)


    print()
    print(f"Train Loss:      {train_loss:.4f}")
    print(f"Train Accuracy:  {train_accuracy:.2f}%")
    print(f"Val Loss:        {val_loss:.4f}")
    print(f"Val Accuracy:    {val_accuracy:.2f}%")


    # ---------------------
    # SAVE BEST MODEL
    # ---------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        model_path = os.path.join(
            MODEL_DIR,
            "cropguard_resnet18.pth"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "accuracy": best_accuracy
            },
            model_path
        )

        print()
        print("✅ Best model saved!")
        print(f"Validation Accuracy: {best_accuracy:.2f}%")


# =========================
# COMPLETE
# =========================

print()
print("=" * 50)
print("🎉 TRAINING COMPLETED")
print("=" * 50)

print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

print()
print("Model saved at:")

print(
    os.path.join(
        MODEL_DIR,
        "cropguard_resnet18.pth"
    )
)