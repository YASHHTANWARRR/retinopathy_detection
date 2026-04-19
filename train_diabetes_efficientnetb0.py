import os
import torch
import cudf
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import WeightedRandomSampler
from torchvision.models import efficientnet_b0
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)


DATA_PATH = "/home/hornet/dataset_folders/retinopathy_dataset2/archive/resized_train/resized_train"
BATCH_SIZE = 8
EPOCHS = 15
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = "/home/hornet/dataset_folders/retinopathy_dataset2/archive/trainLabels.csv"

df = pd.read_csv(csv_path)

df["image"] = df["image"].apply(
    lambda x: os.path.join(DATA_PATH, x + ".jpeg")
)

df.rename(columns={"level": "label"}, inplace=True)

print(df.head())
print("Total samples:", len(df))

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"])

class RetinoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image"]
        label = self.df.loc[idx, "label"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = RetinoDataset(train_df, transform=train_transform)
val_dataset = RetinoDataset(val_df, transform=val_transform)

labels = train_df["label"].values
class_sample_count = np.bincount(labels)

#added weight sampler to handle class imbalance
weights = 1. / class_sample_count
samples_weight = weights[labels]

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#model = models.resnet18(pretrained=True) #if using resnet18

model = efficientnet_b0(pretrained=True)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)#for efficientnet_b4

#model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)#for resnet
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)#for efficientnet
model = model.to(DEVICE)

class_counts = df["label"].value_counts().sort_index()
#added weight class
#weights = 1.0 / class_counts
#weights = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

def train():
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

#updated validate function 
def validate():
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    kappa = cohen_kappa_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    support = cm.sum(axis=1)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("\n📊 Additional Metrics:")
    print("Accuracy:", accuracy)
    print("Cohen’s Kappa:", kappa)
    print("Precision per class:", precision)
    print("Recall per class:", recall)
    print("Support per class:", support)

    os.makedirs("outputs", exist_ok=True)

    classes = np.arange(len(precision))

    # 1. Precision vs Recall
    plt.figure()
    plt.plot(classes, precision)
    plt.plot(classes, recall)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Precision vs Recall per Class")
    plt.legend(["Precision", "Recall"])
    plt.savefig("outputs/precision_vs_recall.png")
    plt.close()

    # 2. Support
    plt.figure()
    plt.plot(classes, support)
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.title("Support per Class")
    plt.savefig("outputs/support.png")
    plt.close()

    # 3. Confusion Matrix Heatmap
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    return total_loss / len(val_loader), accuracy * 100

for epoch in range(EPOCHS):
    train_loss = train()
    val_loss, val_acc = validate()

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2f}%")

torch.save(model.state_dict(), "retinopathy_efficientnet.pth")

print("\nModel saved!")