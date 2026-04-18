import os
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import torch.nn.functional as F

DATA_PATH = "/home/hornet/dataset_folders/retinopathy_dataset2/archive/resized_train/resized_train"
CSV_PATH = "/home/hornet/dataset_folders/retinopathy_dataset2/archive/trainLabels.csv"

BATCH_SIZE = 8
EPOCHS = 15
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs_ensemble_meta_run2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["image"] = df["image"].apply(lambda x: os.path.join(DATA_PATH, x + ".jpeg"))
df.rename(columns={"level": "label"}, inplace=True)

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = RetinoDataset(train_df, transform)
val_dataset = RetinoDataset(val_df, transform)

labels = train_df["label"].values
class_sample_count = np.bincount(labels)

weights = 1.0 / class_sample_count
samples_weight = weights[labels]

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

def get_models():
    eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, NUM_CLASSES)

    res = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    res.fc = nn.Linear(res.fc.in_features, NUM_CLASSES)

    return eff.to(DEVICE), res.to(DEVICE)

efficient_model, resnet_model = get_models()

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer_eff = torch.optim.Adam(efficient_model.parameters(), lr=0.0003)
optimizer_res = torch.optim.Adam(resnet_model.parameters(), lr=0.0003)

def train(model, optimizer):
    model.train()
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_preds, all_labels

def evaluate_ensemble():
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            out_eff = F.softmax(efficient_model(images), dim=1)
            out_res = F.softmax(resnet_model(images), dim=1)
            outputs = 0.7 * out_eff + 0.3 * out_res
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_preds, all_labels

def get_meta_features():
    efficient_model.eval()
    resnet_model.eval()
    features, labels_list = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            out_eff = F.softmax(efficient_model(images), dim=1)
            out_res = F.softmax(resnet_model(images), dim=1)

            entropy_eff = -torch.sum(out_eff * torch.log(out_eff + 1e-8), dim=1, keepdim=True)
            entropy_res = -torch.sum(out_res * torch.log(out_res + 1e-8), dim=1, keepdim=True)

            max_eff = torch.max(out_eff, dim=1, keepdim=True)[0]
            max_res = torch.max(out_res, dim=1, keepdim=True)[0]

            combined = torch.cat([
                out_eff,
                out_res,
                entropy_eff,
                entropy_res,
                max_eff,
                max_res
            ], dim=1)

            features.append(combined.cpu())
            labels_list.append(labels)

    return torch.cat(features), torch.cat(labels_list)

class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_CLASSES * 2 + 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

def evaluate_stacked():
    efficient_model.eval()
    resnet_model.eval()
    meta_model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            out_eff = F.softmax(efficient_model(images), dim=1)
            out_res = F.softmax(resnet_model(images), dim=1)

            entropy_eff = -torch.sum(out_eff * torch.log(out_eff + 1e-8), dim=1, keepdim=True)
            entropy_res = -torch.sum(out_res * torch.log(out_res + 1e-8), dim=1, keepdim=True)

            max_eff = torch.max(out_eff, dim=1, keepdim=True)[0]
            max_res = torch.max(out_res, dim=1, keepdim=True)[0]

            combined = torch.cat([
                out_eff,
                out_res,
                entropy_eff,
                entropy_res,
                max_eff,
                max_res
            ], dim=1)

            final_out = meta_model(combined)
            preds = torch.argmax(final_out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels

history = {
    "resnet": {"acc": [], "kappa": [], "f1": []},
    "efficientnet": {"acc": [], "kappa": [], "f1": []},
    "ensemble": {"acc": [], "kappa": [], "f1": []}
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train(efficient_model, optimizer_eff)
    train(resnet_model, optimizer_res)

    res_preds, res_labels = evaluate_model(resnet_model)
    eff_preds, eff_labels = evaluate_model(efficient_model)
    ens_preds, ens_labels = evaluate_ensemble()

    def metrics(y_true, y_pred):
        return (
            accuracy_score(y_true, y_pred),
            cohen_kappa_score(y_true, y_pred),
            precision_score(y_true, y_pred, average='macro', zero_division=0),
            recall_score(y_true, y_pred, average='macro', zero_division=0),
            f1_score(y_true, y_pred, average='macro', zero_division=0),
        )

    res = metrics(res_labels, res_preds)
    eff = metrics(eff_labels, eff_preds)
    ens = metrics(ens_labels, ens_preds)

    history["resnet"]["acc"].append(res[0])
    history["resnet"]["kappa"].append(res[1])
    history["resnet"]["f1"].append(res[4])

    history["efficientnet"]["acc"].append(eff[0])
    history["efficientnet"]["kappa"].append(eff[1])
    history["efficientnet"]["f1"].append(eff[4])

    history["ensemble"]["acc"].append(ens[0])
    history["ensemble"]["kappa"].append(ens[1])
    history["ensemble"]["f1"].append(ens[4])

    print(f"EffNet → Acc:{eff[0]:.4f} F1:{eff[4]:.4f}")
    print(f"ResNet → Acc:{res[0]:.4f} F1:{res[4]:.4f}")
    print(f"Ensemble → Acc:{ens[0]:.4f} F1:{ens[4]:.4f}")

meta_X, meta_y = get_meta_features()
meta_model = MetaModel().to(DEVICE)

meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=3e-4)
meta_criterion = nn.CrossEntropyLoss()

for epoch in range(8):
    meta_model.train()
    inputs = meta_X.float().to(DEVICE)
    targets = meta_y.to(DEVICE)

    meta_optimizer.zero_grad()
    outputs = meta_model(inputs)
    loss = meta_criterion(outputs, targets)
    loss.backward()
    meta_optimizer.step()

    print(f"Meta Epoch {epoch+1} Loss: {loss.item():.4f}")

stack_preds, stack_labels = evaluate_stacked()

stack_acc = accuracy_score(stack_labels, stack_preds)
stack_kappa = cohen_kappa_score(stack_labels, stack_preds)
stack_f1 = f1_score(stack_labels, stack_preds, average='macro', zero_division=0)

print(f"STACKED → Acc:{stack_acc:.4f} Kappa:{stack_kappa:.4f} F1:{stack_f1:.4f}")

history["stacked"] = {
    "acc": [stack_acc]*EPOCHS,
    "kappa": [stack_kappa]*EPOCHS,
    "f1": [stack_f1]*EPOCHS
}

epochs = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs, history["resnet"]["acc"])
plt.plot(epochs, history["efficientnet"]["acc"])
plt.plot(epochs, history["ensemble"]["acc"])
plt.plot(epochs, history["stacked"]["acc"])
plt.legend(["ResNet", "EffNet", "Ensemble", "Stacked"])
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
plt.close()

plt.figure()
plt.plot(epochs, history["resnet"]["kappa"])
plt.plot(epochs, history["efficientnet"]["kappa"])
plt.plot(epochs, history["ensemble"]["kappa"])
plt.plot(epochs, history["stacked"]["kappa"])
plt.legend(["ResNet", "EffNet", "Ensemble", "Stacked"])
plt.savefig(os.path.join(OUTPUT_DIR, "kappa.png"))
plt.close()

plt.figure()
plt.plot(epochs, history["resnet"]["f1"])
plt.plot(epochs, history["efficientnet"]["f1"])
plt.plot(epochs, history["ensemble"]["f1"])
plt.plot(epochs, history["stacked"]["f1"])
plt.legend(["ResNet", "EffNet", "Ensemble", "Stacked"])
plt.savefig(os.path.join(OUTPUT_DIR, "f1.png"))
plt.close()

cm = confusion_matrix(stack_labels, stack_preds)

plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

recall_vals = recall_score(stack_labels, stack_preds, average=None, zero_division=0)

plt.figure()
plt.plot(range(len(recall_vals)), recall_vals)
plt.savefig(os.path.join(OUTPUT_DIR, "class_recall.png"))
plt.close()

print("\nDONE")