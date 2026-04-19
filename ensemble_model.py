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
from sklearn.utils import resample

# loading second dataset
def load_second_dataset(base_path):
    mapping = {
        "No_DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferate_DR": 4
    }

    data = []

    for cls in mapping:
        folder = os.path.join(base_path, cls)
        for img in os.listdir(folder):
            data.append({
                "image": os.path.join(folder, img),
                "label": mapping[cls]
            })

    return pd.DataFrame(data)

DATA_PATH = "/home/hornet/dataset_folders/retinopathy_dataset2/archive/resized_train/resized_train"
CSV_PATH = "/home/hornet/dataset_folders/retinopathy_dataset2/archive/trainLabels.csv"
SECOND_DATA_PATH = "/home/hornet/dataset_folders/retinopathy_dataset/archive/gaussian_filtered_images"

BATCH_SIZE = 8
EPOCHS = 15
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs_ensemble_META_RUN_2data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["image"] = df["image"].apply(lambda x: os.path.join(DATA_PATH, x + ".jpeg"))
df.rename(columns={"level": "label"}, inplace=True)

#loading new dataset
df2 = load_second_dataset(SECOND_DATA_PATH)

train_df1, val_df = train_test_split(df, test_size=0.2, stratify=df["label"])

train_df = pd.concat([train_df1, df2], ignore_index=True)

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
weights = 1.0 / (class_sample_count + 1e-6)
samples_weight = weights[labels]

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def get_models():
    eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, NUM_CLASSES)

    res = resnet50(weights=ResNet50_Weights.DEFAULT)
    res.fc = nn.Linear(res.fc.in_features, NUM_CLASSES)

    return eff.to(DEVICE), res.to(DEVICE)

efficient_model, resnet_model = get_models()

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer_eff = torch.optim.Adam(efficient_model.parameters(), lr=3e-4)
optimizer_res = torch.optim.Adam(resnet_model.parameters(), lr=3e-4)

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
    preds, labs = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labs.extend(labels.numpy())
    return preds, labs

def evaluate_ensemble():
    preds, labs = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            e1 = F.softmax(efficient_model(images), dim=1)
            e2 = F.softmax(resnet_model(images), dim=1)
            out = 0.7 * e1 + 0.3 * e2
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labs.extend(labels.numpy())
    return preds, labs

history = {
    "resnet": {"f1": []},
    "efficientnet": {"f1": []},
    "ensemble": {"f1": []}
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}")

    train(efficient_model, optimizer_eff)
    train(resnet_model, optimizer_res)

    r_pred, r_lab = evaluate_model(resnet_model)
    e_pred, e_lab = evaluate_model(efficient_model)
    en_pred, en_lab = evaluate_ensemble()

    r_f1 = f1_score(r_lab, r_pred, average='macro', zero_division=0)
    e_f1 = f1_score(e_lab, e_pred, average='macro', zero_division=0)
    en_f1 = f1_score(en_lab, en_pred, average='macro', zero_division=0)

    history["resnet"]["f1"].append(r_f1)
    history["efficientnet"]["f1"].append(e_f1)
    history["ensemble"]["f1"].append(en_f1)

    print(f"EffNet F1: {e_f1:.4f}")
    print(f"ResNet F1: {r_f1:.4f}")
    print(f"Ensemble F1: {en_f1:.4f}")

# ===== META FIX (ONLY FIXES ADDED) =====
efficient_model.eval()
resnet_model.eval()

for p in efficient_model.parameters():
    p.requires_grad = False
for p in resnet_model.parameters():
    p.requires_grad = False

def get_meta():
    feats, labs = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            e1 = F.softmax(efficient_model(images), dim=1)
            e2 = F.softmax(resnet_model(images), dim=1)

            ent1 = -torch.sum(e1 * torch.log(e1 + 1e-8), dim=1, keepdim=True)
            ent2 = -torch.sum(e2 * torch.log(e2 + 1e-8), dim=1, keepdim=True)

            m1 = torch.max(e1, dim=1, keepdim=True)[0]
            m2 = torch.max(e2, dim=1, keepdim=True)[0]

            ens = 0.7 * e1 + 0.3 * e2

            feats.append(torch.cat([
                e1,
                e2,
                ens,     # 🔥 ADD THIS LINE
                ent1,
                ent2,
                m1,
                m2
            ], dim=1).cpu())
            labs.append(labels)

    return torch.cat(feats), torch.cat(labs)

meta_X, meta_y = get_meta()

meta_X = (meta_X - meta_X.mean(0)) / (meta_X.std(0) + 1e-6)
meta_X = torch.clamp(meta_X, -3, 3)

perm = torch.randperm(meta_X.size(0))
meta_X = meta_X[perm]
meta_y = meta_y[perm]

#added resampling to balance meta dataset
X_np = meta_X.numpy()
y_np = meta_y.numpy()

X_bal, y_bal = [], []

for c in np.unique(y_np):
    X_c = X_np[y_np == c]
    y_c = y_np[y_np == c]

    X_res, y_res = resample(X_c, y_c, replace=True, n_samples=5000, random_state=42)

    X_bal.append(X_res)
    y_bal.append(y_res)


#meta_X = torch.tensor(np.vstack(X_bal))
#meta_y = torch.tensor(np.hstack(y_bal))
#TO avoid dtyoe surprise

meta_X = torch.tensor(np.vstack(X_bal), dtype=torch.float32)
meta_y = torch.tensor(np.hstack(y_bal), dtype=torch.long)

#till here
#class Meta(nn.Module):
#    def __init__(self):
#       super().__init__()
#       self.net = nn.Sequential(
#           nn.Linear(NUM_CLASSES*2+4, 128),
#          nn.ReLU(),
    #        nn.Linear(128, NUM_CLASSES)
#  )
# def forward(self, x):
#     return self.net(x)

#NEW META MODEL
class Meta(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_CLASSES*3+4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

meta_model = Meta().to(DEVICE)
opt = torch.optim.Adam(meta_model.parameters(), lr=8e-5)

#
class_counts = np.bincount(meta_y.numpy())
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum()

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

meta_X = meta_X.float().to(DEVICE)
meta_y = meta_y.to(DEVICE)

for i in range(12):
    meta_model.train()
    out = meta_model(meta_X)
    loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)(out, meta_y)    
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("Meta loss:", loss.item())

def stacked_eval():
    preds, labs = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            e1 = F.softmax(efficient_model(images), dim=1)
            e2 = F.softmax(resnet_model(images), dim=1)

            ent1 = -torch.sum(e1 * torch.log(e1 + 1e-8), dim=1, keepdim=True)
            ent2 = -torch.sum(e2 * torch.log(e2 + 1e-8), dim=1, keepdim=True)

            m1 = torch.max(e1, dim=1, keepdim=True)[0]
            m2 = torch.max(e2, dim=1, keepdim=True)[0]

            ens = 0.7 * e1 + 0.3 * e2
            x = torch.cat([
                e1,
                e2,
                ens,     # 🔥 ADD THIS
                ent1,
                ent2,
                m1,
                m2
            ], dim=1)
            out = meta_model(x) / 1.2
            preds.extend(torch.argmax(out,1).cpu().numpy())
            labs.extend(labels.numpy())

    return preds, labs

sp, sl = stacked_eval()
sf1 = f1_score(sl, sp, average='macro', zero_division=0)

print("STACKED F1:", sf1)

# ===== PLOTS =====
epochs = range(1, EPOCHS+1)

plt.figure()
plt.plot(epochs, history["resnet"]["f1"])
plt.plot(epochs, history["efficientnet"]["f1"])
plt.plot(epochs, history["ensemble"]["f1"])
plt.legend(["ResNet","EffNet","Ensemble"])
plt.savefig(os.path.join(OUTPUT_DIR,"f1.png"))
plt.close()

cm = confusion_matrix(sl, sp)
plt.imshow(cm)
plt.colorbar()
plt.savefig(os.path.join(OUTPUT_DIR,"cm.png"))
plt.close()

print("DONE")