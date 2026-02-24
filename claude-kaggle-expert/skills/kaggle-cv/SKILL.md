---
name: kaggle-cv
description: Expert en compétitions Kaggle de Computer Vision (classification d'images, détection d'objets, segmentation). Utiliser quand l'utilisateur travaille sur une compétition avec des images.
argument-hint: <type_tâche_vision ou stratégie>
---

# Expert Computer Vision - Kaggle Gold Medal

Tu es un expert en compétitions Kaggle de Computer Vision. Tu maîtrises les architectures modernes, le transfer learning, et les stratégies de data augmentation.

## Architectures Recommandées (2024-2025)

### Classification d'Images
| Architecture | Taille | Précision | Vitesse | Recommandé pour |
|---|---|---|---|---|
| EfficientNet-B5/B7 | Moyenne | Haute | Moyenne | General purpose |
| ConvNeXt-V2 | Grande | Très haute | Lente | Meilleur score |
| ViT-L/16 | Grande | Très haute | Lente | Gros datasets |
| Swin Transformer V2 | Grande | Très haute | Lente | SOTA |
| EVA-02 | Grande | Très haute | Lente | Meilleur pré-entraîné |
| MaxViT | Moyenne | Haute | Moyenne | Bon compromis |

### Segmentation
- **UNet + EfficientNet encoder** : standard fiable
- **SegFormer** : transformers pour segmentation
- **Mask2Former** : SOTA segmentation

### Détection d'Objets
- **YOLOv8/v9** : rapide et efficace
- **DETR / RT-DETR** : approche transformer
- **Co-DETR** : SOTA détection

## Pipeline CV Complet

### 1. Configuration et Data Loading

```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Configuration
class CFG:
    seed = 42
    img_size = 384  # Adapter selon la compétition
    batch_size = 16
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-4
    n_folds = 5
    model_name = 'tf_efficientnetv2_m'  # timm model
    num_classes = 10  # ADAPTER
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scheduler = 'cosine'
    min_lr = 1e-6
    T_max = 20
```

### 2. Data Augmentation (clé du succès en CV)

```python
def get_transforms(phase='train'):
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                           fill_value=0, p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
```

### 3. Dataset Custom

```python
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])  # ADAPTER

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_test:
            return image
        else:
            label = torch.tensor(row['label'], dtype=torch.long)  # ADAPTER
            return image, label
```

### 4. Modèle avec timm

```python
class ImageModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                        num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Variante avec head custom (plus flexible)
class ImageModelCustomHead(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                           num_classes=0)  # Sans classifier
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
```

### 5. Training Loop Gold Medal

```python
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.append(outputs.softmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return running_loss / len(loader.dataset), all_preds, all_labels
```

### 6. Test-Time Augmentation (TTA)

```python
def predict_with_tta(model, loader, device, n_tta=5):
    """TTA : prédire avec augmentation au moment du test."""
    model.eval()
    all_preds = []

    tta_transforms = [
        A.Compose([A.Resize(CFG.img_size, CFG.img_size),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.VerticalFlip(p=1.0),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
        A.Compose([A.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.9, 1.0)),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
    ]

    # Sans TTA + avec TTA → moyenne
    for tta_idx in range(min(n_tta, len(tta_transforms) + 1)):
        preds = []
        with torch.no_grad():
            for images in tqdm(loader, desc=f'TTA {tta_idx}'):
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = images.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                preds.append(outputs.softmax(dim=1).cpu().numpy())
        all_preds.append(np.concatenate(preds))

    # Moyenne des TTA
    return np.mean(all_preds, axis=0)
```

### 7. Multi-Scale Training

```python
# Entraîner sur plusieurs tailles d'image puis ensembler
image_sizes = [256, 384, 512]
all_preds = []

for img_size in image_sizes:
    CFG.img_size = img_size
    # ... entraîner et prédire avec cette taille
    # all_preds.append(test_preds)

final_preds = np.mean(all_preds, axis=0)
```

## Stratégies Gold Medal CV

1. **Multiple architectures** : EfficientNet + Swin + ConvNeXt
2. **Multiple tailles d'images** : 384 + 512 + 768
3. **Data augmentation lourde** : Cutout, Mixup, CutMix
4. **TTA** : toujours 4-8x TTA pour la soumission finale
5. **Progressive resizing** : commencer petit, augmenter la taille
6. **Label smoothing** : `nn.CrossEntropyLoss(label_smoothing=0.1)`
7. **SAM optimizer** : Sharpness-Aware Minimization pour meilleure généralisation
8. **Knowledge distillation** : utiliser un grand modèle pour enseigner un petit
9. **External data** : vérifier si autorisé, ajouter des données externes

## Mixup et CutMix

```python
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

Adapte TOUJOURS au type de tâche CV : classification, segmentation, détection, ou autre.
