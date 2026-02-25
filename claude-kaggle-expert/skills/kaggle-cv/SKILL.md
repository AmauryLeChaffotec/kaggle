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

## Instance Segmentation (Mask R-CNN)

### Mask R-CNN PyTorch (TorchVision)

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_maskrcnn_model(num_classes=2):
    """Mask R-CNN pré-entraîné COCO, adapté à N classes."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
        box_detections_per_img=512
    )
    # Remplacer le box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Remplacer le mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model
```

### Configuration Mask R-CNN (Matterport)

```python
from mrcnn.config import Config
import mrcnn.model as modellib

class CompetitionConfig(Config):
    NAME = 'competition'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 9
    BACKBONE = 'resnet50'
    NUM_CLASSES = 2  # background + cible

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)    # ADAPTER selon taille des objets
    TRAIN_ROIS_PER_IMAGE = 64
    MAX_GT_INSTANCES = 14
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.95         # Seuil haut = moins de faux positifs
    DETECTION_NMS_THRESHOLD = 0.1           # NMS strict

    STEPS_PER_EPOCH = 150
    VALIDATION_STEPS = 125

    # Poids des loss pour multi-task learning
    LOSS_WEIGHTS = {
        "rpn_class_loss": 30.0,
        "rpn_bbox_loss": 0.8,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.2
    }
```

### Dataset Instance Segmentation (PyTorch)

```python
class InstanceSegDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.image_ids = df['image_id'].unique()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.imread(os.path.join(self.image_dir, image_id + '.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        records = self.df[self.df['image_id'] == image_id]
        masks, boxes = [], []

        for _, row in records.iterrows():
            mask = rle_decode(row['annotation'], (img.shape[0], img.shape[1]))
            masks.append(mask)
            pos = np.where(mask)
            boxes.append([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])

        masks = np.array(masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(masks), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            'boxes': boxes, 'labels': labels, 'masks': masks,
            'image_id': torch.tensor([idx]), 'area': area,
            'iscrowd': torch.zeros(len(masks), dtype=torch.int64)
        }

        img = torch.from_numpy(img).permute(2, 0, 1)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.image_ids)
```

### RLE Encoding / Decoding

```python
def rle_encode(mask, min_threshold=1e-3):
    """Encode un masque binaire en Run-Length Encoding pour soumission."""
    if np.max(mask) < min_threshold:
        return ''
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    """Decode un RLE en masque binaire numpy."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # .T pour aligner avec le format RLE Kaggle

# Multi-instance : séparer les composants connectés
from skimage.morphology import label as sk_label

def multi_rle_encode(mask):
    """Encode chaque instance connectée séparément."""
    labels = sk_label(mask)
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]
```

### Inference + Overlap Removal + NMS

```python
def predict_instances(model, dataset, device, min_score=0.5, mask_threshold=0.5):
    """Inference Mask R-CNN avec suppression des chevauchements."""
    model.eval()
    submissions = []

    for sample in dataset:
        img = sample['image']
        image_id = sample['image_id']

        with torch.no_grad():
            result = model([img.to(device)])[0]

        previous_masks = []
        for i in range(len(result['scores'])):
            score = result['scores'][i].cpu().item()
            if score < min_score:
                continue

            mask = result['masks'][i, 0].cpu().numpy()
            binary = (mask > mask_threshold).astype(np.uint8)

            # Supprimer les pixels qui chevauchent des masques déjà acceptés
            for prev in previous_masks:
                binary[np.logical_and(binary, prev)] = 0

            if binary.sum() > 0:
                previous_masks.append(binary)
                submissions.append((image_id, rle_encode(binary)))

        if not any(img_id == image_id for img_id, _ in submissions):
            submissions.append((image_id, ''))  # Prédiction vide

    return pd.DataFrame(submissions, columns=['ImageId', 'EncodedPixels'])
```

## Multi-Stage Training (Freeze/Unfreeze)

### Pattern 3 Phases — PyTorch

```python
def multi_stage_train(model, train_loader, val_loader, device, num_classes):
    """Entraînement en 3 phases : heads → all (LR haute) → all (LR basse)."""

    # === PHASE 1 : Entraîner seulement le head (backbone gelé) ===
    for param in model.parameters():
        param.requires_grad = False

    # Dégeler seulement le classifier head
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3)  # LR haute pour le head
    print("Phase 1: Training head only (backbone frozen)")
    for epoch in range(3):
        train_one_epoch(model, train_loader, optimizer, None, criterion, device)

    # === PHASE 2 : Dégeler tout, LR moyenne ===
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    print("Phase 2: Training all layers (medium LR)")
    for epoch in range(10):
        train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)

    # === PHASE 3 : Fine-tuning, LR basse ===
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    print("Phase 3: Fine-tuning all layers (low LR)")
    for epoch in range(5):
        train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)

    return model
```

### Pattern Mask R-CNN (Matterport)

```python
LR = 0.003

# Phase 1 : heads only, LR haute, sans augmentation
model.train(dataset_train, dataset_val,
            learning_rate=LR * 2,
            epochs=2,
            layers='heads',
            augmentation=None)

# Phase 2 : all layers, LR normale, avec augmentation
model.train(dataset_train, dataset_val,
            learning_rate=LR,
            epochs=14,
            layers='all',
            augmentation=augmentation)

# Phase 3 : all layers, LR réduite (fine-tuning)
model.train(dataset_train, dataset_val,
            learning_rate=LR / 2,
            epochs=22,
            layers='all',
            augmentation=augmentation)
```

### Differential Learning Rates (Pattern Avancé)

```python
def get_layer_groups(model):
    """Groupes de paramètres avec LR différenciés."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name or 'encoder' in name or 'features' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    return backbone_params, head_params

backbone_params, head_params = get_layer_groups(model)
optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 1e-5},   # Backbone : LR basse
    {'params': head_params, 'lr': 1e-3},        # Head : LR haute (100x)
])
```

## FocalLoss

### Implémentation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss pour classes déséquilibrées.

    gamma=0 : équivalent à BCE
    gamma=2 : standard (focus fort sur exemples difficiles)
    gamma=5 : focus extrême (rares positifs)
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Poids par classe (optionnel)

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(f"Target size ({target.size()}) != input size ({input.size()})")

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        if self.alpha is not None:
            loss = self.alpha * loss

        return loss.sum(dim=1).mean()

# Usage
criterion = FocalLoss(gamma=2.0)

# Avec poids par classe
class_weights = torch.tensor([1.0, 5.0, 10.0])  # Plus de poids sur classes rares
criterion = FocalLoss(gamma=2.0, alpha=class_weights.to(device))
```

### Quand Utiliser FocalLoss

| Situation | Loss recommandée |
|-----------|-----------------|
| Classes équilibrées | CrossEntropyLoss |
| Déséquilibre léger (1:5) | CrossEntropyLoss + class weights |
| Déséquilibre fort (1:10 à 1:100) | **FocalLoss gamma=2** |
| Déséquilibre extrême (1:1000+) | **FocalLoss gamma=3-5** |
| Détection d'objets (beaucoup de background) | **FocalLoss gamma=2** |
| Segmentation médicale (petites lésions) | **FocalLoss gamma=2** + Dice Loss |
| Multi-label avec classes rares | **FocalLoss gamma=2** |

## Threshold Optimization

### OptimizedRounder (Classification Ordinale / QWK)

```python
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

class OptimizedRounder:
    """Optimise les seuils pour classification ordinale (QWK, Kappa)."""
    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=False)
        return -cohen_kappa_score(y, preds, weights='quadratic')

    def fit(self, X, y):
        initial_coef = [0.5, 1.5, 2.5]  # ADAPTER au nombre de classes - 1
        self.coef_ = minimize(self._kappa_loss, initial_coef, args=(X, y),
                              method='nelder-mead',
                              options={'maxiter': 10000}).x
        return self

    def predict(self, X):
        return pd.cut(X, [-np.inf] + list(np.sort(self.coef_)) + [np.inf], labels=False)

# Usage
rounder = OptimizedRounder()
rounder.fit(oof_preds, oof_labels)
print(f"Seuils optimisés : {rounder.coef_}")
optimized_preds = rounder.predict(test_preds)
```

### Per-Class Threshold Optimization (F1 / Multi-label)

```python
from scipy import optimize as opt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def f1_soft(preds, targets, thresholds, d=25.0):
    """F1 avec sigmoid douce pour optimisation gradient-free."""
    preds = sigmoid(d * (preds - thresholds))
    targets = targets.astype(np.float64)
    score = 2.0 * (preds * targets).sum(axis=0) / ((preds + targets).sum(axis=0) + 1e-6)
    return score

def optimize_thresholds(preds, targets, n_classes):
    """Optimise un seuil par classe pour maximiser le F1."""
    params = np.zeros(n_classes)
    wd = 1e-5
    error = lambda p: np.concatenate((f1_soft(preds, targets, p) - 1.0, wd * p), axis=None)
    thresholds, _ = opt.leastsq(error, params)
    return thresholds

# CV pour éviter l'overfitting des seuils
from sklearn.model_selection import KFold

def optimize_thresholds_cv(preds, targets, n_classes, cv=10):
    """Thresholds avec cross-validation."""
    th_sum = np.zeros(n_classes)
    for i in range(cv):
        idx = np.random.permutation(len(preds))
        mid = len(idx) // 2
        th_i = optimize_thresholds(preds[idx[:mid]], targets[idx[:mid]], n_classes)
        th_sum += th_i
    return th_sum / cv

thresholds = optimize_thresholds_cv(oof_preds, oof_labels, NUM_CLASSES)
final_preds = (test_preds > thresholds).astype(int)
```

### Threshold Matching (Distribution du LB)

```python
def fit_to_distribution(preds, target_fractions, n_classes):
    """Ajuste les seuils pour que les prédictions matchent une distribution cible."""
    def count_soft(preds, thresholds, d=50.0):
        return sigmoid(d * (preds - thresholds)).mean(axis=0)

    params = np.zeros(n_classes)
    wd = 1e-5
    error = lambda p: np.concatenate((count_soft(preds, p) - target_fractions, wd * p))
    thresholds, _ = opt.leastsq(error, params)
    return thresholds

# Exemple : matcher les proportions du leaderboard public
lb_fractions = np.array([0.362, 0.044, 0.075, 0.059, ...])  # ADAPTER
thresholds = fit_to_distribution(test_preds, lb_fractions, NUM_CLASSES)
```

## Custom Channels (RGBY, Satellite, N Canaux)

### Charger des Images Multi-canaux

```python
def load_rgby(path, image_id):
    """Charge une image 4 canaux RGBY depuis des fichiers séparés."""
    channels = ['red', 'green', 'blue', 'yellow']
    img = [cv2.imread(os.path.join(path, f'{image_id}_{ch}.png'), cv2.IMREAD_GRAYSCALE)
           .astype(np.float32) / 255.0
           for ch in channels]
    return np.stack(img, axis=-1)  # (H, W, 4)

# Pour images satellite multi-bandes (ex: Sentinel-2, 13 bandes)
def load_multispectral(path, image_id, bands=['B01','B02','B03','B04','B05','B06']):
    """Charge N bandes spectrales."""
    img = [cv2.imread(os.path.join(path, f'{image_id}_{b}.tif'), cv2.IMREAD_UNCHANGED)
           .astype(np.float32)
           for b in bands]
    return np.stack(img, axis=-1)  # (H, W, N)
```

### Modifier le 1er Conv Layer pour N Canaux

```python
import torch.nn as nn

def adapt_first_conv(model, in_channels=4):
    """Remplace le 1er conv layer pour accepter N canaux au lieu de 3.
    Initialise les nouveaux canaux comme moyenne des poids existants.
    """
    # Trouver le premier Conv2d
    if hasattr(model, 'conv1'):
        old_conv = model.conv1
    elif hasattr(model, 'features') and hasattr(model.features[0], 'weight'):
        old_conv = model.features[0]
    elif hasattr(model, 'stem'):
        old_conv = model.stem[0]
    else:
        raise ValueError("Cannot find first conv layer")

    # Créer nouveau conv avec N canaux
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # Copier les poids existants (3 canaux)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # Canaux supplémentaires = moyenne des 3 premiers
        for c in range(3, in_channels):
            new_conv.weight[:, c:c+1, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

    # Remplacer dans le modèle
    if hasattr(model, 'conv1'):
        model.conv1 = new_conv
    elif hasattr(model, 'features'):
        model.features[0] = new_conv
    elif hasattr(model, 'stem'):
        model.stem[0] = new_conv

    return model

# Usage avec timm
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=NUM_CLASSES)
model = adapt_first_conv(model, in_channels=4)

# Ou directement avec timm (quand supporté)
model = timm.create_model('efficientnet_b4', pretrained=True,
                           num_classes=NUM_CLASSES, in_chans=4)
```

### Normalisation Per-Channel

```python
def compute_channel_stats(dataset, n_channels=4):
    """Calcule mean et std par canal sur le dataset."""
    mean_sum = np.zeros(n_channels)
    std_sum = np.zeros(n_channels)
    n = 0
    for img, _ in dataset:
        img = img.numpy() if torch.is_tensor(img) else img
        mean_sum += img.reshape(-1, n_channels).mean(axis=0)
        std_sum += img.reshape(-1, n_channels).std(axis=0)
        n += 1
    return mean_sum / n, std_sum / n

# Exemple pour RGBY
# mean = [0.0807, 0.0526, 0.0549, 0.0828]
# std  = [0.1370, 0.1015, 0.1531, 0.1381]
```

## Medical Imaging (DICOM)

### Chargement DICOM

```python
import pydicom

def load_dicom(filepath):
    """Charge un fichier DICOM et retourne l'image pixel."""
    ds = pydicom.read_file(filepath)
    image = ds.pixel_array

    # Métadonnées utiles
    metadata = {
        'patient_id': str(ds.PatientID) if hasattr(ds, 'PatientID') else None,
        'modality': str(ds.Modality) if hasattr(ds, 'Modality') else None,
        'rows': ds.Rows,
        'cols': ds.Columns,
        'pixel_spacing': list(ds.PixelSpacing) if hasattr(ds, 'PixelSpacing') else None,
        'view_position': str(ds[0x0018, 0x5101].value) if (0x0018, 0x5101) in ds else None,
    }

    # Grayscale → RGB (pour les modèles pré-entraînés sur ImageNet)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = np.stack((image,) * 3, axis=-1)

    return image, metadata
```

### Preprocessing Médical

```python
def crop_image_from_gray(img, tol=7):
    """Supprime les bords gris/noirs (padding) d'une image médicale."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        # Vérifier qu'on ne crop pas trop (> 70% supprimé)
        cropped = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        if cropped.shape[0] / img.shape[0] < 0.3:
            return img  # Retourner l'original si crop trop agressif
        return np.stack([img[:, :, c][np.ix_(mask.any(1), mask.any(0))]
                         for c in range(3)], axis=-1)

def enhance_image(image, alpha=1.5, sigma=1.0):
    """Enhancement Gaussien (sharpening) pour images médicales."""
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    enhanced = cv2.addWeighted(image, alpha, blurred, 1 - alpha, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def medical_preprocess(filepath, target_size=(512, 512)):
    """Pipeline complet preprocessing médical."""
    image, metadata = load_dicom(filepath)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, target_size)
    image = enhance_image(image)
    return image, metadata
```

### Balancement des Classes Médicales (Oversampling)

```python
def balance_medical_dataset(image_paths, annotations, oversample_factor=3):
    """Oversampling des cas positifs (lésions) pour compétitions médicales."""
    positive_ids = annotations[annotations['EncodedPixels'] != '-1']['ImageId'].unique()
    positive_paths = [p for p in image_paths if p.split('/')[-1].replace('.dcm','') in positive_ids]

    print(f"Positifs: {len(positive_paths)}/{len(image_paths)} "
          f"({100*len(positive_paths)/len(image_paths):.1f}%)")

    # Oversampler les positifs
    balanced = list(image_paths) + positive_paths * oversample_factor
    np.random.shuffle(balanced)

    new_pos = sum(1 for p in balanced if p in positive_paths)
    print(f"Après oversampling: {new_pos}/{len(balanced)} ({100*new_pos/len(balanced):.1f}%)")
    return balanced
```

## Class Imbalance Strategies (Récapitulatif)

| Stratégie | Quand | Code |
|-----------|-------|------|
| **Class weights** | Déséquilibre léger | `nn.CrossEntropyLoss(weight=class_weights)` |
| **Focal Loss** | Déséquilibre fort | `FocalLoss(gamma=2)` |
| **Oversampling** | Positifs rares | Dupliquer les exemples positifs 2-4× |
| **Weighted sampler** | Batch équilibré | `WeightedRandomSampler(weights, num_samples)` |
| **Dice + BCE** | Segmentation | `0.5 * dice_loss + 0.5 * bce_loss` |
| **Stratified K-Fold** | CV équilibré | `StratifiedKFold(n_splits=5)` |

```python
# Weighted Random Sampler (batches équilibrés)
from torch.utils.data import WeightedRandomSampler

class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

Adapte TOUJOURS au type de tâche CV : classification, segmentation, détection, ou autre.

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'analyse, TOUJOURS sauvegarder :
1. Rapport dans : `reports/cv/YYYY-MM-DD_<description>.md`
2. Contenu : stratégie recommandée, techniques clés, code snippets, recommandations
3. Confirmer à l'utilisateur le chemin du rapport sauvegardé
