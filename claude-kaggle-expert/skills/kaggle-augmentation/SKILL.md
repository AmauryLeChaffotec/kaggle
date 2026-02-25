---
name: kaggle-augmentation
description: Stratégies d'augmentation de données pour compétitions Kaggle. Utiliser quand l'utilisateur veut augmenter ses données tabulaires, images, texte ou séries temporelles avec SMOTE, Mixup, CutMix, back-translation, etc.
argument-hint: <type de données ou technique d'augmentation>
---

# Data Augmentation Expert - Kaggle Gold Medal

Tu es un expert en augmentation de données pour compétitions Kaggle. L'augmentation est essentielle pour améliorer la généralisation, surtout sur les petits datasets.

## Philosophie

- **Augmenter ≠ créer des données fausses** : l'augmentation doit préserver le signal
- **L'augmentation est spécifique au domaine** : une rotation de 180° est OK pour de la microscopie, pas pour du texte manuscrit
- **Trop d'augmentation = bruit** : commencer léger, augmenter progressivement
- **L'augmentation aide surtout les NN** : les GBDT sont moins sensibles

## 1. Augmentation Tabulaire

### SMOTE et variantes (oversampling)

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

def handle_imbalance(X_train, y_train, method='smote', sampling_ratio=0.5):
    """Traitement du déséquilibre de classes.

    ATTENTION : Appliquer UNIQUEMENT sur le train (pas val/test) !
    ATTENTION : Appliquer APRÈS le split CV !
    """
    print(f"Before: {pd.Series(y_train).value_counts().to_dict()}")

    if method == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_ratio, random_state=42, k_neighbors=5)
    elif method == 'adasyn':
        # Plus d'échantillons pour les exemples difficiles
        sampler = ADASYN(sampling_strategy=sampling_ratio, random_state=42)
    elif method == 'borderline':
        # Focus sur les exemples proches de la frontière
        sampler = BorderlineSMOTE(sampling_strategy=sampling_ratio, random_state=42)
    elif method == 'smote_tomek':
        # SMOTE + nettoyage Tomek Links
        sampler = SMOTETomek(sampling_strategy=sampling_ratio, random_state=42)
    elif method == 'smote_enn':
        # SMOTE + nettoyage ENN
        sampler = SMOTEENN(sampling_strategy=sampling_ratio, random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=42)

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f"After ({method}): {pd.Series(y_resampled).value_counts().to_dict()}")

    return X_resampled, y_resampled

# IMPORTANT : Pour GBDT, préférer class_weight ou scale_pos_weight plutôt que SMOTE
# LightGBM: is_unbalance=True ou scale_pos_weight
# XGBoost: scale_pos_weight = count(neg) / count(pos)
# CatBoost: auto_class_weights='Balanced'
```

### Mixup Tabulaire

```python
def mixup_tabular(X, y, alpha=0.2, n_augmented=None):
    """Mixup pour données tabulaires : interpolation linéaire entre exemples.
    Crée des exemples synthétiques entre paires d'exemples existants.
    """
    n = len(X)
    if n_augmented is None:
        n_augmented = n

    # Indices aléatoires pour les paires
    idx1 = np.random.randint(0, n, n_augmented)
    idx2 = np.random.randint(0, n, n_augmented)

    # Lambda de mixup (distribution Beta)
    lam = np.random.beta(alpha, alpha, n_augmented).reshape(-1, 1)

    X_mix = lam * X.iloc[idx1].values + (1 - lam) * X.iloc[idx2].values
    y_mix = lam.squeeze() * y.iloc[idx1].values + (1 - lam.squeeze()) * y.iloc[idx2].values

    X_aug = pd.DataFrame(X_mix, columns=X.columns)
    return X_aug, y_mix

def noise_injection(X, noise_level=0.01, num_cols=None):
    """Ajouter du bruit gaussien aux features numériques."""
    X_aug = X.copy()
    if num_cols is None:
        num_cols = X.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        std = X[col].std()
        noise = np.random.normal(0, noise_level * std, len(X))
        X_aug[col] = X_aug[col] + noise

    return X_aug
```

### Pseudo-Labeling

```python
def pseudo_labeling(model, X_train, y_train, X_test, confidence_threshold=0.95,
                    task='classification'):
    """Pseudo-labeling : utiliser les prédictions confiantes sur le test
    comme données d'entraînement supplémentaires.

    ATTENTION : Peut amplifier les biais du modèle !
    Utiliser uniquement quand le modèle est déjà bon (>0.85 AUC).
    """
    # Prédire le test
    if task == 'classification':
        test_probs = model.predict_proba(X_test)[:, 1]
        # Sélectionner les prédictions confiantes
        confident_pos = test_probs > confidence_threshold
        confident_neg = test_probs < (1 - confidence_threshold)
        confident_mask = confident_pos | confident_neg

        pseudo_labels = (test_probs > 0.5).astype(int)
    else:
        test_preds = model.predict(X_test)
        # Pour la régression, sélectionner les prédictions proches de la moyenne
        residual_std = np.std(y_train - model.predict(X_train))
        confident_mask = np.abs(test_preds - test_preds.mean()) < 2 * residual_std
        pseudo_labels = test_preds

    n_pseudo = confident_mask.sum()
    print(f"Pseudo-labeled {n_pseudo}/{len(X_test)} test samples "
          f"({n_pseudo/len(X_test)*100:.1f}%)")

    if n_pseudo > 0:
        X_pseudo = X_test[confident_mask]
        y_pseudo = pseudo_labels[confident_mask]

        X_augmented = pd.concat([X_train, X_pseudo], axis=0).reset_index(drop=True)
        y_augmented = np.concatenate([y_train, y_pseudo])

        return X_augmented, y_augmented
    else:
        print("No confident predictions — not using pseudo-labels")
        return X_train, y_train
```

## 2. Augmentation d'Images

### Albumentations Pipeline (léger → fort)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(image_size=224, level='medium'):
    """Pipelines d'augmentation par niveau d'intensité."""

    if level == 'light':
        train_transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    elif level == 'medium':
        train_transforms = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])

    elif level == 'heavy':
        train_transforms = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=45, p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(20, 80)),
                A.GaussianBlur(blur_limit=7),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.ElasticTransform(alpha=1, sigma=50),
                A.OpticalDistortion(distort_limit=0.5),
            ], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=image_size//8,
                           max_width=image_size//8, fill_value=0, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])

    val_transforms = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])

    return train_transforms, val_transforms
```

### CutMix et Mixup pour Images

```python
def cutmix(images, labels, alpha=1.0):
    """CutMix : coupe un rectangle d'une image et le colle sur une autre."""
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)

    rand_index = torch.randperm(batch_size)

    # Bounding box
    W, H = images.size(2), images.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    images[:, :, x1:x2, y1:y2] = images[rand_index, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    labels_mixed = lam * labels + (1 - lam) * labels[rand_index]
    return images, labels_mixed

def mixup_images(images, labels, alpha=0.2):
    """Mixup : interpolation linéaire entre images."""
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size(0))

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    mixed_labels = lam * labels + (1 - lam) * labels[rand_index]

    return mixed_images, mixed_labels
```

## 3. Augmentation de Texte

```python
def text_augmentation(texts, method='all'):
    """Augmentation de texte pour NLP."""
    import random

    augmented = []

    for text in texts:
        words = text.split()

        if method in ['synonym', 'all']:
            # Synonym replacement (nécessite nltk ou wordnet)
            pass  # Voir nlpaug library

        if method in ['random_insert', 'all']:
            # Random insertion
            if len(words) > 2:
                idx = random.randint(0, len(words) - 1)
                words_copy = words.copy()
                words_copy.insert(idx, words[random.randint(0, len(words)-1)])
                augmented.append(' '.join(words_copy))

        if method in ['random_swap', 'all']:
            # Random swap
            if len(words) > 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words_copy = words.copy()
                words_copy[idx1], words_copy[idx2] = words_copy[idx2], words_copy[idx1]
                augmented.append(' '.join(words_copy))

        if method in ['random_delete', 'all']:
            # Random deletion
            if len(words) > 3:
                words_copy = [w for w in words if random.random() > 0.1]
                if words_copy:
                    augmented.append(' '.join(words_copy))

    return augmented

# Back-translation (le plus puissant)
def back_translate(texts, src_lang='en', pivot_lang='fr'):
    """Back-translation : en → fr → en pour paraphraser.
    Utilise MarianMT ou Google Translate API.
    """
    from transformers import MarianMTModel, MarianTokenizer

    # en → fr
    model_name_fwd = f'Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}'
    tokenizer_fwd = MarianTokenizer.from_pretrained(model_name_fwd)
    model_fwd = MarianMTModel.from_pretrained(model_name_fwd)

    # fr → en
    model_name_bwd = f'Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}'
    tokenizer_bwd = MarianTokenizer.from_pretrained(model_name_bwd)
    model_bwd = MarianMTModel.from_pretrained(model_name_bwd)

    augmented = []
    for text in texts:
        # Forward translation
        inputs = tokenizer_fwd(text, return_tensors="pt", truncation=True, max_length=512)
        translated = model_fwd.generate(**inputs)
        pivot_text = tokenizer_fwd.decode(translated[0], skip_special_tokens=True)

        # Backward translation
        inputs = tokenizer_bwd(pivot_text, return_tensors="pt", truncation=True, max_length=512)
        back_translated = model_bwd.generate(**inputs)
        augmented_text = tokenizer_bwd.decode(back_translated[0], skip_special_tokens=True)

        augmented.append(augmented_text)

    return augmented
```

## 4. Augmentation de Séries Temporelles

```python
def ts_jittering(series, noise_level=0.03):
    """Ajouter du bruit gaussien à une série temporelle."""
    noise = np.random.normal(0, noise_level * series.std(), len(series))
    return series + noise

def ts_scaling(series, sigma=0.1):
    """Mise à l'échelle aléatoire."""
    factor = np.random.normal(1, sigma)
    return series * factor

def ts_window_warping(series, window_ratio=0.1, scales=[0.5, 2.0]):
    """Warping temporel sur une fenêtre."""
    n = len(series)
    window_size = max(1, int(n * window_ratio))
    start = np.random.randint(0, n - window_size)
    scale = np.random.choice(scales)

    window = series[start:start + window_size]
    new_size = max(1, int(window_size * scale))
    warped = np.interp(
        np.linspace(0, 1, new_size),
        np.linspace(0, 1, window_size),
        window
    )

    result = np.concatenate([
        series[:start],
        warped,
        series[start + window_size:]
    ])

    # Resampler à la taille originale
    result = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(result)), result)
    return result

def ts_magnitude_warping(series, sigma=0.2, knots=4):
    """Warping de magnitude avec courbe lisse."""
    n = len(series)
    knot_positions = np.linspace(0, n-1, knots+2)
    knot_values = np.random.normal(1, sigma, knots+2)
    warp_curve = np.interp(np.arange(n), knot_positions, knot_values)
    return series * warp_curve
```

## Guide de Choix

| Données | Technique | Quand l'utiliser | Impact typique |
|---------|-----------|-----------------|----------------|
| Tabulaire | SMOTE | Classes déséquilibrées + NN | +0.01-0.03 |
| Tabulaire | Mixup | NN uniquement, dataset petit | +0.005-0.02 |
| Tabulaire | Pseudo-labeling | Modèle déjà bon (>0.85) | +0.001-0.01 |
| Tabulaire | Noise injection | NN uniquement, régularisation | +0.001-0.005 |
| Image | Albumentations light | Baseline | Obligatoire |
| Image | CutMix/Mixup | CNN training | +0.01-0.03 |
| Image | Heavy augmentation | Petit dataset, overfitting | +0.02-0.05 |
| Texte | Back-translation | NLP, dataset petit | +0.01-0.03 |
| Texte | Random augmentation | NLP basique | +0.005-0.01 |
| Time Series | Jittering | Forecasting, régularisation | +0.005-0.01 |
| Time Series | Window warping | Forecasting, diversité | +0.005-0.02 |

## Règles d'Or

1. **JAMAIS augmenter le val/test** : uniquement le train
2. **Augmenter APRÈS le split CV** : sinon data leakage
3. **Commencer léger** : augmentation progressive, mesurer l'impact
4. **Pour GBDT** : préférer class_weight à SMOTE
5. **Pour NN** : l'augmentation est presque toujours bénéfique
6. **Pseudo-labeling** : itératif (train → predict → add → retrain)
7. **Back-translation** : le meilleur ROI en NLP
8. **Diversifier** : combiner plusieurs techniques d'augmentation

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'augmentation, TOUJOURS sauvegarder :
1. Rapport dans : `reports/augmentation/YYYY-MM-DD_augmentation.md` (techniques appliquées, taille avant/après, CV impact)
2. Confirmer à l'utilisateur le chemin du rapport sauvegardé
