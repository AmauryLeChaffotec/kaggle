---
name: kaggle-inference
description: Pipeline d'inférence optimisé pour compétitions Kaggle. Utiliser quand l'utilisateur veut optimiser l'inférence (TTA, batch size, RAM/VRAM), créer un pipeline de prédiction robuste, ou préparer un kernel de soumission.
argument-hint: <type d'inférence ou contrainte>
---

# Inference Pipeline Expert - Kaggle Gold Medal

Tu es un expert en pipeline d'inférence pour compétitions Kaggle. Beaucoup de compétitions ont des contraintes de temps/mémoire à l'inférence. Ton rôle : créer un pipeline de prédiction rapide, robuste et qui maximise le score.

## Philosophie

- **L'inférence n'est PAS juste model.predict()** : TTA, ensembling, post-processing
- **Le temps d'inférence compte** : certaines compétitions ont un time limit strict
- **La RAM/VRAM est limitée** : batch processing, model offloading
- **Reproductibilité** : le pipeline doit donner le même résultat à chaque run

## 1. TTA — Test-Time Augmentation

### TTA pour Images

```python
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_tta_transforms(image_size=224):
    """TTA transforms standard pour CV.
    Chaque transform est appliquée au test, puis on moyenne les prédictions.
    """
    return [
        # Original
        A.Compose([A.Resize(image_size, image_size), A.Normalize(), ToTensorV2()]),
        # Horizontal flip
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0),
                   A.Normalize(), ToTensorV2()]),
        # Vertical flip
        A.Compose([A.Resize(image_size, image_size), A.VerticalFlip(p=1.0),
                   A.Normalize(), ToTensorV2()]),
        # Both flips
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0),
                   A.VerticalFlip(p=1.0), A.Normalize(), ToTensorV2()]),
        # Multi-scale (légèrement plus grand, center crop)
        A.Compose([A.Resize(int(image_size*1.1), int(image_size*1.1)),
                   A.CenterCrop(image_size, image_size),
                   A.Normalize(), ToTensorV2()]),
    ]

def predict_with_tta(model, dataset_class, test_data, tta_transforms,
                     device='cuda', batch_size=32):
    """Prédiction avec TTA : moyenne des prédictions sur chaque augmentation."""
    model.eval()
    all_preds = []

    for tta_idx, transform in enumerate(tta_transforms):
        dataset = dataset_class(test_data, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
        preds = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                output = model(batch)
                preds.append(torch.sigmoid(output).cpu().numpy())

        all_preds.append(np.concatenate(preds))
        print(f"TTA {tta_idx+1}/{len(tta_transforms)} done")

    # Moyenne des TTA
    final_preds = np.mean(all_preds, axis=0)
    print(f"TTA ensemble: {len(tta_transforms)} augmentations averaged")
    return final_preds
```

### TTA pour Tabulaire (Multi-Seed)

```python
def tabular_tta(models_by_seed, X_test, method='mean'):
    """TTA pour tabulaire = multi-seed averaging.
    Les modèles entraînés avec des seeds différents sont les 'augmentations'.
    """
    all_preds = []

    for seed, models in models_by_seed.items():
        # models = liste de modèles par fold
        seed_preds = np.zeros(len(X_test))
        for model in models:
            seed_preds += model.predict(X_test) / len(models)
        all_preds.append(seed_preds)

    if method == 'mean':
        final = np.mean(all_preds, axis=0)
    elif method == 'median':
        final = np.median(all_preds, axis=0)

    print(f"Tabular TTA: {len(models_by_seed)} seeds, method={method}")
    return final
```

### TTA pour NLP

```python
def nlp_tta(model, tokenizer, texts, max_lengths=[256, 384, 512],
            device='cuda', batch_size=16):
    """TTA pour NLP : varier la longueur de tokenisation."""
    all_preds = []

    for max_len in max_lengths:
        encodings = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_len, return_tensors='pt'
        )
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'], encodings['attention_mask']
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        preds = []
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                output = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device)
                )
                preds.append(torch.sigmoid(output.logits).cpu().numpy())

        all_preds.append(np.concatenate(preds))
        print(f"NLP TTA max_length={max_len} done")

    return np.mean(all_preds, axis=0)
```

## 2. Batch Inference (gestion mémoire)

```python
def batch_predict(model, X, batch_size=10000, predict_fn=None):
    """Prédiction par batch pour économiser la RAM."""
    n_samples = len(X)
    all_preds = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X[start:end] if isinstance(X, np.ndarray) else X.iloc[start:end]

        if predict_fn:
            preds = predict_fn(batch)
        else:
            preds = model.predict(batch)

        all_preds.append(preds)

        if start % (batch_size * 10) == 0:
            print(f"  Predicted {end}/{n_samples} ({100*end/n_samples:.0f}%)")

    return np.concatenate(all_preds)

def multi_model_inference(models, X_test, weights=None, batch_size=10000):
    """Inférence multi-modèle séquentielle (1 modèle à la fois en RAM)."""
    import gc

    n_models = len(models)
    if weights is None:
        weights = [1.0 / n_models] * n_models

    final_preds = np.zeros(len(X_test))

    for i, (model_path, weight) in enumerate(zip(models, weights)):
        # Charger le modèle
        if isinstance(model_path, str):
            import joblib
            model = joblib.load(model_path)
        else:
            model = model_path

        # Prédire par batch
        preds = batch_predict(model, X_test, batch_size)
        final_preds += preds * weight

        # Libérer la mémoire
        if isinstance(model_path, str):
            del model
            gc.collect()

        print(f"Model {i+1}/{n_models} done (weight={weight:.3f})")

    return final_preds
```

## 3. GPU Inference Optimization

```python
def gpu_inference_optimized(model, dataloader, device='cuda', use_amp=True):
    """Inférence GPU optimisée avec mixed precision."""
    model = model.to(device)
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) for b in batch]
            else:
                batch = batch.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(*batch) if isinstance(batch, list) else model(batch)
            else:
                output = model(*batch) if isinstance(batch, list) else model(batch)

            all_preds.append(output.cpu().numpy())

    return np.concatenate(all_preds)

# Optimisations supplémentaires pour l'inférence
# torch.backends.cudnn.benchmark = True  # Optimise les convolutions
# model = torch.jit.script(model)        # JIT compilation
# model.half()                           # FP16 inference (2x plus rapide)
```

## 4. Kaggle Submission Kernel Template

```python
def create_submission_kernel(model_paths, feature_pipeline_fn,
                             postprocess_fn=None, tta=False):
    """Template pour un kernel de soumission Kaggle.
    Gère : chargement modèles, preprocessing, inférence, post-processing, soumission.
    """
    import time
    start_time = time.time()

    # 1. Charger les données
    test = pd.read_csv('data/test.csv')
    sample_sub = pd.read_csv('data/sample_submission.csv')
    print(f"Test: {test.shape}")

    # 2. Feature engineering
    test = feature_pipeline_fn(test)
    features = [c for c in test.columns if c not in ['id', 'ID']]
    print(f"Features: {len(features)}")

    # 3. Inférence multi-modèle
    preds = multi_model_inference(model_paths, test[features])

    # 4. Post-processing
    if postprocess_fn:
        preds = postprocess_fn(preds)

    # 5. Soumission
    submission = sample_sub.copy()
    submission.iloc[:, -1] = preds
    submission.to_csv('submission.csv', index=False)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Submission: {submission.shape}")
    print(submission.head())

    return submission
```

## Definition of Done (DoD)

Le pipeline d'inférence est COMPLET quand :

- [ ] Pipeline end-to-end fonctionnel (load → preprocess → predict → submit)
- [ ] TTA implémenté si applicable (images: flips, NLP: max_lengths, tabular: multi-seed)
- [ ] Gestion mémoire : batch processing si dataset > RAM
- [ ] Temps d'inférence mesuré et dans les limites de la compétition
- [ ] Multi-modèle : chargement séquentiel si RAM limitée
- [ ] Post-processing intégré au pipeline
- [ ] Résultat identique à chaque run (reproductible)
- [ ] Submission validée (format, NaN, shape)

## Rapport de Sortie (OBLIGATOIRE)

À la fin du pipeline d'inférence, TOUJOURS sauvegarder :
1. Rapport dans : `reports/inference/YYYY-MM-DD_inference.md` (TTA config, temps d'inférence, RAM utilisée)
2. Script d'inférence dans : `src/inference.py` (pipeline end-to-end reproductible)
3. Submission dans : `submissions/sub_<description>_YYYY-MM-DD.csv`
4. Confirmer à l'utilisateur les chemins sauvegardés
