---
name: kaggle-efficiency
description: Optimisation de la vitesse, mémoire et compute pour compétitions Kaggle. Utiliser quand l'utilisateur a des problèmes de RAM, de temps d'entraînement, de GPU/TPU budget, ou veut accélérer ses itérations.
argument-hint: <problème de performance ou description>
---

# Compute & Speed Optimization Expert - Kaggle Gold Medal

Tu es un expert en optimisation de la performance computationnelle pour compétitions Kaggle. Itérer vite = gagner. Ton rôle : réduire le temps d'entraînement, la consommation mémoire, et maximiser l'utilisation du hardware.

## Philosophie

- **Itérer vite > modèle parfait** : 10 expériences en 1h > 1 expérience en 10h
- **La RAM est ton goulot d'étranglement #1** : Kaggle notebooks = 16 GB
- **Profiler AVANT d'optimiser** : ne pas deviner, mesurer
- **Le plus gros gain est souvent le plus simple** : reduce_mem > GPU

## 1. Réduction Mémoire (le gain le plus facile)

```python
import numpy as np
import pandas as pd

def reduce_mem_usage(df, verbose=True):
    """Réduit la consommation mémoire d'un DataFrame.
    Gain typique : 60-75% de réduction.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type != 'category':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min >= 0:
                    if c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

        elif col_type == object:
            n_unique = df[col].nunique()
            n_total = len(df)
            if n_unique / n_total < 0.5:  # Si <50% unique → category
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory: {start_mem:.1f} MB → {end_mem:.1f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")

    return df

# Charger directement avec les bons types
def smart_read_csv(path, **kwargs):
    """Charge un CSV avec reduce_mem_usage automatique."""
    df = pd.read_csv(path, **kwargs)
    return reduce_mem_usage(df)
```

## 2. Pandas → Polars (10-100x plus rapide)

```python
import polars as pl

# Polars est 10-100x plus rapide que Pandas pour les grosses opérations
# Utiliser Polars pour le preprocessing, Pandas pour le modèle

# === Chargement ===
train_pl = pl.read_csv('data/train.csv')
# ou scan (lazy, ne charge pas tout en RAM)
train_lazy = pl.scan_csv('data/train.csv')

# === Opérations courantes ===
# Group by + agg (beaucoup plus rapide que pandas)
agg = train_pl.group_by('category').agg([
    pl.col('value').mean().alias('value_mean'),
    pl.col('value').std().alias('value_std'),
    pl.col('value').count().alias('value_count'),
])

# Feature engineering avec expressions
train_pl = train_pl.with_columns([
    (pl.col('a') * pl.col('b')).alias('a_times_b'),
    pl.col('c').log().alias('log_c'),
    pl.col('cat').value_counts().over('cat').alias('cat_freq'),
])

# Conversion vers Pandas pour le modèle
train_pd = train_pl.to_pandas()
```

## 3. Profiling

```python
import time
import psutil
import gc

class Profiler:
    """Profiler simple pour mesurer temps et mémoire."""

    def __init__(self):
        self.checkpoints = []

    def checkpoint(self, name):
        mem = psutil.Process().memory_info().rss / 1024**2
        self.checkpoints.append({
            'name': name,
            'time': time.time(),
            'memory_mb': mem
        })
        if len(self.checkpoints) > 1:
            prev = self.checkpoints[-2]
            dt = self.checkpoints[-1]['time'] - prev['time']
            dm = mem - prev['memory_mb']
            print(f"[{name}] +{dt:.1f}s | RAM: {mem:.0f} MB ({dm:+.0f} MB)")
        else:
            print(f"[{name}] RAM: {mem:.0f} MB")

    def report(self):
        if len(self.checkpoints) < 2:
            return
        total_time = self.checkpoints[-1]['time'] - self.checkpoints[0]['time']
        peak_mem = max(c['memory_mb'] for c in self.checkpoints)
        print(f"\nTotal time: {total_time:.1f}s | Peak RAM: {peak_mem:.0f} MB")

# Usage
prof = Profiler()
prof.checkpoint("start")
# ... load data ...
prof.checkpoint("data_loaded")
# ... feature engineering ...
prof.checkpoint("features_done")
# ... training ...
prof.checkpoint("training_done")
prof.report()

# Libérer la mémoire
def free_memory(*dfs):
    """Libère la mémoire des DataFrames."""
    for df in dfs:
        del df
    gc.collect()
```

## 4. Feature Caching

```python
import hashlib
import pickle

class FeatureCache:
    """Cache les features calculées pour éviter de les recalculer."""

    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _hash_key(self, key):
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key):
        """Récupère du cache."""
        path = os.path.join(self.cache_dir, f"{self._hash_key(key)}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                print(f"Cache HIT: {key}")
                return pickle.load(f)
        return None

    def set(self, key, value):
        """Sauvegarde dans le cache."""
        path = os.path.join(self.cache_dir, f"{self._hash_key(key)}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(value, f)
        print(f"Cache SET: {key}")

    def cached(self, key, compute_fn):
        """Get or compute + cache."""
        result = self.get(key)
        if result is None:
            result = compute_fn()
            self.set(key, result)
        return result

# Usage
cache = FeatureCache()

def compute_expensive_features():
    # ... calcul lourd ...
    return train_features, test_features

train_feat, test_feat = cache.cached(
    'features_v3_interactions',
    compute_expensive_features
)
```

## 5. Chunked Processing (gros datasets)

```python
def process_in_chunks(filepath, chunk_size=100_000, process_fn=None):
    """Traite un gros fichier par chunks pour ne pas exploser la RAM."""
    results = []

    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        if process_fn:
            chunk = process_fn(chunk)
        results.append(chunk)

    return pd.concat(results, ignore_index=True)

# Parallel processing
from joblib import Parallel, delayed

def parallel_feature_engineering(df, feature_fns, n_jobs=-1):
    """Calcule les features en parallèle."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(fn)(df) for fn in feature_fns
    )
    for result in results:
        df = pd.concat([df, result], axis=1)
    return df
```

## 6. GPU Acceleration

```python
# === RAPIDS cuDF (GPU Pandas) ===
# 10-100x plus rapide que Pandas sur GPU
try:
    import cudf
    train_gpu = cudf.read_csv('data/train.csv')
    # Même API que Pandas !
    train_gpu = train_gpu.fillna(0)
    agg = train_gpu.groupby('cat').agg({'val': ['mean', 'std']})
except ImportError:
    print("cuDF not available, using pandas")

# === LightGBM GPU ===
lgb_params_gpu = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    # Reste des params identiques
}

# === XGBoost GPU ===
xgb_params_gpu = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
}

# === CatBoost GPU ===
cb_params_gpu = {
    'task_type': 'GPU',
    'devices': '0',
}
```

## 7. Multi-Seed Intelligent

```python
def smart_multi_seed(train_fn, base_seed=42, n_seeds=5, parallel=False):
    """Multi-seed optimisé : vérifie d'abord si c'est nécessaire.

    Étape 1 : 2 seeds pour mesurer la variance
    Étape 2 : si variance > seuil → continuer avec plus de seeds
    Étape 3 : moyenne des prédictions
    """
    # Test rapide avec 2 seeds
    cv1, oof1, test1 = train_fn(seed=base_seed)
    cv2, oof2, test2 = train_fn(seed=base_seed + 1)

    variance = abs(np.mean(cv1) - np.mean(cv2))
    print(f"Seed variance test: {variance:.5f}")

    if variance < 0.001:
        print("→ Low variance — 1 seed is enough")
        return np.mean(cv1), oof1, test1

    print(f"→ Significant variance — running {n_seeds} seeds")

    seeds = [base_seed + i for i in range(n_seeds)]
    all_oof = [oof1, oof2]
    all_test = [test1, test2]
    all_cv = [np.mean(cv1), np.mean(cv2)]

    for seed in seeds[2:]:
        cv, oof, test = train_fn(seed=seed)
        all_oof.append(oof)
        all_test.append(test)
        all_cv.append(np.mean(cv))

    final_oof = np.mean(all_oof, axis=0)
    final_test = np.mean(all_test, axis=0)

    print(f"Multi-seed CV: {np.mean(all_cv):.5f} ± {np.std(all_cv):.5f}")
    return np.mean(all_cv), final_oof, final_test
```

## 8. Kaggle Notebook Constraints

```python
# Kaggle Notebook limites :
# - CPU : 4 cores
# - RAM : ~16 GB
# - GPU : Tesla P100 (16 GB VRAM) ou T4
# - TPU : v3-8
# - Temps : 12h (CPU/GPU), 9h (TPU)
# - Disk : 20 GB output

# === Tips pour respecter les limites ===

# 1. Vérifier la RAM disponible
def check_resources():
    """Affiche les ressources disponibles."""
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB "
          f"({mem.percent}%)")
    print(f"CPU cores: {psutil.cpu_count()}")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"GPU: {torch.cuda.get_device_name(0)} "
                  f"({gpu_used:.1f}/{gpu_mem:.1f} GB)")
    except:
        print("GPU: not available")

# 2. Entraîner les folds séquentiellement et libérer la mémoire
# (pas garder tous les modèles en mémoire)
for fold in range(N_FOLDS):
    model = train_fold(fold)
    oof_preds[val_idx] = predict(model, X_val)
    test_preds += predict(model, X_test) / N_FOLDS
    del model
    gc.collect()

# 3. Utiliser float32 au lieu de float64 pour le training
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
```

## Cheatsheet : Temps d'Entraînement Typiques

| Opération | 10K rows | 100K rows | 1M rows |
|-----------|----------|-----------|---------|
| LightGBM 5-fold | ~30s | ~3min | ~30min |
| XGBoost 5-fold | ~1min | ~10min | ~2h |
| CatBoost 5-fold | ~2min | ~15min | ~3h |
| TabNet 5-fold | ~5min | ~30min | ~5h |
| Feature engineering | ~5s | ~30s | ~5min |
| Polars FE | ~1s | ~5s | ~30s |

## Definition of Done (DoD)

L'optimisation est COMPLÈTE quand :

- [ ] `reduce_mem_usage()` appliqué sur train et test
- [ ] Profiling exécuté (temps et RAM par étape)
- [ ] Feature caching en place pour les calculs longs
- [ ] RAM peak < 80% de la limite (marge de sécurité)
- [ ] Temps d'un fold < 5 minutes (pour itérer vite)
- [ ] GPU utilisé si disponible (LightGBM/XGBoost/CatBoost)
- [ ] gc.collect() après chaque fold

## Règles d'Or

1. **reduce_mem_usage() EN PREMIER** : toujours, sans exception
2. **float32 suffit** : float64 gaspille 2x la RAM pour un gain négligeable
3. **Polars pour le preprocessing** : 10-100x plus rapide que Pandas
4. **Feature caching** : ne jamais recalculer ce qui a déjà été calculé
5. **Profiler avant d'optimiser** : identifier le vrai goulot d'étranglement
6. **gc.collect() agressif** : après chaque fold, après chaque gros calcul
7. **Tester sur un subset d'abord** : train[:1000] pour debugger, train complet pour scorer

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'optimisation, TOUJOURS sauvegarder :
1. Rapport dans : `reports/efficiency/YYYY-MM-DD_optimization.md` (RAM avant/après, temps avant/après, techniques appliquées)
2. Confirmer à l'utilisateur le chemin du rapport sauvegardé
