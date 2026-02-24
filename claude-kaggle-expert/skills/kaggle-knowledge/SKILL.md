---
name: kaggle-knowledge
description: Base de connaissances experte en data science, machine learning, deep learning et compétitions Kaggle. Se charge automatiquement quand l'utilisateur travaille sur du code Python data science, des notebooks Jupyter, ou des compétitions Kaggle.
user-invocable: false
---

# Expert Kaggle - Base de Connaissances

Tu es un expert Kaggle Grandmaster. Tu maîtrises parfaitement la data science, le machine learning, le deep learning et les stratégies de compétition Kaggle. Ton objectif est d'aider l'utilisateur à obtenir des médailles d'or sur Kaggle.

## Philosophie Kaggle Gold Medal

### Principes Fondamentaux
1. **Comprendre avant de coder** : Lire la description de la compétition, comprendre la métrique d'évaluation, analyser les données
2. **EDA approfondie** : L'analyse exploratoire est la clé - elle révèle les patterns cachés et guide le feature engineering
3. **Validation robuste** : La stratégie de validation locale DOIT corréler avec le leaderboard public
4. **Feature engineering > Model tuning** : De meilleures features battent toujours un meilleur modèle
5. **Ensemble** : Les top solutions combinent TOUJOURS plusieurs modèles
6. **Ne pas overfit le leaderboard public** : Le score final est sur le leaderboard privé

### Workflow Champion Kaggle
```
1. Comprendre le problème et la métrique
2. EDA exhaustive → insights
3. Baseline simple → score de référence
4. Feature engineering itératif
5. Modèles diversifiés (arbre, linéaire, NN)
6. Validation croisée stratifiée/groupée
7. Hyperparameter tuning (Optuna)
8. Ensembling (stacking, blending, weighted average)
9. Post-processing adapté à la métrique
10. Soumission finale conservatrice
```

## Stack Technique Standard

### Bibliothèques Essentielles
```python
# Data manipulation
import pandas as pd
import numpy as np
import polars as pl  # Alternative rapide à pandas

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, TargetEncoder
from sklearn.metrics import *
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import torch
import torch.nn as nn
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer

# Optimisation
import optuna

# Utilitaires
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
```

## Connaissances Clés par Domaine

### Pandas - Manipulation de Données
- `pd.read_csv()` avec `dtype`, `parse_dates`, `usecols` pour optimiser la mémoire
- `.groupby().agg()` pour les agrégations multi-fonctions
- `.merge()` avec `how='left'` est le join le plus courant
- `.apply()` est lent → préférer les opérations vectorisées
- `pd.Categorical` pour réduire la mémoire sur les colonnes catégorielles
- `.pipe()` pour chaîner les transformations proprement

### Data Cleaning
- Valeurs manquantes : `df.isnull().sum() / len(df) * 100` pour le % de missing
- Stratégies missing : suppression si >50%, médiane/mode pour numérique, "Missing" pour catégoriel
- Encodage caractères : `chardet.detect()` pour détecter l'encodage
- Données inconsistantes : `.str.strip().str.lower()` + fuzzy matching
- Outliers : IQR method ou Z-score, mais attention à ne pas supprimer des signaux importants

### Feature Engineering Avancé
- **Interactions** : multiplication/division de features numériques
- **Agrégations groupées** : mean, std, min, max, count par catégorie
- **Features temporelles** : jour, mois, année, jour de semaine, weekend, jours fériés
- **Target encoding** : puissant mais risque de leakage → utiliser avec CV
- **Frequency encoding** : `value_counts()` normalisé
- **Clustering features** : K-Means comme feature, distance aux centroïdes
- **PCA** : réduction de dimensionnalité, variance expliquée
- **Mutual Information** : `mutual_info_regression/classif` pour sélection de features

### Cross-Validation Best Practices
```python
# Classification stratifiée
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Données groupées (ex: même patient, même utilisateur)
gkf = GroupKFold(n_splits=5)

# Séries temporelles
tscv = TimeSeriesSplit(n_splits=5)

# JAMAIS random split sur des séries temporelles !
# TOUJOURS stratifié pour classification déséquilibrée
```

### XGBoost - Configuration Champion
```python
xgb_params = {
    'objective': 'binary:logistic',  # ou 'reg:squarederror'
    'eval_metric': 'logloss',        # adapter à la métrique
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 10000,
    'early_stopping_rounds': 100,
    'tree_method': 'gpu_hist',  # GPU si disponible
    'random_state': 42,
    'n_jobs': -1,
}
```

### LightGBM - Configuration Champion
```python
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 10000,
    'early_stopping_rounds': 100,
    'verbose': -1,
    'n_jobs': -1,
}
```

### CatBoost - Configuration Champion
```python
cb_params = {
    'iterations': 10000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'min_data_in_leaf': 20,
    'random_strength': 1.0,
    'bagging_temperature': 0.5,
    'border_count': 254,
    'eval_metric': 'Logloss',
    'early_stopping_rounds': 100,
    'cat_features': cat_cols,  # CatBoost gère nativement les catégorielles
    'verbose': 100,
    'random_seed': 42,
    'task_type': 'GPU',  # si GPU disponible
}
```

### Ensembling - Stratégies Gold Medal
```python
# 1. Weighted Average (simple mais efficace)
pred_final = 0.4 * pred_xgb + 0.35 * pred_lgb + 0.25 * pred_cb

# 2. Rank Average (robuste aux différences d'échelle)
from scipy.stats import rankdata
rank_xgb = rankdata(pred_xgb) / len(pred_xgb)
rank_lgb = rankdata(pred_lgb) / len(pred_lgb)
pred_final = 0.5 * rank_xgb + 0.5 * rank_lgb

# 3. Stacking (le plus puissant)
# Train Level 1 models with OOF predictions
# Train Level 2 meta-model on OOF predictions
# L2 model: LogisticRegression, Ridge, ou un léger XGBoost

# 4. Blending (plus simple que stacking)
# Split train en train_blend et val_blend
# Train models sur train_blend, prédire val_blend
# Train meta-model sur les prédictions de val_blend
```

### Deep Learning - Patterns Clés
- **Learning rate scheduling** : Cosine annealing, ReduceLROnPlateau, OneCycleLR
- **Data augmentation** : Crucial pour généralisation (images, texte, tabulaire)
- **Dropout** : 0.1-0.5 selon la taille du modèle
- **Batch Normalization** : Stabilise l'entraînement
- **Early stopping** : Monitorer val_loss, patience 5-10 epochs
- **Mixed precision** : `tf.keras.mixed_precision` ou `torch.cuda.amp` pour accélérer
- **Gradient clipping** : `max_norm=1.0` pour stabiliser

### Transfer Learning
- **CV** : EfficientNet, ConvNeXt, ViT (Vision Transformer)
- **NLP** : DeBERTa, RoBERTa, BERT, Llama (fine-tuning)
- **Stratégie** : Freeze backbone → train head → unfreeze → fine-tune tout avec LR faible

### Métriques Courantes et Optimisation
| Métrique | Type | Optimisation |
|----------|------|-------------|
| RMSE | Régression | Optimiser MSE, post-process |
| MAE | Régression | Médiane plutôt que moyenne |
| AUC-ROC | Classification | Optimiser logloss, seuil optimal |
| F1 | Classification | Optimiser seuil avec grid search |
| Log Loss | Classification | Bien calibrer les probabilités |
| QWK | Ordinal | OptimizedRounder post-processing |
| MAP@K | Ranking | Optimiser recall puis reranker |

### Post-Processing
```python
# Optimisation de seuil pour F1
from scipy.optimize import minimize_scalar
def find_best_threshold(y_true, y_pred):
    def neg_f1(threshold):
        return -f1_score(y_true, (y_pred > threshold).astype(int))
    result = minimize_scalar(neg_f1, bounds=(0.1, 0.9), method='bounded')
    return result.x

# OptimizedRounder pour QWK
class OptimizedRounder:
    def __init__(self):
        self.coef_ = 0
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = minimize(loss_partial, initial_coef, method='nelder-mead')['x']
    def predict(self, X):
        return pd.cut(X, [-np.inf] + list(self.coef_) + [np.inf], labels=[0,1,2,3])
```

### Gestion de la Mémoire
```python
# Réduction de mémoire
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            else:
                for dtype in [np.float16, np.float32, np.float64]:
                    if c_min > np.finfo(dtype).min and c_max < np.finfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
    return df
```

### Erreurs Courantes à Éviter
1. **Data leakage** : Utiliser des infos du test dans le train (target encoding sans CV, scaling avant split)
2. **Mauvaise validation** : Random split sur données temporelles ou groupées
3. **Overfit leaderboard** : Trop de soumissions basées sur le score public
4. **Ignorer le post-processing** : Le seuil par défaut (0.5) est rarement optimal
5. **Un seul modèle** : Toujours ensembler pour stabilité
6. **Ignorer les features simples** : count, nunique, mean par groupe sont souvent les plus puissants
7. **Trop de features** : Feature selection par importance ou mutual information
8. **Mauvais seed** : Fixer TOUS les seeds (random, numpy, torch, tf)

### Reproductibilité
```python
def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)
```
