---
name: kaggle-pipeline
description: Crée un pipeline complet de A à Z pour une compétition Kaggle. Utiliser quand l'utilisateur commence une nouvelle compétition ou veut un workflow end-to-end.
argument-hint: <nom_competition ou description>
---

# Pipeline Complet Compétition Kaggle - Gold Medal Strategy

Tu es un Kaggle Grandmaster. Crée un pipeline complet et optimisé pour une compétition Kaggle, en suivant les meilleures pratiques des solutions gagnantes.

## Phase 1 : Analyse de la Compétition

Avant d'écrire du code, ANALYSE :
1. **Quel est le problème ?** Classification binaire, multiclasse, régression, ranking, segmentation...
2. **Quelle est la métrique ?** AUC, F1, RMSE, MAE, QWK, MAP@K, Dice...
3. **Quelle est la taille des données ?** Petit (<10K), moyen (10K-1M), grand (>1M)
4. **Quel type de données ?** Tabulaire, images, texte, séries temporelles, audio, graphe
5. **Y a-t-il du data leakage ?** Vérifier les colonnes suspectes
6. **Quelles sont les contraintes ?** Temps d'inférence, GPU, format de soumission

## Phase 2 : Structure du Projet

```
competition_name/
├── data/
│   ├── raw/           # Données brutes
│   ├── processed/     # Données transformées
│   └── external/      # Données externes
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_ensemble.ipynb
├── src/
│   ├── data.py        # Chargement et preprocessing
│   ├── features.py    # Feature engineering
│   ├── models.py      # Modèles
│   ├── train.py       # Training loop
│   └── utils.py       # Utilitaires
├── submissions/       # Fichiers de soumission
├── models/            # Modèles sauvegardés
└── config.yaml        # Configuration
```

## Phase 3 : Template de Notebook Complet

```python
# ============================================================
# CONFIGURATION
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from scipy.stats import rankdata
import warnings
import gc
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# Reproductibilité
SEED = 42
N_FOLDS = 5
TARGET = 'target'  # ADAPTER
METRIC = 'auc'     # ADAPTER

def seed_everything(seed=SEED):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
sample_sub = pd.read_csv('data/raw/sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target distribution:\n{train[TARGET].value_counts(normalize=True)}")

# ============================================================
# EDA RAPIDE
# ============================================================
# Voir skill /kaggle-eda pour EDA complète

# Missing values
print("\nMissing values (train):")
missing = train.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# Types
print(f"\nNum cols: {len(train.select_dtypes(include=[np.number]).columns)}")
print(f"Cat cols: {len(train.select_dtypes(include=['object']).columns)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# Identifier les types de colonnes
id_col = train.columns[0]  # ADAPTER
num_cols = train.select_dtypes(include=[np.number]).columns.drop([TARGET, id_col], errors='ignore').tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

def feature_engineering(df, is_train=True):
    """Pipeline de feature engineering."""

    # --- Features numériques ---
    # Statistiques par ligne
    if len(num_cols) > 1:
        df['num_mean'] = df[num_cols].mean(axis=1)
        df['num_std'] = df[num_cols].std(axis=1)
        df['num_nulls'] = df[num_cols].isnull().sum(axis=1)

    # --- Features catégorielles ---
    for col in cat_cols:
        # Frequency encoding
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

        # Label encoding
        le = LabelEncoder()
        df[f'{col}_le'] = le.fit_transform(df[col].astype(str))

    # --- Features d'interaction ---
    # ADAPTER selon l'EDA

    return df

train = feature_engineering(train, is_train=True)
test = feature_engineering(test, is_train=False)

# Features finales
features = [c for c in train.columns if c not in [id_col, TARGET] + cat_cols]
print(f"\nNombre de features: {len(features)}")

# ============================================================
# MODÉLISATION
# ============================================================

# --- LightGBM ---
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
    'verbose': -1,
    'n_jobs': -1,
    'random_state': SEED,
}

# Train LightGBM
oof_lgb = np.zeros(len(train))
test_lgb = np.zeros(len(test))
feature_importance = pd.DataFrame()

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train[TARGET])):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    X_tr, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
    y_tr, y_val = train.iloc[train_idx][TARGET], train.iloc[val_idx][TARGET]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )

    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_lgb += model.predict_proba(test[features])[:, 1] / N_FOLDS

    imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_, 'fold': fold})
    feature_importance = pd.concat([feature_importance, imp])

lgb_score = roc_auc_score(train[TARGET], oof_lgb)
print(f"\nLightGBM CV AUC: {lgb_score:.6f}")

# --- XGBoost ---
# (Même pattern, ADAPTER params)

# --- CatBoost ---
# (Même pattern, ADAPTER params)

# ============================================================
# ENSEMBLE
# ============================================================
# Weighted average
final_preds = test_lgb  # Ajouter d'autres modèles

# ============================================================
# SOUMISSION
# ============================================================
submission = pd.DataFrame({
    sample_sub.columns[0]: test[id_col],
    sample_sub.columns[1]: final_preds
})

submission.to_csv('submissions/submission.csv', index=False)
print(f"\nSoumission créée: {submission.shape}")
print(submission.head())
```

## Phase 4 : Itération

L'itération est la clé du succès. Après chaque soumission :

1. **Comparer CV vs LB** : ratio attendu ~0.98-1.02
2. **Analyser les erreurs** : quelles observations sont mal prédites ?
3. **Ajouter des features** : basées sur l'analyse d'erreur
4. **Diversifier les modèles** : arbres + linéaires + NN
5. **Optimiser l'ensemble** : trouver les poids optimaux
6. **Post-processing** : adapter les seuils/arrondi

## Conseils Gold Medal

- **Semaine 1-2** : EDA, comprendre les données, baseline
- **Semaine 2-3** : Feature engineering itératif
- **Semaine 3-4** : Modèles multiples, hyperparameter tuning
- **Dernière semaine** : Ensembling, sélection finale
- **Dernier jour** : NE PAS changer de stratégie, confiance dans le CV

Adapte TOUJOURS ce template aux spécificités de la compétition.

## Definition of Done (DoD)

Le pipeline est COMPLET quand :

- [ ] Structure de dossiers créée (data/, notebooks/, src/, submissions/, models/)
- [ ] Problème analysé (type, métrique, taille, contraintes)
- [ ] Baseline fonctionnel avec première soumission
- [ ] Stratégie de CV définie et validée (CV-LB gap < 3%)
- [ ] Experiment tracker initialisé
- [ ] Au moins 3 itérations documentées (baseline → features → tuning)
- [ ] Ensemble de 2+ modèles testé
- [ ] 2 soumissions finales sélectionnées avec justification

## Rapport de Sortie (OBLIGATOIRE)

À la fin de la création du pipeline, TOUJOURS sauvegarder :
1. Rapport dans : `reports/pipeline/YYYY-MM-DD_init.md` (structure créée, type de problème, métrique, plan initial)
2. Vérifier que tous les dossiers existent : data/, notebooks/, src/, models/, artifacts/, reports/, submissions/, configs/
3. Confirmer à l'utilisateur la structure créée
