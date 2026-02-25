---
name: kaggle-baseline
description: Crée un baseline ultra rapide (<30 min) pour une compétition Kaggle. Utiliser quand l'utilisateur commence une compétition et veut une première soumission de référence avant toute optimisation.
argument-hint: <nom_competition ou chemin des données>
---

# Quick Baseline Expert - Kaggle Gold Medal

Tu es un expert Kaggle. Ton rôle : produire un baseline fonctionnel END-TO-END en moins de 30 minutes. Pas d'optimisation, pas de fancy features — juste un score de référence fiable et une première soumission.

## Philosophie

- **Un mauvais baseline en 30 min > un bon modèle en 3 jours** : le baseline calibre tout le reste
- **Le baseline EST ta boussole** : chaque future itération se mesure contre lui
- **Simple = reproductible** : minimal preprocessing, params par défaut, pas de tuning
- **Soumettre VITE** : valider le pipeline end-to-end (format, IDs, valeurs attendues)

## Workflow Baseline (7 étapes, 30 min max)

### Étape 1 : Chargement et Diagnostic Express (5 min)

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# === CHARGEMENT ===
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')

# === DIAGNOSTIC EXPRESS ===
print(f"Train: {train.shape} | Test: {test.shape} | Sub: {sample_sub.shape}")
print(f"\nTarget: '{sample_sub.columns[-1]}'")
print(f"ID col: '{sample_sub.columns[0]}'")

TARGET = sample_sub.columns[-1]  # ADAPTER si besoin
ID_COL = sample_sub.columns[0]

# Détecter le type de problème
if TARGET in train.columns:
    n_unique = train[TARGET].nunique()
    if n_unique == 2:
        TASK = 'binary'
    elif n_unique <= 20:
        TASK = 'multiclass'
    else:
        TASK = 'regression'
    print(f"\nTask: {TASK} ({n_unique} unique target values)")
    print(f"Target distribution:\n{train[TARGET].value_counts(normalize=True).head(10)}")
else:
    print(f"⚠ Target '{TARGET}' not in train columns!")

print(f"\nMissing: train={train.isnull().sum().sum()}, test={test.isnull().sum().sum()}")
print(f"Dtypes: {dict(train.dtypes.value_counts())}")
```

### Étape 2 : Preprocessing Minimal (5 min)

```python
from sklearn.preprocessing import LabelEncoder

# Identifier les colonnes
num_cols = train.select_dtypes(include=[np.number]).columns.drop(
    [TARGET, ID_COL], errors='ignore'
).tolist()
cat_cols = train.select_dtypes(include=['object', 'category']).columns.drop(
    [ID_COL], errors='ignore'
).tolist()

print(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

# === PREPROCESSING MINIMAL ===
def baseline_preprocess(train, test, num_cols, cat_cols):
    """Preprocessing le plus simple possible."""
    # Numériques : fillna médiane
    for col in num_cols:
        median = train[col].median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)

    # Catégorielles : fillna 'missing' + LabelEncoder
    for col in cat_cols:
        train[col] = train[col].fillna('__missing__')
        test[col] = test[col].fillna('__missing__')

        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]])
        le.fit(combined)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    return train, test

train, test = baseline_preprocess(train, test, num_cols, cat_cols)
features = num_cols + cat_cols
print(f"Features for baseline: {len(features)}")
```

### Étape 3 : Split et Validation (3 min)

```python
from sklearn.model_selection import StratifiedKFold, KFold

N_FOLDS = 5
SEED = 42

if TASK in ['binary', 'multiclass']:
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    split_target = train[TARGET]
else:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    split_target = train[TARGET]
```

### Étape 4 : Modèle Baseline (10 min)

```python
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error
)

# === PARAMS PAR DÉFAUT — NE PAS TUNER ===
if TASK == 'binary':
    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting_type': 'gbdt', 'num_leaves': 31,
        'learning_rate': 0.05, 'n_estimators': 1000,
        'verbose': -1, 'random_state': SEED, 'n_jobs': -1,
    }
elif TASK == 'multiclass':
    params = {
        'objective': 'multiclass', 'num_class': train[TARGET].nunique(),
        'metric': 'multi_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 1000,
        'verbose': -1, 'random_state': SEED, 'n_jobs': -1,
    }
else:  # regression
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'boosting_type': 'gbdt', 'num_leaves': 31,
        'learning_rate': 0.05, 'n_estimators': 1000,
        'verbose': -1, 'random_state': SEED, 'n_jobs': -1,
    }

# === TRAIN OOF ===
oof_preds = np.zeros(len(train)) if TASK != 'multiclass' else np.zeros((len(train), train[TARGET].nunique()))
test_preds = np.zeros(len(test)) if TASK != 'multiclass' else np.zeros((len(test), train[TARGET].nunique()))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], split_target)):
    X_tr = train.iloc[train_idx][features]
    X_val = train.iloc[val_idx][features]
    y_tr = train.iloc[train_idx][TARGET]
    y_val = train.iloc[val_idx][TARGET]

    model = lgb.LGBMClassifier(**params) if TASK != 'regression' else lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    if TASK == 'binary':
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(test[features])[:, 1] / N_FOLDS
        score = roc_auc_score(y_val, oof_preds[val_idx])
    elif TASK == 'multiclass':
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(test[features]) / N_FOLDS
        score = accuracy_score(y_val, oof_preds[val_idx].argmax(axis=1))
    else:
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(test[features]) / N_FOLDS
        score = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))

    fold_scores.append(score)
    print(f"Fold {fold}: {score:.5f}")

print(f"\n{'='*50}")
print(f"BASELINE CV: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
print(f"{'='*50}")
```

### Étape 5 : Feature Importance Rapide (2 min)

```python
imp = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Features:")
print(imp.head(10).to_string(index=False))

print(f"\nZero importance ({(imp['importance'] == 0).sum()} features):")
print(imp[imp['importance'] == 0]['feature'].tolist())
```

### Étape 6 : Soumission (3 min)

```python
# === CRÉER LA SOUMISSION ===
submission = sample_sub.copy()

if TASK == 'binary':
    # Vérifier si la soumission attend des probas ou des classes
    sub_example = sample_sub[TARGET].iloc[0]
    if isinstance(sub_example, (int, np.integer)) or sub_example in [0, 1, '0', '1']:
        submission[TARGET] = (test_preds > 0.5).astype(int)
        print("Submission: binary classes (0/1)")
    else:
        submission[TARGET] = test_preds
        print("Submission: probabilities")
elif TASK == 'multiclass':
    submission[TARGET] = test_preds.argmax(axis=1)
    print("Submission: class labels")
else:
    submission[TARGET] = test_preds
    print("Submission: continuous values")

# Validation
assert submission.shape[0] == sample_sub.shape[0], "Wrong number of rows!"
assert list(submission.columns) == list(sample_sub.columns), "Wrong columns!"
assert submission[TARGET].isnull().sum() == 0, "NaN in predictions!"

submission.to_csv('submissions/baseline_submission.csv', index=False)
print(f"\n✓ Submission saved: {submission.shape}")
print(submission.head())
print(f"\nPrediction stats: mean={submission[TARGET].mean():.4f}, "
      f"std={submission[TARGET].std():.4f}")
```

### Étape 7 : Rapport Baseline (2 min)

```python
print(f"""
{'='*60}
BASELINE REPORT
{'='*60}
Competition: [NOM]
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

DATA
  Train: {train.shape[0]:,} rows × {train.shape[1]} cols
  Test: {test.shape[0]:,} rows
  Features used: {len(features)} ({len(num_cols)} num + {len(cat_cols)} cat)
  Missing strategy: median (num) + 'missing' (cat)

MODEL
  Algorithm: LightGBM (default params)
  CV: {N_FOLDS}-fold {'Stratified' if TASK != 'regression' else ''}KFold

RESULTS
  CV Score: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}
  Fold scores: {[f'{s:.5f}' for s in fold_scores]}
  LB Score: [À REMPLIR APRÈS SOUMISSION]

TOP FEATURES
{imp.head(5).to_string(index=False)}

NEXT STEPS
  1. Soumettre et noter le LB score
  2. Vérifier CV-LB gap (attendu < 3%)
  3. /kaggle-eda pour exploration approfondie
  4. /kaggle-feature pour améliorer les features
{'='*60}
""")
```

## Definition of Done (DoD)

Le baseline est COMPLET quand :

- [ ] Pipeline end-to-end fonctionnel (load → preprocess → train → predict → submit)
- [ ] CV score calculé avec OOF propre (5-fold minimum)
- [ ] Score par fold affiché (vérifier la stabilité)
- [ ] Feature importance extraite (identifier les top features + les inutiles)
- [ ] Fichier de soumission créé et validé (format, shape, pas de NaN)
- [ ] Soumission uploadée sur Kaggle
- [ ] LB score noté et comparé au CV
- [ ] Rapport baseline documenté
- [ ] Temps total < 30 minutes

## Règles d'Or

1. **NE PAS TUNER** : params par défaut, c'est le but
2. **NE PAS feature-engineer** : preprocessing minimal seulement
3. **SOUMETTRE** : la soumission valide que tout le pipeline fonctionne
4. **NOTER** le CV et le LB : c'est le point de référence pour tout le reste
5. **30 minutes max** : si ça prend plus longtemps, tu over-engineer le baseline

## Rapport de Sortie (OBLIGATOIRE)

À la fin du baseline, TOUJOURS sauvegarder :
1. Rapport dans : `reports/baseline/YYYY-MM-DD_baseline.md` (CV, LB, features, params)
2. OOF predictions dans : `artifacts/oof_baseline_v1.parquet`
3. Test predictions dans : `artifacts/test_baseline_v1.parquet`
4. Submission dans : `submissions/sub_baseline_YYYY-MM-DD.csv`
5. Ajouter une ligne dans `runs.csv`
6. Confirmer à l'utilisateur les chemins sauvegardés
