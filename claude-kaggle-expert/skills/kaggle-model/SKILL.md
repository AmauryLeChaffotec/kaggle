---
name: kaggle-model
description: Construit, entraîne et optimise des modèles de machine learning pour compétitions Kaggle. Utiliser quand l'utilisateur veut créer un modèle, entraîner, optimiser les hyperparamètres, ou faire de l'ensembling.
argument-hint: <type_de_modèle ou stratégie>
---

# Modélisation Expert - Kaggle Gold Medal

Tu es un expert en modélisation ML/DL. Construis des modèles performants en utilisant les meilleures pratiques des compétitions Kaggle.

## Framework de Modélisation Complet

### 1. Baseline Rapide

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import *
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

def quick_baseline(train, test, target_col, task='classification'):
    """Baseline rapide avec LightGBM pour établir un score de référence."""
    features = [c for c in train.columns if c != target_col]

    if task == 'classification':
        model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.1, verbose=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'roc_auc'
    else:
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.1, verbose=-1)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'neg_root_mean_squared_error'

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, train[features], train[target_col],
                            cv=cv, scoring=scoring, n_jobs=-1)

    print(f"Baseline CV Score: {scores.mean():.6f} (+/- {scores.std():.6f})")
    return scores
```

### 2. Training avec Cross-Validation (Pattern Gold Medal)

```python
def train_lgb_cv(train, test, features, target_col, params, n_splits=5,
                 task='classification', group_col=None):
    """Entraînement LightGBM avec CV et prédictions OOF + test."""

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    feature_importance = pd.DataFrame()
    scores = []

    if group_col:
        from sklearn.model_selection import GroupKFold
        kf = GroupKFold(n_splits=n_splits)
        split_args = (train[features], train[target_col], train[group_col])
    elif task == 'classification':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_args = (train[features], train[target_col])
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_args = (train[features],)

    for fold, (train_idx, val_idx) in enumerate(kf.split(*split_args)):
        print(f"\n{'='*40} Fold {fold+1}/{n_splits} {'='*40}")

        X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_train, y_val = train.iloc[train_idx][target_col], train.iloc[val_idx][target_col]

        model = lgb.LGBMRegressor(**params) if task == 'regression' \
                else lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=True),
                lgb.log_evaluation(200)
            ]
        )

        if task == 'classification':
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(test[features])[:, 1] / n_splits
        else:
            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(test[features]) / n_splits

        # Feature importance
        imp = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_,
            'fold': fold
        })
        feature_importance = pd.concat([feature_importance, imp])

        # Score du fold
        if task == 'classification':
            fold_score = roc_auc_score(y_val, oof_preds[val_idx])
        else:
            fold_score = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        scores.append(fold_score)
        print(f"Fold {fold+1} Score: {fold_score:.6f}")

    # Score global
    if task == 'classification':
        overall_score = roc_auc_score(train[target_col], oof_preds)
    else:
        overall_score = np.sqrt(mean_squared_error(train[target_col], oof_preds))

    print(f"\n{'='*60}")
    print(f"Overall CV Score: {overall_score:.6f}")
    print(f"Mean Fold Score: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")

    return oof_preds, test_preds, feature_importance
```

### 3. Même pattern pour XGBoost et CatBoost

```python
def train_xgb_cv(train, test, features, target_col, params, n_splits=5, task='classification'):
    """Pattern identique pour XGBoost."""
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train[target_col])):
        X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_train, y_val = train.iloc[train_idx][target_col], train.iloc[val_idx][target_col]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params, dtrain,
            num_boost_round=10000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=200
        )

        oof_preds[val_idx] = model.predict(dval)
        test_preds += model.predict(xgb.DMatrix(test[features])) / n_splits

    return oof_preds, test_preds

def train_catboost_cv(train, test, features, target_col, cat_features, params, n_splits=5):
    """Pattern identique pour CatBoost."""
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train[target_col])):
        X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
        y_train, y_val = train.iloc[train_idx][target_col], train.iloc[val_idx][target_col]

        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            verbose=200
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(test[features])[:, 1] / n_splits

    return oof_preds, test_preds
```

### 4. Hyperparameter Tuning avec Optuna

```python
import optuna

def optimize_lgb(train, features, target_col, n_trials=100, task='classification'):
    """Optimisation des hyperparamètres avec Optuna."""

    def objective(trial):
        params = {
            'n_estimators': 10000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
        }

        if task == 'classification':
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            model = lgb.LGBMClassifier(**params)
            scoring = 'roc_auc'
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
            model = lgb.LGBMRegressor(**params)
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, train[features], train[target_col],
                                cv=cv, scoring=scoring, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nMeilleur score: {study.best_value:.6f}")
    print(f"Meilleurs paramètres: {study.best_params}")

    return study.best_params
```

### 5. Ensembling - Stratégies Gold Medal

```python
from scipy.stats import rankdata
from sklearn.linear_model import Ridge, LogisticRegression

def weighted_ensemble(predictions_dict, weights=None):
    """Ensemble pondéré simple."""
    models = list(predictions_dict.keys())
    if weights is None:
        weights = {m: 1/len(models) for m in models}

    result = sum(predictions_dict[m] * weights[m] for m in models)
    return result

def rank_ensemble(predictions_dict, weights=None):
    """Ensemble par rang (robuste aux différences d'échelle)."""
    models = list(predictions_dict.keys())
    if weights is None:
        weights = {m: 1/len(models) for m in models}

    ranked = {m: rankdata(predictions_dict[m]) / len(predictions_dict[m])
              for m in models}
    result = sum(ranked[m] * weights[m] for m in models)
    return result

def stacking(oof_dict, test_dict, y_train, task='classification'):
    """Stacking avec un méta-modèle."""
    # Construire la matrice de stacking
    oof_stack = np.column_stack([oof_dict[m] for m in oof_dict])
    test_stack = np.column_stack([test_dict[m] for m in test_dict])

    # Méta-modèle
    if task == 'classification':
        meta = LogisticRegression(C=1.0, random_state=42)
    else:
        meta = Ridge(alpha=1.0)

    # CV pour le méta-modèle
    kf = StratifiedKFold(5, shuffle=True, random_state=42) if task == 'classification' \
         else KFold(5, shuffle=True, random_state=42)

    meta_oof = np.zeros(len(y_train))
    meta_test = np.zeros(len(test_stack))

    for train_idx, val_idx in kf.split(oof_stack, y_train):
        meta.fit(oof_stack[train_idx], y_train.iloc[train_idx])
        if task == 'classification':
            meta_oof[val_idx] = meta.predict_proba(oof_stack[val_idx])[:, 1]
            meta_test += meta.predict_proba(test_stack)[:, 1] / 5
        else:
            meta_oof[val_idx] = meta.predict(oof_stack[val_idx])
            meta_test += meta.predict(test_stack) / 5

    return meta_oof, meta_test

def find_optimal_weights(oof_dict, y_train, metric_func, task='classification'):
    """Trouver les poids optimaux pour l'ensemble."""
    from scipy.optimize import minimize

    models = list(oof_dict.keys())
    preds_array = np.column_stack([oof_dict[m] for m in models])

    def objective(weights):
        weights = weights / weights.sum()  # normaliser
        blended = preds_array @ weights
        return -metric_func(y_train, blended)

    initial = np.ones(len(models)) / len(models)
    bounds = [(0, 1)] * len(models)
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

    result = minimize(objective, initial, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    optimal_weights = dict(zip(models, result.x))
    print("Poids optimaux:")
    for m, w in optimal_weights.items():
        print(f"  {m}: {w:.4f}")

    return optimal_weights
```

### 6. Feature Importance Visualization

```python
def plot_feature_importance(feature_importance_df, top_n=30):
    """Visualiser l'importance des features."""
    import matplotlib.pyplot as plt

    mean_imp = feature_importance_df.groupby('feature')['importance'].mean()
    mean_imp = mean_imp.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    mean_imp.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Mean Importance')
    plt.tight_layout()
    plt.show()

    return mean_imp
```

### 7. Multi-Seed Averaging

```python
def multi_seed_train(train, test, features, target_col, params, seeds=[42, 123, 456, 789, 2024]):
    """Entraîner avec plusieurs seeds pour plus de stabilité."""
    all_test_preds = []

    for seed in seeds:
        params_seed = params.copy()
        params_seed['random_state'] = seed

        _, test_preds, _ = train_lgb_cv(
            train, test, features, target_col, params_seed
        )
        all_test_preds.append(test_preds)

    final_preds = np.mean(all_test_preds, axis=0)
    print(f"Multi-seed averaging avec {len(seeds)} seeds terminé.")
    return final_preds
```

## Checklist Avant Soumission

1. Le score CV est-il stable entre les folds ? (std < 1% du mean)
2. As-tu vérifié la corrélation CV vs LB sur les premières soumissions ?
3. As-tu essayé au moins 3 types de modèles différents ?
4. As-tu fait de l'ensembling ?
5. As-tu optimisé les hyperparamètres ?
6. As-tu fait du multi-seed averaging ?
7. As-tu vérifié le post-processing (seuils, arrondi) ?

Adapte TOUJOURS la stratégie au type de compétition, à la métrique, et aux données.

## Definition of Done (DoD)

La modélisation est COMPLÈTE quand :

- [ ] Au moins 1 baseline + 2 modèles variants entraînés
- [ ] Chaque modèle a des OOF predictions propres (5-fold minimum)
- [ ] Tuning contrôlé avec budget défini (Optuna ou grid)
- [ ] Feature importance extraite et analysée
- [ ] Analyse d'erreurs sur les worst predictions
- [ ] Tableau récap : model | CV mean | CV std | LB score | n_features
- [ ] OOF predictions sauvegardées (.npy) pour l'ensembling
- [ ] Test predictions sauvegardées (.npy) pour la soumission
- [ ] Params et config sauvegardés (reproductibilité)
