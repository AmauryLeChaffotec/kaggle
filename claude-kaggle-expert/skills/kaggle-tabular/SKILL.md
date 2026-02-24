---
name: kaggle-tabular
description: Expert en compétitions Kaggle sur données tabulaires (structured data). Utiliser quand l'utilisateur travaille sur une compétition avec des données en table (CSV, colonnes numériques et catégorielles).
argument-hint: <nom_competition ou stratégie>
---

# Expert Compétitions Tabulaires - Kaggle Gold Medal

Tu es un expert des compétitions Kaggle tabulaires. Les données tabulaires représentent la majorité des compétitions et nécessitent une approche méthodique.

## Stratégie Globale Tabulaire

### Les 3 piliers des solutions Gold Medal tabulaires :
1. **Feature Engineering massif** (60% du succès)
2. **Modèles d'arbres gradient-boosted** (30%)
3. **Ensembling intelligent** (10%)

## Stack Technique Tabulaire

### Modèles Principaux (par ordre de priorité)
1. **LightGBM** : rapide, efficace, excellent par défaut
2. **XGBoost** : légèrement différent de LightGBM → bon pour l'ensemble
3. **CatBoost** : excellent pour les features catégorielles natives
4. **Neural Networks tabulaires** : TabNet, NODE, FT-Transformer (diversité)
5. **Modèles linéaires** : Ridge, Lasso, ElasticNet (diversité)

### Configuration Optimale LightGBM (Tabulaire)

```python
# Classification
lgb_params_clf = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'max_depth': -1,
    'n_estimators': 10000,
    'verbose': -1,
    'n_jobs': -1,
}

# Régression
lgb_params_reg = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'n_estimators': 10000,
    'verbose': -1,
    'n_jobs': -1,
}
```

## Feature Engineering Tabulaire Avancé

### 1. Agrégations Groupées (Le plus puissant)

```python
def create_agg_features(df, group_col, agg_col):
    """Feature engineering le plus efficace pour le tabulaire."""
    aggs = df.groupby(group_col)[agg_col].agg(['mean', 'std', 'min', 'max', 'median', 'count'])
    aggs.columns = [f'{group_col}_{agg_col}_{stat}' for stat in aggs.columns]

    df = df.merge(aggs, on=group_col, how='left')

    # Différence par rapport au groupe
    df[f'{group_col}_{agg_col}_diff'] = df[agg_col] - df[f'{group_col}_{agg_col}_mean']

    # Ratio par rapport au groupe
    df[f'{group_col}_{agg_col}_ratio'] = df[agg_col] / (df[f'{group_col}_{agg_col}_mean'] + 1e-8)

    # Percentile dans le groupe
    df[f'{group_col}_{agg_col}_rank'] = df.groupby(group_col)[agg_col].rank(pct=True)

    return df
```

### 2. Features d'Interaction

```python
def create_interactions(df, cols, max_pairs=20):
    """Créer des interactions entre les features les plus importantes."""
    from itertools import combinations

    for i, (c1, c2) in enumerate(combinations(cols, 2)):
        if i >= max_pairs:
            break
        df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
        df[f'{c1}_div_{c2}'] = df[c1] / (df[c2] + 1e-8)
        df[f'{c1}_plus_{c2}'] = df[c1] + df[c2]
        df[f'{c1}_minus_{c2}'] = df[c1] - df[c2]

    return df
```

### 3. Encoding Catégoriel Avancé

```python
def advanced_categorical_encoding(train, test, cat_cols, target_col):
    """Encodages catégoriels complets."""

    for col in cat_cols:
        # 1. Frequency encoding
        freq = train[col].value_counts(normalize=True)
        train[f'{col}_freq'] = train[col].map(freq)
        test[f'{col}_freq'] = test[col].map(freq).fillna(0)

        # 2. Count encoding
        count = train[col].value_counts()
        train[f'{col}_count'] = train[col].map(count)
        test[f'{col}_count'] = test[col].map(count).fillna(0)

        # 3. Target encoding (CV-safe)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train[f'{col}_target'] = np.nan
        global_mean = train[target_col].mean()

        for tr_idx, val_idx in kf.split(train):
            means = train.iloc[tr_idx].groupby(col)[target_col].mean()
            train.iloc[val_idx, train.columns.get_loc(f'{col}_target')] = \
                train.iloc[val_idx][col].map(means)

        means_full = train.groupby(col)[target_col].mean()
        test[f'{col}_target'] = test[col].map(means_full)

        train[f'{col}_target'].fillna(global_mean, inplace=True)
        test[f'{col}_target'].fillna(global_mean, inplace=True)

        # 4. Target encoding multi-statistiques
        for stat in ['std', 'median', 'min', 'max']:
            stat_map = train.groupby(col)[target_col].agg(stat)
            train[f'{col}_target_{stat}'] = train[col].map(stat_map)
            test[f'{col}_target_{stat}'] = test[col].map(stat_map)

    return train, test
```

### 4. Features de Null Pattern

```python
def null_features(df, cols):
    """Les patterns de valeurs manquantes sont souvent informatifs."""
    df['n_nulls'] = df[cols].isnull().sum(axis=1)
    df['pct_nulls'] = df['n_nulls'] / len(cols)

    # Quelles colonnes sont nulles (bitwise pattern)
    for col in cols:
        df[f'{col}_is_null'] = df[col].isnull().astype(int)

    return df
```

### 5. Gestion des Outliers

```python
def handle_outliers(df, cols, method='clip', threshold=3):
    """Gérer les outliers sans les supprimer."""
    for col in cols:
        if method == 'clip':
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[f'{col}_clipped'] = df[col].clip(q1, q99)
        elif method == 'flag':
            mean, std = df[col].mean(), df[col].std()
            df[f'{col}_is_outlier'] = (np.abs(df[col] - mean) > threshold * std).astype(int)
        elif method == 'log':
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

    return df
```

## Validation Tabulaire

### Choisir la bonne stratégie de validation

```python
def get_cv_strategy(train, target_col, group_col=None, time_col=None):
    """Choisir automatiquement la meilleure stratégie CV."""

    if time_col:
        # Données temporelles → TimeSeriesSplit ou custom
        from sklearn.model_selection import TimeSeriesSplit
        print("TimeSeriesSplit recommandé (données temporelles)")
        return TimeSeriesSplit(n_splits=5)

    elif group_col:
        # Données groupées → GroupKFold
        from sklearn.model_selection import GroupKFold
        print(f"GroupKFold recommandé (groupes: {group_col})")
        return GroupKFold(n_splits=5)

    elif train[target_col].nunique() <= 20:
        # Classification → StratifiedKFold
        from sklearn.model_selection import StratifiedKFold
        print("StratifiedKFold recommandé (classification)")
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    else:
        # Régression → KFold ou StratifiedKFold sur bins
        from sklearn.model_selection import KFold
        print("KFold recommandé (régression)")
        return KFold(n_splits=5, shuffle=True, random_state=42)
```

## Adversarial Validation

```python
def adversarial_validation(train, test, features):
    """Détecter le drift train/test."""

    df = pd.concat([
        train[features].assign(is_test=0),
        test[features].assign(is_test=1)
    ], ignore_index=True)

    from sklearn.model_selection import cross_val_score
    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    scores = cross_val_score(model, df[features], df['is_test'],
                            cv=5, scoring='roc_auc')

    auc = scores.mean()
    print(f"Adversarial Validation AUC: {auc:.4f}")

    if auc > 0.7:
        print("⚠ DRIFT significatif train/test détecté !")
        print("Actions recommandées :")
        print("  - Identifier les features responsables du drift")
        print("  - Considérer un downsampling du train")
        print("  - Pondérer les samples proches du test")
    else:
        print("Distributions train/test similaires.")

    return auc
```

## Pseudo-Labeling (Semi-supervisé)

```python
def pseudo_labeling(train, test, features, target_col, model, threshold=0.95):
    """Ajouter des pseudo-labels au train pour augmenter les données."""

    # Prédictions sur le test
    if hasattr(model, 'predict_proba'):
        test_proba = model.predict_proba(test[features])[:, 1]
        confident_mask = (test_proba > threshold) | (test_proba < (1 - threshold))
        pseudo_labels = (test_proba > 0.5).astype(int)
    else:
        test_pred = model.predict(test[features])
        confident_mask = np.ones(len(test), dtype=bool)  # Adapter selon le cas
        pseudo_labels = test_pred

    pseudo_df = test[confident_mask].copy()
    pseudo_df[target_col] = pseudo_labels[confident_mask]

    print(f"Pseudo-labels ajoutés: {len(pseudo_df)} samples ({len(pseudo_df)/len(test)*100:.1f}%)")

    augmented_train = pd.concat([train, pseudo_df[train.columns]], ignore_index=True)
    return augmented_train
```

## Checklist Compétition Tabulaire

1. EDA complète avec `/kaggle-eda`
2. Feature engineering avec `/kaggle-feature`
3. Adversarial validation pour détecter le drift
4. Baseline LightGBM avec 5-fold CV
5. Feature engineering itératif (agrégations, interactions)
6. LightGBM + XGBoost + CatBoost séparément
7. Optuna pour chaque modèle
8. Ensemble weighted average ou stacking
9. Multi-seed averaging
10. Post-processing adapté à la métrique
11. Soumission finale avec `/kaggle-submit`

Adapte TOUJOURS au contexte spécifique de la compétition.
