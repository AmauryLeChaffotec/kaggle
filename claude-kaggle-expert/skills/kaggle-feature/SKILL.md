---
name: kaggle-feature
description: Effectue du feature engineering avancé pour une compétition Kaggle. Utiliser quand l'utilisateur veut créer des features, transformer des variables, ou améliorer son dataset.
argument-hint: <type_de_features ou description>
---

# Feature Engineering Expert - Kaggle Gold Medal

Tu es un expert en feature engineering. Crée des features puissantes et pertinentes pour maximiser la performance des modèles.

## Principes de Feature Engineering

1. **Commencer simple** : count, mean, std par groupe avant les features complexes
2. **Mutual Information** : quantifier la relation feature-target AVANT de créer des features
3. **Éviter le leakage** : JAMAIS utiliser d'info du test ou de la target dans les features
4. **Valider chaque feature** : vérifier l'apport réel avec importance ou CV score
5. **Supprimer le bruit** : une feature inutile peut dégrader le modèle

## Feature Engineering par Type de Données

### 1. Features Numériques

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def create_numerical_features(df, num_cols):
    """Créer des features à partir de colonnes numériques."""

    # Transformations mathématiques
    for col in num_cols:
        df[f'{col}_log1p'] = np.log1p(df[col].clip(lower=0))
        df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_is_zero'] = (df[col] == 0).astype(int)

    # Interactions entre paires de features
    from itertools import combinations
    for col1, col2 in combinations(num_cols[:10], 2):  # limiter aux top features
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

    # Statistiques par ligne
    df['num_mean'] = df[num_cols].mean(axis=1)
    df['num_std'] = df[num_cols].std(axis=1)
    df['num_max'] = df[num_cols].max(axis=1)
    df['num_min'] = df[num_cols].min(axis=1)
    df['num_range'] = df['num_max'] - df['num_min']
    df['num_skew'] = df[num_cols].skew(axis=1)
    df['num_n_zeros'] = (df[num_cols] == 0).sum(axis=1)
    df['num_n_nulls'] = df[num_cols].isnull().sum(axis=1)

    return df
```

### 2. Features Catégorielles

```python
def create_categorical_features(df, cat_cols, target_col=None):
    """Créer des features à partir de colonnes catégorielles."""

    # Frequency encoding
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

    # Count encoding
    for col in cat_cols:
        count = df[col].value_counts()
        df[f'{col}_count'] = df[col].map(count)

    # Label encoding (pour les modèles basés sur les arbres)
    from sklearn.preprocessing import LabelEncoder
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_label'] = le.fit_transform(df[col].astype(str))

    # Interactions catégorielles
    from itertools import combinations
    for col1, col2 in combinations(cat_cols[:5], 2):
        df[f'{col1}_{col2}_combo'] = df[col1].astype(str) + '_' + df[col2].astype(str)
        freq = df[f'{col1}_{col2}_combo'].value_counts(normalize=True)
        df[f'{col1}_{col2}_combo_freq'] = df[f'{col1}_{col2}_combo'].map(freq)

    return df
```

### 3. Target Encoding (avec protection contre le leakage)

```python
def target_encode_cv(train, test, cat_cols, target_col, n_splits=5):
    """Target encoding sécurisé avec cross-validation."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cat_cols:
        train[f'{col}_target_enc'] = np.nan
        global_mean = train[target_col].mean()

        for train_idx, val_idx in kf.split(train):
            # Calculer la moyenne sur le fold d'entraînement
            means = train.iloc[train_idx].groupby(col)[target_col].mean()
            # Appliquer sur le fold de validation
            train.iloc[val_idx, train.columns.get_loc(f'{col}_target_enc')] = \
                train.iloc[val_idx][col].map(means)

        # Pour le test, utiliser tout le train
        means = train.groupby(col)[target_col].mean()
        test[f'{col}_target_enc'] = test[col].map(means)

        # Remplir les NaN avec la moyenne globale
        train[f'{col}_target_enc'].fillna(global_mean, inplace=True)
        test[f'{col}_target_enc'].fillna(global_mean, inplace=True)

    return train, test
```

### 4. Features Temporelles

```python
def create_datetime_features(df, date_col):
    """Créer des features à partir d'une colonne datetime."""
    df[date_col] = pd.to_datetime(df[date_col])

    # Features de base
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    df[f'{date_col}_hour'] = df[date_col].dt.hour
    df[f'{date_col}_minute'] = df[date_col].dt.minute

    # Features avancées
    df[f'{date_col}_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    df[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df[f'{date_col}_quarter'] = df[date_col].dt.quarter
    df[f'{date_col}_dayofyear'] = df[date_col].dt.dayofyear
    df[f'{date_col}_weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)

    # Features cycliques (pour capturer la nature cyclique)
    df[f'{date_col}_month_sin'] = np.sin(2 * np.pi * df[f'{date_col}_month'] / 12)
    df[f'{date_col}_month_cos'] = np.cos(2 * np.pi * df[f'{date_col}_month'] / 12)
    df[f'{date_col}_dow_sin'] = np.sin(2 * np.pi * df[f'{date_col}_dayofweek'] / 7)
    df[f'{date_col}_dow_cos'] = np.cos(2 * np.pi * df[f'{date_col}_dayofweek'] / 7)

    # Jours depuis une date de référence
    ref_date = df[date_col].min()
    df[f'{date_col}_days_since_start'] = (df[date_col] - ref_date).dt.days

    return df
```

### 5. Features de Lag et Rolling (Séries Temporelles)

```python
def create_lag_features(df, value_col, group_col=None, lags=[1,7,14,28]):
    """Créer des features de lag et rolling."""

    for lag in lags:
        if group_col:
            df[f'{value_col}_lag_{lag}'] = df.groupby(group_col)[value_col].shift(lag)
        else:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        if group_col:
            rolled = df.groupby(group_col)[value_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1)
            )
        else:
            rolled = df[value_col].shift(1).rolling(window, min_periods=1)

        df[f'{value_col}_rolling_mean_{window}'] = rolled.mean()
        df[f'{value_col}_rolling_std_{window}'] = rolled.std()
        df[f'{value_col}_rolling_min_{window}'] = rolled.min()
        df[f'{value_col}_rolling_max_{window}'] = rolled.max()

    # Expanding statistics
    if group_col:
        expanding = df.groupby(group_col)[value_col].transform(
            lambda x: x.shift(1).expanding(min_periods=1)
        )
    else:
        expanding = df[value_col].shift(1).expanding(min_periods=1)

    df[f'{value_col}_expanding_mean'] = expanding.mean()

    return df
```

### 6. Features de Clustering

```python
def create_cluster_features(df, num_cols, n_clusters=[3, 5, 8]):
    """Utiliser le clustering comme feature engineering."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols].fillna(0))

    for k in n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df[f'cluster_{k}'] = kmeans.fit_predict(X_scaled)

        # Distance au centroïde le plus proche
        distances = kmeans.transform(X_scaled)
        df[f'cluster_{k}_dist_min'] = distances.min(axis=1)
        df[f'cluster_{k}_dist_mean'] = distances.mean(axis=1)

    return df
```

### 7. Features PCA

```python
def create_pca_features(df, num_cols, n_components=5):
    """Réduction de dimensionnalité avec PCA."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols].fillna(0))

    pca = PCA(n_components=min(n_components, len(num_cols)))
    pca_features = pca.fit_transform(X_scaled)

    for i in range(pca_features.shape[1]):
        df[f'pca_{i}'] = pca_features[:, i]

    print(f"Variance expliquée: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

    return df
```

### 8. Feature Selection

```python
def select_features(X, y, method='mutual_info', top_k=50):
    """Sélectionner les features les plus pertinentes."""

    if method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        if y.nunique() <= 20:
            mi = mutual_info_classif(X.fillna(0), y, random_state=42)
        else:
            mi = mutual_info_regression(X.fillna(0), y, random_state=42)

        mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        selected = mi_scores.head(top_k).index.tolist()

    elif method == 'importance':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X.fillna(0), y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        selected = imp.sort_values(ascending=False).head(top_k).index.tolist()

    elif method == 'correlation':
        # Supprimer les features corrélées entre elles (>0.95)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        selected = [col for col in X.columns if col not in to_drop]

    return selected
```

### 9. Agrégations Groupées (très puissant)

```python
def create_group_aggregations(df, group_cols, num_cols, agg_funcs=['mean','std','min','max','median']):
    """Créer des agrégations par groupe - souvent le feature engineering le plus efficace."""

    for group_col in group_cols:
        for num_col in num_cols:
            for func in agg_funcs:
                col_name = f'{group_col}_{num_col}_{func}'
                df[col_name] = df.groupby(group_col)[num_col].transform(func)

            # Différence par rapport à la moyenne du groupe
            df[f'{group_col}_{num_col}_diff_mean'] = \
                df[num_col] - df.groupby(group_col)[num_col].transform('mean')

            # Ratio par rapport à la moyenne du groupe
            group_mean = df.groupby(group_col)[num_col].transform('mean')
            df[f'{group_col}_{num_col}_ratio_mean'] = df[num_col] / (group_mean + 1e-8)

            # Rang dans le groupe
            df[f'{group_col}_{num_col}_rank'] = \
                df.groupby(group_col)[num_col].rank(pct=True)

    return df
```

## Workflow Recommandé

1. **Identifier les types** : numériques, catégorielles, temporelles, texte
2. **Mutual Information** : évaluer la pertinence de chaque feature brute
3. **Features simples d'abord** : agrégations, counts, frequency encoding
4. **Interactions** : entre les features les plus importantes (top 10-15)
5. **Features domaine** : spécifiques au problème (ratios métier, etc.)
6. **Feature selection** : supprimer les features à faible importance
7. **Valider** : comparer le score CV avec et sans les nouvelles features

Adapte TOUJOURS les features au contexte spécifique de la compétition et aux données de l'utilisateur.
