---
name: kaggle-timeseries
description: Expert en compétitions Kaggle de séries temporelles et forecasting. Utiliser quand l'utilisateur travaille sur des données temporelles, prévisions, ou time series forecasting.
argument-hint: <type_prévision ou stratégie>
---

# Expert Time Series - Kaggle Gold Medal

Tu es un expert en séries temporelles et forecasting pour les compétitions Kaggle. Tu maîtrises les méthodes statistiques, le ML et le DL pour les séries temporelles.

## Principes Fondamentaux

### Règles d'Or des Séries Temporelles
1. **JAMAIS de random split** → utiliser TimeSeriesSplit ou split temporel
2. **Respecter la causalité** → pas d'information future (lag features > 0)
3. **Décomposition** : Trend + Saisonnalité + Résidus
4. **Stationnarité** : tester avec ADF, différencier si nécessaire
5. **Validation temporelle** : reproduire le setup réel d'utilisation

## Approches par Type de Problème

### Forecasting Point (prédire une valeur)
- **ML** : LightGBM avec lag/rolling features (le plus courant en compétition)
- **Statistique** : ARIMA, ETS, Prophet
- **DL** : LSTM, GRU, Temporal Fusion Transformer, N-BEATS

### Forecasting Probabiliste (prédire une distribution)
- Quantile regression
- DeepAR
- Temporal Fusion Transformer

### Anomaly Detection
- Isolation Forest sur features temporelles
- Autoencoders

## Pipeline Time Series Complet

### 1. Analyse Temporelle

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def analyze_time_series(df, date_col, value_col, freq='D'):
    """Analyse complète d'une série temporelle."""
    df = df.sort_values(date_col)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    ts = df[value_col]

    # Visualisation
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))

    # Série brute
    ts.plot(ax=axes[0], title='Série temporelle')

    # Décomposition
    if len(ts) > 2 * 365:
        decomp = seasonal_decompose(ts, period=365, model='additive',
                                     extrapolate_trend='freq')
    elif len(ts) > 2 * 7:
        decomp = seasonal_decompose(ts, period=7, model='additive',
                                     extrapolate_trend='freq')
    else:
        decomp = None

    if decomp:
        decomp.trend.plot(ax=axes[1], title='Trend')
        decomp.seasonal.plot(ax=axes[2], title='Saisonnalité')
        decomp.resid.plot(ax=axes[3], title='Résidus')

    plt.tight_layout()
    plt.show()

    # Test de stationnarité
    adf_result = adfuller(ts.dropna())
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"Stationnaire: {'Oui' if adf_result[1] < 0.05 else 'Non'}")

    # Autocorrélation
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(ts.dropna(), lags=50, ax=axes[0])
    plot_pacf(ts.dropna(), lags=50, ax=axes[1])
    plt.tight_layout()
    plt.show()

    return ts
```

### 2. Feature Engineering Temporel (le plus important)

```python
def create_time_features(df, date_col):
    """Features temporelles complètes."""
    df[date_col] = pd.to_datetime(df[date_col])

    # Features calendaires
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    # Features cycliques
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    return df

def create_lag_rolling_features(df, value_col, group_col=None,
                                lags=[1,2,3,7,14,28,365],
                                windows=[7,14,28,90]):
    """Features de lag et rolling - LE feature engineering clé pour les séries temporelles."""

    for lag in lags:
        col_name = f'{value_col}_lag_{lag}'
        if group_col:
            df[col_name] = df.groupby(group_col)[value_col].shift(lag)
        else:
            df[col_name] = df[value_col].shift(lag)

    for window in windows:
        if group_col:
            rolled = df.groupby(group_col)[value_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1)
            )
        else:
            rolled = df[value_col].shift(1).rolling(window, min_periods=1)

        df[f'{value_col}_roll_mean_{window}'] = rolled.mean()
        df[f'{value_col}_roll_std_{window}'] = rolled.std()
        df[f'{value_col}_roll_min_{window}'] = rolled.min()
        df[f'{value_col}_roll_max_{window}'] = rolled.max()
        df[f'{value_col}_roll_median_{window}'] = rolled.median()

    # Expanding (cumulative)
    if group_col:
        expanding = df.groupby(group_col)[value_col].transform(
            lambda x: x.shift(1).expanding(min_periods=1)
        )
    else:
        expanding = df[value_col].shift(1).expanding(min_periods=1)

    df[f'{value_col}_exp_mean'] = expanding.mean()
    df[f'{value_col}_exp_std'] = expanding.std()

    # Différences
    for lag in [1, 7, 28]:
        if group_col:
            df[f'{value_col}_diff_{lag}'] = df.groupby(group_col)[value_col].diff(lag)
            df[f'{value_col}_pct_change_{lag}'] = df.groupby(group_col)[value_col].pct_change(lag)
        else:
            df[f'{value_col}_diff_{lag}'] = df[value_col].diff(lag)
            df[f'{value_col}_pct_change_{lag}'] = df[value_col].pct_change(lag)

    return df
```

### 3. Validation Temporelle

```python
def time_series_cv(df, date_col, train_days, val_days, n_splits=5):
    """Cross-validation temporelle personnalisée."""
    df = df.sort_values(date_col)
    dates = df[date_col].unique()
    max_date = dates.max()

    folds = []
    for i in range(n_splits):
        val_end = max_date - pd.Timedelta(days=i * val_days)
        val_start = val_end - pd.Timedelta(days=val_days)
        train_end = val_start
        train_start = train_end - pd.Timedelta(days=train_days)

        train_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
        val_mask = (df[date_col] >= val_start) & (df[date_col] < val_end)

        train_idx = df[train_mask].index.tolist()
        val_idx = df[val_mask].index.tolist()

        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
            print(f"Fold {len(folds)}: Train {train_start.date()} - {train_end.date()} "
                  f"({len(train_idx)}), Val {val_start.date()} - {val_end.date()} ({len(val_idx)})")

    return folds

# Expanding window CV
def expanding_window_cv(df, date_col, min_train_days, val_days, step_days):
    """Validation avec fenêtre croissante."""
    df = df.sort_values(date_col)
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    folds = []
    train_end = min_date + pd.Timedelta(days=min_train_days)

    while train_end + pd.Timedelta(days=val_days) <= max_date:
        val_start = train_end
        val_end = val_start + pd.Timedelta(days=val_days)

        train_mask = df[date_col] < train_end
        val_mask = (df[date_col] >= val_start) & (df[date_col] < val_end)

        folds.append((df[train_mask].index.tolist(), df[val_mask].index.tolist()))
        train_end += pd.Timedelta(days=step_days)

    return folds
```

### 4. Modèle Hybride (Statistique + ML)

```python
def hybrid_model(train, test, date_col, value_col, features):
    """Modèle hybride : trend statistique + résidus ML."""
    from sklearn.linear_model import Ridge

    # Étape 1 : Capturer le trend avec un modèle linéaire
    train['time_idx'] = np.arange(len(train))
    test['time_idx'] = np.arange(len(train), len(train) + len(test))

    trend_model = Ridge()
    trend_model.fit(train[['time_idx']], train[value_col])

    train['trend'] = trend_model.predict(train[['time_idx']])
    test['trend'] = trend_model.predict(test[['time_idx']])

    # Étape 2 : Prédire les résidus avec LightGBM
    train['residual'] = train[value_col] - train['trend']

    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
    lgb_model.fit(train[features], train['residual'])

    train['pred_residual'] = lgb_model.predict(train[features])
    test['pred_residual'] = lgb_model.predict(test[features])

    # Combinaison
    train['prediction'] = train['trend'] + train['pred_residual']
    test['prediction'] = test['trend'] + test['pred_residual']

    return train, test
```

### 5. Récursive vs Direct Forecasting

```python
# DIRECT : entraîner un modèle par horizon
def direct_forecast(train, features, target_col, horizons=[1,7,14,28]):
    """Un modèle par horizon de prévision."""
    models = {}
    for h in horizons:
        train_h = train.copy()
        train_h[f'target_h{h}'] = train_h.groupby('id')[target_col].shift(-h)
        train_h = train_h.dropna(subset=[f'target_h{h}'])

        model = lgb.LGBMRegressor(n_estimators=1000, verbose=-1)
        model.fit(train_h[features], train_h[f'target_h{h}'])
        models[h] = model

    return models

# RÉCURSIF : un modèle, prédire step by step
def recursive_forecast(model, last_known, features, n_steps):
    """Prédiction récursive pas à pas."""
    predictions = []
    current = last_known.copy()

    for step in range(n_steps):
        pred = model.predict(current[features].values.reshape(1, -1))[0]
        predictions.append(pred)
        # Mettre à jour les features (lag, rolling) avec la nouvelle prédiction
        # ADAPTER selon les features utilisées
        current = update_features(current, pred, step)

    return predictions
```

### 6. Features Externes (souvent décisives)

```python
def add_external_features(df, date_col):
    """Ajouter des features externes souvent disponibles."""

    # Jours fériés
    # pip install holidays
    import holidays
    country_holidays = holidays.country_holidays('FR')  # ADAPTER
    df['is_holiday'] = df[date_col].apply(lambda x: x in country_holidays).astype(int)

    # Veille/lendemain de férié
    df['is_day_before_holiday'] = df[date_col].apply(
        lambda x: (x + pd.Timedelta(days=1)) in country_holidays
    ).astype(int)

    # Météo (si pertinent, via API externe)
    # Événements spéciaux (Black Friday, Noël, etc.)
    df['is_black_friday'] = ((df[date_col].dt.month == 11) &
                              (df[date_col].dt.day >= 23) &
                              (df[date_col].dt.day <= 29) &
                              (df[date_col].dt.dayofweek == 4)).astype(int)

    df['is_christmas_period'] = ((df[date_col].dt.month == 12) &
                                  (df[date_col].dt.day >= 15)).astype(int)

    return df
```

## Stratégies Gold Medal Time Series

1. **LightGBM avec lag/rolling features** comme baseline forte
2. **Validation temporelle rigoureuse** (jamais de random split)
3. **Features calendaires + cycliques** obligatoires
4. **Données externes** (jours fériés, événements, météo)
5. **Modèle hybride** : trend statistique + ML pour résidus
6. **Multi-horizon** : direct forecasting pour chaque horizon
7. **Ensemble** : LightGBM + XGBoost + CatBoost + modèle statistique
8. **Attention aux fuites temporelles** : vérifier chaque feature
9. **Grouper si multi-séries** : features par groupe (magasin, produit, etc.)
10. **Post-processing** : clip aux bornes réalistes, arrondi si nécessaire

Adapte TOUJOURS la stratégie au type de série temporelle et à l'horizon de prévision.
