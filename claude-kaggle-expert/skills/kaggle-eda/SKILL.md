---
name: kaggle-eda
description: Effectue une analyse exploratoire des données (EDA) exhaustive sur un dataset Kaggle. Utiliser quand l'utilisateur veut explorer, analyser ou comprendre un dataset.
argument-hint: <chemin_du_dataset ou description>
---

# Analyse Exploratoire des Données (EDA) - Expert Kaggle

Tu es un expert en analyse exploratoire de données. Effectue une EDA exhaustive et professionnelle sur le dataset fourni par l'utilisateur.

## Étape 1 : Chargement et Vue d'Ensemble

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Charger les données
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\n{'='*50}")
print("TYPES DE DONNÉES:")
print(train.dtypes)
print(f"\n{'='*50}")
print("PREMIÈRES LIGNES:")
train.head()
```

## Étape 2 : Statistiques Descriptives

```python
# Statistiques numériques
print("STATISTIQUES NUMÉRIQUES:")
display(train.describe().T)

# Statistiques catégorielles
print("\nSTATISTIQUES CATÉGORIELLES:")
display(train.describe(include='object').T)

# Colonnes uniques
for col in train.columns:
    n_unique = train[col].nunique()
    print(f"{col}: {n_unique} valeurs uniques ({train[col].dtype})")
```

## Étape 3 : Analyse des Valeurs Manquantes

```python
# Pourcentage de valeurs manquantes
missing = pd.DataFrame({
    'train_missing': train.isnull().sum(),
    'train_pct': train.isnull().sum() / len(train) * 100,
    'test_missing': test.isnull().sum(),
    'test_pct': test.isnull().sum() / len(test) * 100,
})
missing = missing[missing['train_pct'] > 0].sort_values('train_pct', ascending=False)

if len(missing) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing)*0.4)))
    missing['train_pct'].plot(kind='barh', ax=ax, color='coral')
    ax.set_title('Pourcentage de Valeurs Manquantes (Train)')
    ax.set_xlabel('%')
    plt.tight_layout()
    plt.show()
    display(missing)
else:
    print("Aucune valeur manquante !")
```

## Étape 4 : Analyse de la Variable Cible

```python
# Adapter selon le type de problème
target_col = 'target'  # À adapter

if train[target_col].dtype in ['int64', 'float64'] and train[target_col].nunique() > 20:
    # RÉGRESSION
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    train[target_col].hist(bins=50, ax=axes[0], color='steelblue')
    axes[0].set_title(f'Distribution de {target_col}')

    np.log1p(train[target_col]).hist(bins=50, ax=axes[1], color='coral')
    axes[1].set_title(f'Distribution de log1p({target_col})')

    from scipy import stats
    stats.probplot(train[target_col], plot=axes[2])
    axes[2].set_title('Q-Q Plot')

    plt.tight_layout()
    plt.show()

    print(f"Skewness: {train[target_col].skew():.4f}")
    print(f"Kurtosis: {train[target_col].kurtosis():.4f}")
else:
    # CLASSIFICATION
    fig, ax = plt.subplots(figsize=(8, 5))
    train[target_col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'Distribution de {target_col}')
    plt.tight_layout()
    plt.show()

    print("Distribution des classes:")
    print(train[target_col].value_counts(normalize=True).round(4))
    print(f"\nDéséquilibre: {train[target_col].value_counts().min() / train[target_col].value_counts().max():.4f}")
```

## Étape 5 : Distributions des Features Numériques

```python
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col)

n_cols_plot = min(len(num_cols), 20)
if n_cols_plot > 0:
    fig, axes = plt.subplots(
        (n_cols_plot + 3) // 4, 4,
        figsize=(20, 5 * ((n_cols_plot + 3) // 4))
    )
    axes = axes.flatten() if n_cols_plot > 1 else [axes]

    for i, col in enumerate(num_cols[:n_cols_plot]):
        train[col].hist(bins=50, ax=axes[i], color='steelblue', alpha=0.7)
        axes[i].set_title(col, fontsize=10)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distributions des Features Numériques', fontsize=14)
    plt.tight_layout()
    plt.show()
```

## Étape 6 : Corrélations

```python
if len(num_cols) > 1:
    # Matrice de corrélation
    corr = train[num_cols + [target_col]].corr()

    fig, ax = plt.subplots(figsize=(min(20, len(num_cols)), min(16, len(num_cols)*0.8)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=len(num_cols) <= 15, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax, square=True)
    ax.set_title('Matrice de Corrélation')
    plt.tight_layout()
    plt.show()

    # Top corrélations avec la cible
    target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
    print(f"Top corrélations avec {target_col}:")
    print(target_corr.head(15).round(4))
```

## Étape 7 : Features Catégorielles

```python
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

if len(cat_cols) > 0:
    for col in cat_cols[:10]:
        n_unique = train[col].nunique()
        print(f"\n{'='*40}")
        print(f"{col} ({n_unique} catégories)")

        if n_unique <= 20:
            fig, ax = plt.subplots(figsize=(10, 4))
            order = train[col].value_counts().index
            sns.countplot(data=train, x=col, order=order, ax=ax)
            ax.set_title(f'Distribution de {col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print(f"  Top 10 valeurs:")
            print(train[col].value_counts().head(10))
```

## Étape 8 : Relations Feature-Target

```python
# Numériques vs Target
for col in num_cols[:8]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    if train[target_col].nunique() <= 10:
        sns.boxplot(data=train, x=target_col, y=col, ax=axes[0])
        sns.violinplot(data=train, x=target_col, y=col, ax=axes[1])
    else:
        sns.scatterplot(data=train.sample(min(5000, len(train))),
                       x=col, y=target_col, alpha=0.3, ax=axes[0])
        sns.regplot(data=train.sample(min(5000, len(train))),
                   x=col, y=target_col, scatter=False, ax=axes[1], color='red')
        axes[1].set_title(f'Trend: {col} vs {target_col}')

    axes[0].set_title(f'{col} vs {target_col}')
    plt.tight_layout()
    plt.show()
```

## Étape 9 : Train vs Test Distribution

```python
# Vérifier le drift train/test
for col in num_cols[:10]:
    fig, ax = plt.subplots(figsize=(10, 4))
    train[col].hist(bins=50, alpha=0.5, label='Train', density=True, ax=ax)
    test[col].hist(bins=50, alpha=0.5, label='Test', density=True, ax=ax)
    ax.legend()
    ax.set_title(f'Distribution Train vs Test: {col}')
    plt.tight_layout()
    plt.show()
```

## Étape 10 : Résumé et Recommandations

À la fin de l'EDA, TOUJOURS fournir un résumé structuré avec :
1. **Taille des données** : nombre de lignes, colonnes, types
2. **Qualité des données** : % missing, outliers identifiés, inconsistances
3. **Variable cible** : distribution, déséquilibre éventuel, transformation suggérée
4. **Features prometteuses** : corrélations fortes, patterns identifiés
5. **Feature engineering suggéré** : interactions, agrégations, encodages
6. **Stratégie de validation recommandée** : StratifiedKFold, GroupKFold, TimeSeriesSplit
7. **Modèles recommandés** : en fonction du type de problème et des données
8. **Points d'attention** : leakage potentiel, drift train/test, features inutiles

Adapte TOUJOURS le code aux données réelles de l'utilisateur (noms de colonnes, types de problème, taille du dataset).

## Definition of Done (DoD)

L'EDA est COMPLÈTE quand :

- [ ] Dataset chargé et shape affiché (train + test)
- [ ] Types de colonnes identifiés (numériques, catégorielles, datetime)
- [ ] Missing values analysées (% par colonne + pattern)
- [ ] Target analysée (distribution, déséquilibre, type de problème)
- [ ] Au moins 3 visualisations produites (distributions, corrélations, target)
- [ ] Train/test drift vérifié (distributions comparées)
- [ ] Résumé structuré avec les 8 points (taille, qualité, target, features, FE suggéré, CV, modèles, attention)
- [ ] Recommandations concrètes pour la suite (features à créer, modèles à tester)
