---
name: kaggle-viz
description: Expert en visualisation de données avancée avec Seaborn, Matplotlib et Plotly pour compétitions Kaggle. Utiliser quand l'utilisateur veut créer des graphiques, visualiser des données, faire des plots avancés, ou améliorer ses visualisations.
argument-hint: <type de graphique ou données à visualiser>
---

# Expert Data Visualization - Kaggle Gold Medal

Tu es un expert en visualisation de données. Tu maîtrises Seaborn, Matplotlib et Plotly pour créer des visualisations qui révèlent des insights et guident le feature engineering.

## Setup Standard

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration globale
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Thèmes Seaborn disponibles : darkgrid, whitegrid, dark, white, ticks
sns.set_style("whitegrid")
sns.set_palette("husl")  # Palettes : husl, Set2, deep, muted, bright, pastel, dark, colorblind
```

## 1. Graphiques de Tendance (Trends)

### Line Plot

```python
# Simple
fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=df, x='date', y='value', ax=ax)
ax.set_title('Évolution temporelle')
plt.tight_layout()
plt.show()

# Multi-séries
fig, ax = plt.subplots(figsize=(14, 6))
for col in ['série_A', 'série_B', 'série_C']:
    sns.lineplot(data=df, x='date', y=col, label=col, ax=ax)
ax.legend(title='Séries')
ax.set_title('Comparaison de séries')
plt.tight_layout()
plt.show()

# Avec intervalle de confiance (agrégation automatique)
fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=df, x='month', y='sales', hue='category',
             style='category', markers=True, dashes=False, ax=ax)
ax.set_title('Ventes mensuelles par catégorie')
plt.tight_layout()
plt.show()

# Zone ombrée (fill_between)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['date'], df['value'], color='steelblue', linewidth=2)
ax.fill_between(df['date'], df['value_low'], df['value_high'],
                alpha=0.2, color='steelblue', label='IC 95%')
ax.legend()
ax.set_title('Prédiction avec intervalle de confiance')
plt.tight_layout()
plt.show()
```

## 2. Graphiques de Comparaison

### Bar Plot

```python
# Horizontal (meilleur pour beaucoup de catégories)
fig, ax = plt.subplots(figsize=(10, 8))
order = df.groupby('category')['value'].mean().sort_values(ascending=True).index
sns.barplot(data=df, y='category', x='value', order=order, ax=ax,
            palette='viridis', orient='h')
ax.set_title('Valeur moyenne par catégorie')
plt.tight_layout()
plt.show()

# Groupé (comparaison côte à côte)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=df, x='category', y='value', hue='group', ax=ax)
ax.set_title('Comparaison par groupe')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Avec annotations
fig, ax = plt.subplots(figsize=(12, 6))
bars = sns.barplot(data=df_agg, x='category', y='count', ax=ax, palette='coolwarm')
for bar in bars.patches:
    bars.annotate(f'{bar.get_height():.0f}',
                  (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                  ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Distribution avec annotations')
plt.tight_layout()
plt.show()
```

### Count Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))
order = df['category'].value_counts().index
sns.countplot(data=df, x='category', order=order, palette='viridis', ax=ax)
ax.set_title('Nombre d\'observations par catégorie')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## 3. Heatmaps

### Matrice de Corrélation

```python
# Heatmap de corrélation classique
fig, ax = plt.subplots(figsize=(14, 12))
corr = df.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Masquer la moitié supérieure
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Matrice de Corrélation')
plt.tight_layout()
plt.show()
```

### Heatmap de Données

```python
# Heatmap de données (ex: ventes par mois et catégorie)
pivot = df.pivot_table(values='sales', index='month', columns='category', aggfunc='mean')

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5, ax=ax)
ax.set_title('Ventes moyennes par mois et catégorie')
plt.tight_layout()
plt.show()
```

### Clustermap (Heatmap avec clustering hiérarchique)

```python
# Clustermap : regroupe automatiquement les lignes/colonnes similaires
g = sns.clustermap(corr, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                   figsize=(14, 12), linewidths=0.5)
g.fig.suptitle('Clustermap de Corrélation', y=1.02)
plt.show()
```

## 4. Graphiques de Relation (Scatter, Regression)

### Scatter Plot

```python
# Scatter avec hue (3 variables)
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=df, x='feature_1', y='feature_2', hue='target',
                size='feature_3', sizes=(20, 200), alpha=0.6, ax=ax,
                palette='viridis')
ax.set_title('Relation feature_1 vs feature_2')
plt.tight_layout()
plt.show()
```

### Regression Plot

```python
# Regression linéaire
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=df, x='feature_1', y='target', scatter_kws={'alpha': 0.3},
            line_kws={'color': 'red', 'linewidth': 2}, ax=ax)
ax.set_title(f'Régression : feature_1 vs target (r={df["feature_1"].corr(df["target"]):.3f})')
plt.tight_layout()
plt.show()

# Regression par groupe (lmplot)
g = sns.lmplot(data=df, x='feature_1', y='target', hue='category',
               col='category', col_wrap=3, height=4, aspect=1.2)
g.fig.suptitle('Régression par catégorie', y=1.02)
plt.show()
```

### Pair Plot (matrice de scatter plots)

```python
# Pair plot — excellent pour explorer les relations entre features
g = sns.pairplot(df[features + ['target']], hue='target',
                 diag_kind='kde', corner=True,
                 plot_kws={'alpha': 0.4, 's': 20},
                 palette='Set1', height=2.5)
g.fig.suptitle('Pair Plot des Features', y=1.02)
plt.show()

# Version sélective (top features seulement)
top_features = ['feat1', 'feat2', 'feat3', 'feat4', 'target']
g = sns.pairplot(df[top_features], hue='target', diag_kind='kde',
                 corner=True, palette='coolwarm')
plt.show()
```

### Joint Plot (scatter + distributions marginales)

```python
# Joint plot avec KDE marginale
g = sns.jointplot(data=df, x='feature_1', y='feature_2', kind='scatter',
                  hue='target', height=8, alpha=0.5)
g.fig.suptitle('Joint Plot', y=1.02)
plt.show()

# Joint plot KDE 2D (densité)
g = sns.jointplot(data=df, x='feature_1', y='feature_2', kind='kde',
                  fill=True, height=8, cmap='viridis')
plt.show()

# Joint plot hexbin (pour gros datasets)
g = sns.jointplot(data=df, x='feature_1', y='feature_2', kind='hex',
                  height=8, cmap='YlOrRd')
plt.show()
```

## 5. Graphiques de Distribution

### Histogram

```python
# Histogram simple
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x='feature', bins=50, kde=True, ax=ax,
             color='steelblue', edgecolor='white')
ax.set_title('Distribution de feature')
plt.tight_layout()
plt.show()

# Histogram par catégorie
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x='feature', hue='target', bins=50, kde=True,
             ax=ax, multiple='stack', palette='Set1')  # 'stack', 'dodge', 'layer'
ax.set_title('Distribution par classe')
plt.tight_layout()
plt.show()

# Histogram log-scale (pour distributions asymétriques)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.histplot(data=df, x='feature', bins=50, ax=axes[0])
axes[0].set_title('Distribution originale')
sns.histplot(data=np.log1p(df['feature']), bins=50, ax=axes[1], color='coral')
axes[1].set_title('Distribution log1p(feature)')
plt.tight_layout()
plt.show()
```

### KDE Plot (Kernel Density Estimation)

```python
# KDE simple
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=df, x='feature', fill=True, ax=ax, color='steelblue', alpha=0.5)
ax.set_title('Densité de feature')
plt.tight_layout()
plt.show()

# KDE par catégorie (excellent pour comparer des distributions)
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=df, x='feature', hue='target', fill=True, ax=ax,
            palette='Set1', alpha=0.4, common_norm=False)
ax.set_title('Densité par classe cible')
plt.tight_layout()
plt.show()

# KDE 2D
fig, ax = plt.subplots(figsize=(10, 8))
sns.kdeplot(data=df, x='feature_1', y='feature_2', fill=True,
            cmap='viridis', levels=20, thresh=0.05, ax=ax)
ax.set_title('Densité 2D')
plt.tight_layout()
plt.show()
```

### Box Plot et Violin Plot

```python
# Box plot
fig, ax = plt.subplots(figsize=(12, 6))
order = df.groupby('category')['value'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='category', y='value', order=order, ax=ax,
            palette='viridis', showfliers=True)
ax.set_title('Distribution par catégorie')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Violin plot (box + KDE)
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=df, x='category', y='value', ax=ax, palette='Set2',
               inner='quartile', cut=0)
ax.set_title('Violin Plot par catégorie')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Box + strip (points individuels)
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='target', y='feature', ax=ax, palette='Set1',
            showfliers=False)
sns.stripplot(data=df, x='target', y='feature', ax=ax, color='black',
              alpha=0.3, size=3, jitter=True)
ax.set_title('Distribution par target')
plt.tight_layout()
plt.show()
```

### Swarm Plot

```python
# Swarm plot : chaque point visible, pas de superposition
fig, ax = plt.subplots(figsize=(10, 6))
sns.swarmplot(data=df.sample(min(500, len(df))), x='target', y='feature',
              ax=ax, palette='Set1', size=3)
ax.set_title('Swarm Plot')
plt.tight_layout()
plt.show()
```

## 6. Subplots et Compositions Avancées

### Grille de subplots

```python
# Grille simple
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(features[:6]):
    row, col_idx = i // 3, i % 3
    sns.histplot(data=df, x=col, hue='target', bins=30, ax=axes[row][col_idx],
                 kde=True, palette='Set1', alpha=0.5)
    axes[row][col_idx].set_title(col)
plt.suptitle('Distributions des Features par Classe', fontsize=16)
plt.tight_layout()
plt.show()
```

### GridSpec (disposition flexible)

```python
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Grand graphique à gauche
ax1 = fig.add_subplot(gs[0, :2])
sns.scatterplot(data=df, x='feat1', y='feat2', hue='target', ax=ax1, alpha=0.5)
ax1.set_title('Scatter feat1 vs feat2')

# Petit graphique à droite en haut
ax2 = fig.add_subplot(gs[0, 2])
sns.histplot(data=df, x='target', ax=ax2, palette='Set1')
ax2.set_title('Distribution target')

# Trois graphiques en bas
for i, col in enumerate(['feat1', 'feat2', 'feat3']):
    ax = fig.add_subplot(gs[1, i])
    sns.kdeplot(data=df, x=col, hue='target', fill=True, ax=ax, alpha=0.4)
    ax.set_title(f'KDE {col}')

plt.suptitle('Dashboard EDA', fontsize=16)
plt.show()
```

### FacetGrid (grille par catégorie)

```python
# FacetGrid : un graphique par valeur de catégorie
g = sns.FacetGrid(df, col='category', col_wrap=4, height=3, aspect=1.5)
g.map_dataframe(sns.histplot, x='value', bins=30, kde=True)
g.set_titles(col_template='{col_name}')
g.fig.suptitle('Distribution par catégorie', y=1.02)
plt.show()

# Avec hue
g = sns.FacetGrid(df, col='category', hue='target', col_wrap=3, height=4)
g.map_dataframe(sns.kdeplot, x='feature', fill=True, alpha=0.4)
g.add_legend()
plt.show()
```

## 7. Plotly (Graphiques Interactifs)

```python
# Scatter interactif
fig = px.scatter(df, x='feat1', y='feat2', color='target',
                 size='feat3', hover_data=['id', 'feat4'],
                 title='Scatter Interactif', template='plotly_white')
fig.show()

# Line plot interactif
fig = px.line(df, x='date', y='value', color='category',
              title='Séries Temporelles', template='plotly_white')
fig.show()

# Distribution interactive
fig = px.histogram(df, x='feature', color='target', nbins=50,
                   marginal='box', barmode='overlay', opacity=0.7,
                   title='Distribution avec boxplot marginal')
fig.show()

# Heatmap interactif
fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1, title='Matrice de Corrélation')
fig.show()

# Parallel coordinates (visualiser des features multi-dimensionnelles)
fig = px.parallel_coordinates(df, dimensions=features[:8], color='target',
                              color_continuous_scale='Viridis',
                              title='Coordonnées Parallèles')
fig.show()
```

## 8. Visualisations Spécifiques Compétition

### Train vs Test Distribution

```python
def plot_train_test_dist(train, test, features, ncols=4):
    """Comparer les distributions train/test pour détecter le drift."""
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.kdeplot(data=train, x=col, ax=axes[i], fill=True, alpha=0.4,
                    label='Train', color='steelblue')
        sns.kdeplot(data=test, x=col, ax=axes[i], fill=True, alpha=0.4,
                    label='Test', color='coral')
        axes[i].legend(fontsize=8)
        axes[i].set_title(col, fontsize=10)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution Train vs Test', fontsize=16)
    plt.tight_layout()
    plt.show()
```

### Analyse des Valeurs Manquantes

```python
def plot_missing(df, title='Valeurs Manquantes'):
    """Visualiser les valeurs manquantes."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if len(missing) == 0:
        print("Aucune valeur manquante !")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(missing) * 0.35)))

    # Bar plot des pourcentages
    pct = (missing / len(df) * 100)
    pct.plot(kind='barh', ax=axes[0], color='coral')
    axes[0].set_title(f'{title} (%)')
    axes[0].set_xlabel('% manquant')

    # Heatmap pattern de null (sample)
    sample = df[missing.index].sample(min(200, len(df)))
    sns.heatmap(sample.isnull(), cbar=False, ax=axes[1], cmap='YlOrRd')
    axes[1].set_title('Pattern des valeurs manquantes')

    plt.tight_layout()
    plt.show()
```

### Feature Importance Visualisation

```python
def plot_importance(importance_df, top_n=30, title='Feature Importance'):
    """Visualiser l'importance des features."""
    top = importance_df.head(top_n).sort_values('importance')

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top)))
    ax.barh(top['feature'], top['importance'], color=colors)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.show()
```

### Matrice de Confusion

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Matrice de confusion avec Seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title('Matrice de Confusion')
    plt.tight_layout()
    plt.show()
```

### Learning Curves

```python
def plot_learning_curves(train_losses, val_losses, train_metrics=None, val_metrics=None):
    """Courbes d'apprentissage pour diagnostic overfitting/underfitting."""
    fig, axes = plt.subplots(1, 2 if train_metrics else 1,
                              figsize=(14 if train_metrics else 8, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].plot(train_losses, label='Train Loss', color='steelblue', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', color='coral', linewidth=2)
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].grid(True, alpha=0.3)

    if train_metrics and len(axes) > 1:
        axes[1].plot(train_metrics, label='Train Metric', color='steelblue', linewidth=2)
        axes[1].plot(val_metrics, label='Val Metric', color='coral', linewidth=2)
        axes[1].legend()
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Metric Curves')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

## Guide de Sélection de Graphiques

| Objectif | Graphique | Seaborn |
|----------|-----------|---------|
| Tendance temporelle | Line plot | `sns.lineplot()` |
| Comparer catégories | Bar plot | `sns.barplot()` |
| Corrélation 2 var | Scatter + reg | `sns.regplot()` |
| Matrice corrélation | Heatmap | `sns.heatmap()` |
| Distribution 1 var | Histogram/KDE | `sns.histplot()` / `sns.kdeplot()` |
| Distribution par groupe | Violin/Box | `sns.violinplot()` / `sns.boxplot()` |
| Relations multi-var | Pair plot | `sns.pairplot()` |
| Densité 2D | Joint/KDE 2D | `sns.jointplot()` / `sns.kdeplot()` |
| Comptage catégories | Count plot | `sns.countplot()` |
| Points individuels | Swarm/Strip | `sns.swarmplot()` / `sns.stripplot()` |

Adapte TOUJOURS les visualisations au type de données et à l'objectif d'analyse de l'utilisateur.
