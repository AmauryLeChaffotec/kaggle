# La Bible de la Data Science & du Machine Learning pour Kaggle

> Guide complet, étape par étape, pour passer de zéro à compétiteur Kaggle. Basé sur l'intégralité des cours et guides Kaggle Learn.

---

## Table des Matières

- **[PARTIE 1 : LES FONDATIONS PYTHON](#partie-1--les-fondations-python)**
  - [Chapitre 1 : Python Essentiel](#chapitre-1--python-essentiel)
  - [Chapitre 2 : Pandas - Maîtriser ses Données](#chapitre-2--pandas--maîtriser-ses-données)
  - [Chapitre 3 : Visualisation des Données](#chapitre-3--visualisation-des-données)
- **[PARTIE 2 : PRÉPARER SES DONNÉES](#partie-2--préparer-ses-données)**
  - [Chapitre 4 : Nettoyage des Données](#chapitre-4--nettoyage-des-données)
  - [Chapitre 5 : Feature Engineering](#chapitre-5--feature-engineering)
- **[PARTIE 3 : MACHINE LEARNING](#partie-3--machine-learning)**
  - [Chapitre 6 : Introduction au Machine Learning](#chapitre-6--introduction-au-machine-learning)
  - [Chapitre 7 : Machine Learning Intermédiaire](#chapitre-7--machine-learning-intermédiaire)
  - [Chapitre 8 : Comprendre ses Modèles (Explainability)](#chapitre-8--comprendre-ses-modèles-explainability)
- **[PARTIE 4 : DEEP LEARNING](#partie-4--deep-learning)**
  - [Chapitre 9 : Introduction au Deep Learning](#chapitre-9--introduction-au-deep-learning)
  - [Chapitre 10 : Computer Vision](#chapitre-10--computer-vision)
  - [Chapitre 11 : Séries Temporelles](#chapitre-11--séries-temporelles)
- **[PARTIE 5 : RÉUSSIR SUR KAGGLE](#partie-5--réussir-sur-kaggle)**
  - [Chapitre 12 : Stratégie de Compétition](#chapitre-12--stratégie-de-compétition)
  - [Chapitre 13 : Le Pipeline Complet de A à Z](#chapitre-13--le-pipeline-complet-de-a-à-z)
- **[ANNEXES](#annexes)**
  - [Cheat Sheets](#cheat-sheets)
  - [Parcours d'Apprentissage Recommandés](#parcours-dapprentissage-recommandés)

---

# PARTIE 1 : LES FONDATIONS PYTHON

Avant de faire du Machine Learning, il faut maîtriser les outils. Python est LE langage de la data science. Pandas est LA librairie pour manipuler les données. Seaborn est LA librairie pour les visualiser.

---

## Chapitre 1 : Python Essentiel

### 1.1 Variables et Arithmétique

Python n'a pas besoin de déclarer les types. Une variable est créée dès qu'on lui assigne une valeur.

```python
# Assigner des variables
age = 25
prix = 19.95
nom = "Kaggle"

# Vérifier le type
type(age)    # int
type(prix)   # float
type(nom)    # str
```

**Les opérateurs arithmétiques :**

| Opérateur | Nom | Exemple | Résultat |
|-----------|-----|---------|----------|
| `+` | Addition | `3 + 2` | `5` |
| `-` | Soustraction | `3 - 2` | `1` |
| `*` | Multiplication | `3 * 2` | `6` |
| `/` | Division | `5 / 2` | `2.5` |
| `//` | Division entière | `5 // 2` | `2` |
| `%` | Modulo (reste) | `5 % 2` | `1` |
| `**` | Puissance | `2 ** 3` | `8` |

> **Piège courant :** `/` retourne toujours un `float` (même `6/2` donne `3.0`). Utilisez `//` pour obtenir un entier.

**Fonctions numériques intégrées :**

```python
min(1, 2, 3)    # 1
max(1, 2, 3)    # 3
abs(-32)        # 32
round(3.14159, 2)  # 3.14
int("807")      # 807 (conversion string -> int)
float(10)       # 10.0
```

### 1.2 Fonctions

```python
def calculer_prix_total(prix_ht, tva=0.20):
    """Calcule le prix TTC à partir du prix HT et du taux de TVA.

    Args:
        prix_ht: Prix hors taxes
        tva: Taux de TVA (défaut: 20%)

    Returns:
        Prix TTC
    """
    return prix_ht * (1 + tva)

# Utilisation
calculer_prix_total(100)          # 120.0 (TVA par défaut 20%)
calculer_prix_total(100, tva=0.055)  # 105.5 (TVA réduite)
```

**Points clés :**
- Toujours écrire un **docstring** pour documenter la fonction.
- Une fonction sans `return` retourne `None`.
- Utiliser `help(fonction)` pour voir la documentation de n'importe quelle fonction.

```python
help(round)  # Affiche la documentation de round()
```

### 1.3 Conditions et Booléens

```python
# Opérateurs de comparaison
3 == 3    # True
3 != 4    # True
3 < 5     # True
3 >= 3    # True

# Combiner avec and, or, not
age = 25
a_permis = True

peut_conduire = (age >= 18) and a_permis  # True

# Structure if/elif/else
def classifier_age(age):
    if age < 13:
        return "enfant"
    elif age < 18:
        return "adolescent"
    elif age < 65:
        return "adulte"
    else:
        return "senior"
```

> **Valeurs "truthy" et "falsey" :** `0`, `""`, `[]`, `None` sont considérés comme `False`. Tout le reste est `True`.

### 1.4 Listes et Tuples

```python
# Créer une liste
planetes = ['Mercure', 'Venus', 'Terre', 'Mars']

# Accéder aux éléments (index commence à 0)
planetes[0]     # 'Mercure' (premier)
planetes[-1]    # 'Mars' (dernier)
planetes[1:3]   # ['Venus', 'Terre'] (du 2e au 3e, 3e exclu)

# Modifier
planetes.append('Jupiter')  # Ajouter à la fin
planetes.pop()              # Retirer le dernier -> 'Jupiter'
planetes[0] = 'Mercury'     # Remplacer

# Fonctions utiles
len(planetes)       # 4
sorted(planetes)    # Tri alphabétique (nouvelle liste)
'Terre' in planetes # True

# Tuples (immutables - ne peuvent pas être modifiés)
coordonnees = (48.8566, 2.3522)  # Paris
lat, lon = coordonnees  # Unpacking

# Astuce : échanger deux variables
a, b = b, a
```

### 1.5 Boucles et List Comprehensions

```python
# Boucle for
for planete in planetes:
    print(planete)

# Boucle avec range
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

# LIST COMPREHENSION - La syntaxe signature de Python
# Structure: [EXPRESSION for ITEM in ITERABLE if CONDITION]

# Exemple simple : carrés de 0 à 9
carres = [n**2 for n in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Avec filtre : planètes courtes (< 6 lettres)
courtes = [p for p in planetes if len(p) < 6]
# ['Venus', 'Terre', 'Mars']

# Transformation + filtre
courtes_majuscules = [p.upper() for p in planetes if len(p) < 6]
# ['VENUS', 'TERRE', 'MARS']

# Astuce : compter avec sum() et bool
nombres = [3, -1, 5, -2, 0, 7]
nb_negatifs = sum(n < 0 for n in nombres)  # 2 (True=1, False=0)
```

> **Analogie SQL pour les comprehensions :**
> ```python
> [p.upper()           # SELECT
>  for p in planetes   # FROM
>  if len(p) < 6]      # WHERE
> ```

### 1.6 Strings et Dictionnaires

```python
# Méthodes de string essentielles
texte = "  Hello, World!  "
texte.strip()          # "Hello, World!" (supprime espaces)
texte.lower()          # "  hello, world!  "
texte.upper()          # "  HELLO, WORLD!  "
texte.split(",")       # ['  Hello', ' World!  ']
"-".join(['2024', '01', '15'])  # '2024-01-15'

# Formatage de strings
nom = "Alice"
score = 0.9543
f"Le score de {nom} est {score:.2%}"  # "Le score de Alice est 95.43%"
f"Valeur: {1234567:,}"               # "Valeur: 1,234,567"

# Dictionnaires (paires clé-valeur)
params = {
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'max_depth': 6
}
params['learning_rate']          # 0.05
params['min_samples'] = 20      # Ajouter une clé
'max_depth' in params            # True

# Dict comprehension
{p: len(p) for p in planetes}
# {'Mercure': 7, 'Venus': 5, 'Terre': 5, 'Mars': 4}

# Parcourir un dictionnaire
for cle, valeur in params.items():
    print(f"{cle} = {valeur}")
```

### 1.7 Librairies Externes

```python
# Imports standards en data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# JAMAIS faire from numpy import * (risque de collision de noms)

# Explorer un objet inconnu : 3 outils
type(objet)    # Quel est son type ?
dir(objet)     # Quelles méthodes a-t-il ?
help(objet)    # Documentation détaillée
```

---

## Chapitre 2 : Pandas - Maîtriser ses Données

Pandas est la librairie de manipulation de données la plus utilisée en data science. Tout tourne autour de deux structures : le **DataFrame** (tableau) et la **Series** (colonne).

### 2.1 Charger et Explorer les Données

```python
import pandas as pd

# Charger un CSV
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Si le CSV a une colonne index (évite "Unnamed: 0")
df = pd.read_csv("data.csv", index_col=0)

# Premières vérifications TOUJOURS faire
print(f"Shape: {train.shape}")      # (nombre_lignes, nombre_colonnes)
train.head()                         # 5 premières lignes
train.dtypes                         # Types de chaque colonne
train.describe()                     # Statistiques (count, mean, std, min, 25%, 50%, 75%, max)
train.info()                         # Résumé complet (types, non-null count, mémoire)
```

> **Astuce :** `describe()` donne le `count` pour chaque colonne. Si `count` varie entre colonnes, c'est qu'il y a des valeurs manquantes.

### 2.2 Sélectionner des Données

```python
# Sélectionner une colonne (retourne une Series)
train['Age']
train.Age          # Équivalent (mais ne marche pas si le nom a des espaces)

# Sélectionner plusieurs colonnes (retourne un DataFrame)
train[['Age', 'Name', 'Survived']]

# iloc - Sélection par POSITION (comme un tableau)
train.iloc[0]           # Première ligne
train.iloc[0:5, 0:3]   # 5 premières lignes, 3 premières colonnes
train.iloc[-5:]         # 5 dernières lignes

# loc - Sélection par LABEL (nom de colonne/index)
train.loc[0, 'Age']                           # Valeur à la ligne 0, colonne 'Age'
train.loc[:, ['Name', 'Age', 'Survived']]     # Toutes les lignes, colonnes spécifiques

# DIFFÉRENCE CRITIQUE :
# iloc[0:10] -> lignes 0 à 9 (fin exclue, comme Python)
# loc[0:10]  -> lignes 0 à 10 (fin INCLUSE)
```

### 2.3 Filtrer les Données

```python
# Condition simple
femmes = train.loc[train.Sex == 'female']

# Conditions multiples (TOUJOURS mettre des parenthèses)
riches_survivantes = train.loc[
    (train.Sex == 'female') & (train.Pclass == 1) & (train.Survived == 1)
]

# OU logique
classe_1_ou_2 = train.loc[(train.Pclass == 1) | (train.Pclass == 2)]

# isin - pour tester l'appartenance à une liste
premieres_classes = train.loc[train.Pclass.isin([1, 2])]

# Filtrer les non-null
avec_age = train.loc[train.Age.notnull()]
```

### 2.4 Statistiques et Transformations

```python
# Statistiques de base
train.Age.mean()          # Moyenne
train.Age.median()        # Médiane
train.Age.std()           # Écart-type
train.Survived.unique()   # Valeurs uniques -> array([0, 1])
train.Pclass.value_counts()  # Comptage par valeur
train.Pclass.value_counts(normalize=True)  # En proportions

# map() - Transformer une colonne élément par élément
train['Age_bin'] = train.Age.map(lambda x: 'enfant' if x < 18 else 'adulte')

# apply() - Transformer ligne par ligne
def extraire_titre(row):
    return row['Name'].split(',')[1].split('.')[0].strip()

train['Title'] = train.apply(extraire_titre, axis='columns')

# OPÉRATIONS VECTORISÉES (plus rapide que map/apply)
train['Age_normalized'] = (train.Age - train.Age.mean()) / train.Age.std()
train['Famille'] = train.SibSp + train.Parch  # Addition de colonnes
```

> **Règle d'or :** Préférez TOUJOURS les opérations vectorisées à `map()` et `apply()`. Elles sont 10 à 100x plus rapides.

### 2.5 Grouper et Agréger

```python
# GroupBy : diviser-appliquer-combiner
train.groupby('Pclass')['Survived'].mean()
# Pclass
# 1    0.629630
# 2    0.472826
# 3    0.242363

# Plusieurs agrégations en une fois avec agg()
train.groupby('Pclass')['Age'].agg(['count', 'mean', 'min', 'max'])

# Grouper par plusieurs colonnes
train.groupby(['Pclass', 'Sex'])['Survived'].mean()

# Si le résultat a un MultiIndex, aplatir avec reset_index()
resultat = train.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()

# Trier les résultats
resultat.sort_values('Survived', ascending=False)
```

### 2.6 Valeurs Manquantes

```python
# Compter les valeurs manquantes
train.isnull().sum()
# Age       177
# Cabin     687
# Embarked    2

# Pourcentage de valeurs manquantes
(train.isnull().sum() / len(train) * 100).round(1)

# Remplir avec une valeur
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna('S', inplace=True)  # Mode (valeur la plus fréquente)

# Supprimer les lignes avec des NaN
train.dropna()               # Supprime TOUTES les lignes avec au moins un NaN
train.dropna(subset=['Age']) # Supprime seulement si Age est NaN

# Supprimer les colonnes avec trop de NaN
train.drop('Cabin', axis=1, inplace=True)
```

### 2.7 Types et Conversions

```python
# Vérifier les types
train.dtypes
# PassengerId      int64
# Name            object   <- Les strings sont "object" en pandas
# Age            float64
# Survived         int64

# Convertir
train['Pclass'] = train['Pclass'].astype('category')
train['Age'] = train['Age'].astype('int32')  # Après avoir rempli les NaN
```

### 2.8 Combiner des DataFrames

```python
# concat - Empiler verticalement (mêmes colonnes)
combined = pd.concat([train, test])

# merge - Joindre comme en SQL
df = pd.merge(train, infos_pays, on='Country', how='left')

# join - Joindre sur l'index
left.join(right, lsuffix='_train', rsuffix='_test')
```

### Aide-mémoire Pandas

| Tâche | Code |
|-------|------|
| Charger CSV | `pd.read_csv("fichier.csv", index_col=0)` |
| Aperçu | `df.head()`, `df.shape`, `df.dtypes` |
| Sélection colonne | `df['col']` |
| Sélection position | `df.iloc[ligne, colonne]` |
| Sélection label | `df.loc[ligne, 'nom_col']` |
| Filtrer | `df.loc[(cond1) & (cond2)]` |
| Statistiques | `df.col.describe()`, `.mean()`, `.value_counts()` |
| Transformation | `df.col.map(func)` ou opérations vectorisées |
| Grouper | `df.groupby('col').agg(['mean', 'count'])` |
| Trier | `df.sort_values('col', ascending=False)` |
| Manquants | `df.isnull().sum()`, `df.fillna(valeur)` |
| Renommer | `df.rename(columns={'ancien': 'nouveau'})` |
| Combiner | `pd.concat([df1, df2])`, `pd.merge(df1, df2, on='clé')` |

---

## Chapitre 3 : Visualisation des Données

La visualisation est essentielle à chaque étape : exploration, feature engineering, évaluation de modèles. Seaborn + Matplotlib sont le duo standard.

### 3.1 Setup Standard

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline  # Pour Jupyter notebooks

# Optionnel : changer le thème
sns.set_style("whitegrid")  # Thèmes: darkgrid, whitegrid, dark, white, ticks
```

### 3.2 Quel Graphique Choisir ?

| Objectif | Type de graphique | Commande Seaborn |
|----------|-------------------|------------------|
| **Tendance** dans le temps | Ligne | `sns.lineplot(data=df)` |
| **Comparer** des groupes | Barres | `sns.barplot(x=..., y=...)` |
| **Patterns** dans un tableau | Heatmap | `sns.heatmap(data=df, annot=True)` |
| **Relation** entre 2 variables | Scatter | `sns.scatterplot(x=..., y=..., hue=...)` |
| **Relation** + tendance | Régression | `sns.regplot(x=..., y=...)` |
| **Distribution** d'une variable | Histogramme | `sns.histplot(data=df, x='col')` |
| **Distribution** lissée | KDE | `sns.kdeplot(data=df, x='col', fill=True)` |
| **Distribution** par catégorie | Swarm | `sns.swarmplot(x='cat', y='num', data=df)` |

### 3.3 Exemples Pratiques

```python
# --- LINE CHART (tendance) ---
plt.figure(figsize=(14, 6))
plt.title("Évolution des ventes")
sns.lineplot(data=ventes_df)
plt.xlabel("Date")

# --- BAR CHART (comparaison) ---
plt.figure(figsize=(10, 6))
plt.title("Taux de survie par classe")
sns.barplot(x='Pclass', y='Survived', data=train)

# --- HEATMAP (corrélations) ---
plt.figure(figsize=(12, 8))
correlation = train.select_dtypes(include='number').corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)

# --- SCATTER PLOT (relation entre 2 variables) ---
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train)

# --- SCATTER + REGRESSION ---
sns.regplot(x='Age', y='Fare', data=train)

# --- SCATTER multi-groupe ---
sns.lmplot(x='Age', y='Fare', hue='Pclass', data=train)

# --- HISTOGRAMME ---
sns.histplot(data=train, x='Age', hue='Survived', bins=30)

# --- KDE (distribution lissée) ---
sns.kdeplot(data=train, x='Age', hue='Survived', fill=True)

# --- DISTRIBUTION 2D ---
sns.jointplot(x=train['Age'], y=train['Fare'], kind='kde')
```

### 3.4 Graphiques Essentiels pour l'EDA Kaggle

```python
# 1. Distribution de la target
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=train)
plt.title("Distribution de la target")

# 2. Matrice de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')

# 3. Distribution de chaque feature numérique
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, col in enumerate(num_cols[:9]):
    ax = axes[i // 3, i % 3]
    sns.histplot(train[col], ax=ax, bins=30)
    ax.set_title(col)
plt.tight_layout()

# 4. Boxplot pour détecter les outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=train)

# 5. Countplot pour les catégorielles
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(['Pclass', 'Sex', 'Embarked']):
    sns.countplot(x=col, hue='Survived', data=train, ax=axes[i])
```

---

# PARTIE 2 : PRÉPARER SES DONNÉES

La qualité des données détermine la qualité du modèle. "Garbage in, garbage out."

---

## Chapitre 4 : Nettoyage des Données

### 4.1 Valeurs Manquantes - Les 3 Stratégies

```python
from sklearn.impute import SimpleImputer

# Stratégie 1 : SUPPRIMER les colonnes (simple mais perd de l'info)
colonnes_avec_nan = [col for col in X_train.columns if X_train[col].isnull().any()]
X_reduit = X_train.drop(colonnes_avec_nan, axis=1)
# MAE typique : 183,550 (pire)

# Stratégie 2 : IMPUTER avec la moyenne (recommandé)
imputer = SimpleImputer(strategy='mean')  # ou 'median', 'most_frequent'
X_impute = pd.DataFrame(imputer.fit_transform(X_train))
X_impute.columns = X_train.columns
# MAE typique : 178,166 (meilleur)

# Stratégie 3 : IMPUTER + indicateur de valeur manquante
for col in colonnes_avec_nan:
    X_train[col + '_manquant'] = X_train[col].isnull()
X_impute = pd.DataFrame(imputer.fit_transform(X_train))
# MAE typique : 178,928 (parfois aide, parfois non)
```

> **Règle critique :** TOUJOURS faire `fit_transform()` sur le TRAIN et `transform()` sur le VALIDATION/TEST. Jamais fit sur tout le dataset !

```python
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)   # fit + transform
X_valid_imputed = imputer.transform(X_valid)        # transform seulement !
X_test_imputed = imputer.transform(X_test)          # transform seulement !
```

### 4.2 Mise à l'Échelle et Normalisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

# SCALING : change l'échelle (0-1), garde la distribution
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
# Quand : SVM, KNN, réseaux de neurones

# STANDARDISATION : moyenne=0, écart-type=1
scaler = StandardScaler()
X_standard = scaler.fit_transform(X_train)
# Quand : régression linéaire, régularisation L1/L2

# NORMALISATION : rend la distribution gaussienne
# Box-Cox (données strictement positives uniquement)
data_normalized = stats.boxcox(data_positive)
# Quand : méthodes qui supposent la normalité (LDA, Naive Bayes)
```

> Les modèles basés sur les arbres (Random Forest, XGBoost, LightGBM) n'ont **PAS BESOIN** de scaling. Ils ne sont pas sensibles à l'échelle.

### 4.3 Parser les Dates

```python
# Les dates sont souvent chargées comme des strings
df['date'].dtype  # object (string)

# Parser avec un format explicite
df['date_parsed'] = pd.to_datetime(df['date'], format="%m/%d/%Y")
# Formats courants :
# "01/15/2024" -> "%m/%d/%Y"
# "15-01-2024" -> "%d-%m-%Y"
# "2024-01-15" -> "%Y-%m-%d"

# Extraire des composantes
df['jour'] = df['date_parsed'].dt.day
df['mois'] = df['date_parsed'].dt.month
df['annee'] = df['date_parsed'].dt.year
df['jour_semaine'] = df['date_parsed'].dt.dayofweek  # 0=lundi, 6=dimanche
```

### 4.4 Encodages de Caractères

```python
import charset_normalizer

# Détecter l'encodage d'un fichier
with open("fichier.csv", 'rb') as f:
    resultat = charset_normalizer.detect(f.read(10000))
print(resultat)  # {'encoding': 'Windows-1252', 'confidence': 0.73}

# Lire avec le bon encodage
df = pd.read_csv("fichier.csv", encoding='Windows-1252')

# Sauvegarder en UTF-8 (le standard)
df.to_csv("fichier_utf8.csv", encoding='utf-8', index=False)
```

### 4.5 Incohérences dans les Données Texte

```python
import fuzzywuzzy
from fuzzywuzzy import process

# Étape 1 : Nettoyer la base (80% des problèmes)
df['pays'] = df['pays'].str.lower().str.strip()

# Étape 2 : Trouver les doublons avec fuzzy matching
pays_uniques = df['pays'].unique()
matches = process.extract("south korea", pays_uniques, limit=5,
                          scorer=fuzzywuzzy.fuzz.token_sort_ratio)
# [('south korea', 100), ('southkorea', 48), ...]

# Étape 3 : Remplacer les variantes
def corriger_matches(df, colonne, valeur_correcte, seuil=47):
    matches = process.extract(valeur_correcte, df[colonne].unique(),
                              limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    proches = [m[0] for m in matches if m[1] >= seuil]
    df.loc[df[colonne].isin(proches), colonne] = valeur_correcte

corriger_matches(df, 'pays', 'south korea')
```

---

## Chapitre 5 : Feature Engineering

Le feature engineering est l'art de transformer les données brutes en features que le modèle peut exploiter efficacement. C'est souvent LA différence entre un modèle moyen et un modèle gagnant.

### 5.1 Principe Fondamental

> **Un feature est utile si la relation qu'il a avec la target est une relation que votre modèle peut apprendre.**

- Un modèle linéaire ne peut apprendre que des relations linéaires → créez des ratios, des polynômes.
- Un arbre de décision divise sur des seuils → il gère bien les features brutes mais mal les sommes.

**Toujours mesurer l'impact :**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Baseline SANS les nouvelles features
baseline_score = -cross_val_score(
    RandomForestRegressor(random_state=0), X, y,
    cv=5, scoring='neg_mean_absolute_error'
).mean()
print(f"Baseline MAE: {baseline_score:.3f}")

# Score AVEC les nouvelles features
new_score = -cross_val_score(
    RandomForestRegressor(random_state=0), X_new, y,
    cv=5, scoring='neg_mean_absolute_error'
).mean()
print(f"New MAE: {new_score:.3f}")
print(f"Amélioration: {baseline_score - new_score:.3f}")
```

### 5.2 Mutual Information - Évaluer l'Utilité des Features

La Mutual Information (MI) mesure la dépendance entre un feature et la target. Contrairement à la corrélation, elle détecte **tout type** de relation (linéaire ou non).

```python
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# Pour la régression
mi_scores = mutual_info_regression(X, y)

# Pour la classification
mi_scores = mutual_info_classif(X, y)

# Visualiser
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
mi_series.plot.barh(figsize=(8, 10))
plt.title("Mutual Information Scores")
```

**Interpréter les scores MI :**
- MI = 0.0 : le feature est indépendant de la target (inutile seul).
- MI > 2.0 : rare, relation très forte.
- Un MI faible ne veut PAS dire que le feature est inutile → il peut être précieux en interaction avec d'autres features.

### 5.3 Créer des Features - Les 5 Techniques

#### Technique 1 : Transformations Mathématiques

```python
# Ratios (très puissants, surtout pour les modèles linéaires)
df['prix_par_m2'] = df['prix'] / df['surface']
df['ratio_chambre'] = df['chambres'] / df['pieces_total']

# Log transform (pour les distributions asymétriques)
df['log_revenu'] = np.log1p(df['revenu'])  # log1p gère les zéros

# Polynômes
df['age_carre'] = df['age'] ** 2
```

#### Technique 2 : Comptages

```python
# Compter les features binaires
colonnes_bool = ['has_pool', 'has_garage', 'has_garden', 'has_basement']
df['nb_equipements'] = df[colonnes_bool].sum(axis=1)

# Compter les composants non-nuls
ingredients = ['Cement', 'Water', 'Sand', 'Gravel']
df['nb_ingredients'] = df[ingredients].gt(0).sum(axis=1)
```

#### Technique 3 : Décomposer / Combiner des Features

```python
# Décomposer une colonne complexe
df[['Type', 'Niveau']] = df['Categorie'].str.split(' ', expand=True)

# Combiner pour créer des interactions
df['marque_modele'] = df['marque'] + '_' + df['modele']
```

#### Technique 4 : Agrégations par Groupe (Group Transforms)

```python
# Moyenne par catégorie
df['revenu_moyen_ville'] = df.groupby('ville')['revenu'].transform('mean')

# Frequency encoding
df['ville_freq'] = df.groupby('ville')['ville'].transform('count') / len(df)

# Écart par rapport à la moyenne du groupe
df['revenu_vs_ville'] = df['revenu'] - df['revenu_moyen_ville']
```

> **Attention au leakage !** Pour les group transforms, calculez sur le train puis mergez dans le test :
> ```python
> # Sur le train
> train['mean_prix_ville'] = train.groupby('ville')['prix'].transform('mean')
> # Pour le test, merger depuis le train
> mapping = train[['ville', 'mean_prix_ville']].drop_duplicates()
> test = test.merge(mapping, on='ville', how='left')
> ```

#### Technique 5 : Features Temporelles

```python
# À partir d'une date
df['mois'] = df['date'].dt.month
df['jour_semaine'] = df['date'].dt.dayofweek
df['est_weekend'] = df['jour_semaine'].isin([5, 6]).astype(int)
df['trimestre'] = df['date'].dt.quarter
df['jours_depuis_debut'] = (df['date'] - df['date'].min()).dt.days
```

### 5.4 Clustering comme Feature

Le clustering K-Means peut découvrir des groupes cachés dans les données.

```python
from sklearn.cluster import KMeans

# Sélectionner les features à clusterer
features_cluster = ['latitude', 'longitude', 'revenu_median']
X_cluster = df[features_cluster].copy()

# IMPORTANT : Normaliser avant le clustering (K-Means est sensible à l'échelle)
X_cluster = (X_cluster - X_cluster.mean()) / X_cluster.std()

# Créer le feature cluster
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(X_cluster)
df['cluster'] = df['cluster'].astype('category')

# Tester différents k via cross-validation
```

### 5.5 PCA (Analyse en Composantes Principales)

PCA décompose la variance des données en composantes orthogonales. Utile pour :
- **Réduire la dimensionnalité** (garder les composantes importantes)
- **Découvrir des patterns** (les loadings révèlent quelles features se combinent)
- **Débruiter** (le signal se concentre dans les premières composantes)

```python
from sklearn.decomposition import PCA

features_pca = ['highway_mpg', 'engine_size', 'horsepower', 'curb_weight']
X_pca = df[features_pca].copy()

# OBLIGATOIRE : Standardiser avant PCA
X_scaled = (X_pca - X_pca.mean()) / X_pca.std()

# Ajuster PCA
pca = PCA()
X_transformed = pca.fit_transform(X_scaled)

# Voir la variance expliquée par composante
print(pca.explained_variance_ratio_)
# [0.68, 0.22, 0.07, 0.03]  -> PC1 capture 68% de la variance

# Examiner les loadings (comprendre ce que chaque PC représente)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(features_pca))],
    index=features_pca
)
print(loadings)
# Si PC1 a des poids [+0.5, +0.5, +0.5, +0.5] -> c'est une "taille globale"
# Si PC2 a des poids [+0.7, -0.7, 0, 0] -> c'est un contraste mpg vs engine_size

# Utiliser les composantes comme features
df['PC1'] = X_transformed[:, 0]
df['PC2'] = X_transformed[:, 1]

# OU créer un feature inspiré par les loadings
df['puissance_vs_poids'] = df['horsepower'] / df['curb_weight']
```

### 5.6 Target Encoding

Le target encoding remplace chaque catégorie par la moyenne de la target pour cette catégorie. Très puissant pour les features à haute cardinalité (beaucoup de catégories uniques).

```python
from category_encoders import MEstimateEncoder

# CRITIQUE : Encoder sur un split SÉPARÉ pour éviter l'overfitting
X_encode = X.sample(frac=0.25, random_state=42)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]

# Créer l'encoder avec lissage (m = smoothing)
encoder = MEstimateEncoder(cols=['code_postal'], m=5.0)
encoder.fit(X_encode, y_encode)
X_train_encoded = encoder.transform(X_pretrain)
```

**Comment fonctionne le lissage m-estimate :**

```
poids = n / (n + m)
encoding = poids * moyenne_categorie + (1 - poids) * moyenne_globale
```

- Catégorie fréquente (n grand) → encoding proche de la moyenne de la catégorie.
- Catégorie rare (n petit) → encoding tiré vers la moyenne globale.
- `m` plus grand = plus de régularisation.

> **Quand utiliser le Target Encoding :** Features à haute cardinalité (code postal, ville, etc.) où le one-hot encoding créerait trop de colonnes.

---

# PARTIE 3 : MACHINE LEARNING

---

## Chapitre 6 : Introduction au Machine Learning

### 6.1 Les 4 Étapes de la Modélisation

Tout modèle de machine learning suit le même workflow :

```
1. DÉFINIR le modèle (choisir l'algorithme et ses paramètres)
2. FIT (entraîner sur les données de train)
3. PREDICT (faire des prédictions)
4. ÉVALUER (mesurer la qualité des prédictions)
```

### 6.2 Votre Premier Modèle

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Charger et explorer
data = pd.read_csv('train.csv')
data = data.dropna(axis=0)  # Simplification : supprimer les NaN
print(data.describe())

# 2. Séparer target (y) et features (X)
y = data['Price']
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea']
X = data[features]

# 3. Séparer en train / validation
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Par défaut : 75% train, 25% validation

# 4. Entraîner et évaluer
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

predictions = model.predict(val_X)
mae = mean_absolute_error(val_y, predictions)
print(f"MAE: {mae:,.0f}")  # Mean Absolute Error
```

> **JAMAIS évaluer sur les données d'entraînement !** Le modèle les a mémorisées. L'erreur sur le train sera trompeusement faible (~435$) alors que l'erreur réelle est bien plus élevée (~265,000$).

### 6.3 Sous-apprentissage vs Sur-apprentissage

C'est LE concept le plus important en Machine Learning.

- **Sous-apprentissage (Underfitting)** : le modèle est trop simple. Il ne capture pas les patterns. Mauvais sur le train ET la validation.
- **Sur-apprentissage (Overfitting)** : le modèle est trop complexe. Il mémorise le bruit. Bon sur le train, mauvais sur la validation.

**L'objectif : trouver le point optimal entre les deux.**

```python
# Tester différentes complexités d'arbre
for max_leaf_nodes in [5, 50, 500, 5000]:
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    print(f"Max leaves: {max_leaf_nodes:>5} → MAE: {mae:,.0f}")

# Résultat typique :
# Max leaves:     5 → MAE: 347,380  (underfitting - trop simple)
# Max leaves:    50 → MAE: 258,171
# Max leaves:   500 → MAE: 243,495  ← POINT OPTIMAL
# Max leaves: 5,000 → MAE: 254,983  (overfitting - trop complexe)
```

### 6.4 Random Forest - Le Premier "Vrai" Modèle

Un Random Forest utilise **beaucoup d'arbres de décision** et **moyenne leurs prédictions**. C'est comme demander l'avis de 100 experts plutôt qu'un seul.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(train_X, train_y)

preds = model.predict(val_X)
mae = mean_absolute_error(val_y, preds)
print(f"Random Forest MAE: {mae:,.0f}")
# ~191,670 (vs ~243,495 pour le meilleur Decision Tree)
```

**Pourquoi Random Forest est excellent pour débuter :**
- Fonctionne bien avec les paramètres par défaut.
- Pas besoin de scaling/normalisation.
- Gère les relations non-linéaires.
- Résistant à l'overfitting grâce à l'ensemble.

---

## Chapitre 7 : Machine Learning Intermédiaire

### 7.1 Variables Catégorielles - Les 3 Approches

```python
# Identifier les colonnes catégorielles
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# --- Approche 1 : Ordinal Encoding ---
# Assigne un entier à chaque catégorie (0, 1, 2, ...)
# Bien pour : catégories ordonnées (Low < Medium < High)
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_valid[cat_cols] = encoder.transform(X_valid[cat_cols])

# --- Approche 2 : One-Hot Encoding ---
# Crée une colonne binaire par catégorie
# Bien pour : catégories non-ordonnées, cardinalité < 15
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_train = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]))
OH_valid = pd.DataFrame(encoder.transform(X_valid[cat_cols]))

# Sélectionner les colonnes à faible cardinalité
low_cardinality = [col for col in cat_cols if X_train[col].nunique() < 10]

# --- Approche 3 : Target Encoding ---
# Voir Chapitre 5.6
# Bien pour : haute cardinalité (code postal, etc.)
```

### 7.2 Pipelines - Assembler le Tout

Les pipelines combinent preprocessing + modèle en un seul objet. C'est plus propre, moins de bugs, et **obligatoire** pour une bonne cross-validation.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Preprocessing pour les colonnes numériques
num_transformer = SimpleImputer(strategy='median')

# Preprocessing pour les colonnes catégorielles
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combiner les deux dans un ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Pipeline complet : preprocessing + modèle
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# Utilisation simple
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)
```

### 7.3 Cross-Validation

Un seul split train/validation peut être trompeur. La cross-validation divise les données en **k folds** et entraîne k modèles différents.

```python
from sklearn.model_selection import cross_val_score

# sklearn utilise le MAE négatif (convention), on multiplie par -1
scores = -cross_val_score(
    pipeline, X, y,
    cv=5,                             # 5 folds
    scoring='neg_mean_absolute_error'
)

print(f"MAE par fold: {scores}")
print(f"MAE moyen: {scores.mean():.0f} (+/- {scores.std():.0f})")
```

**Quand utiliser la cross-validation :**
- **Petit dataset** (< 50,000 lignes) → Cross-validation (plus fiable).
- **Grand dataset** (> 50,000 lignes) → Un seul split suffit (plus rapide).

> **Pourquoi le preprocessing DOIT être dans le pipeline :** Si vous faites le preprocessing AVANT le split, les données de validation "fuient" dans le preprocessing (ex: la moyenne d'imputation inclut les valeurs de validation). C'est du **data leakage**.

### 7.4 XGBoost - L'Algorithme Roi de Kaggle

XGBoost (eXtreme Gradient Boosting) est l'algorithme le plus utilisé en compétitions Kaggle pour les données tabulaires.

**Comment ça marche :**
1. Créer un premier modèle simple (naïf).
2. Calculer les erreurs (résidus).
3. Créer un nouveau modèle pour prédire ces erreurs.
4. Ajouter ce modèle à l'ensemble.
5. Répéter.

Chaque arbre **corrige les erreurs** du précédent (contrairement au Random Forest où les arbres sont indépendants).

```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Version simple
model = XGBRegressor(random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
print(f"MAE: {mean_absolute_error(y_valid, preds):.0f}")
```

**Les 3 paramètres les plus importants :**

```python
model = XGBRegressor(
    n_estimators=1000,       # Nombre d'arbres (mettre haut avec early stopping)
    learning_rate=0.05,      # Contribution de chaque arbre (plus petit = mieux mais plus lent)
    early_stopping_rounds=5, # Arrêter si pas d'amélioration après 5 rounds
    n_jobs=4,                # Parallélisme
    random_state=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],  # Obligatoire pour early stopping
    verbose=False
)
```

> **La formule gagnante XGBoost :**
> `n_estimators=1000` + `learning_rate=0.05` + `early_stopping_rounds=5`

### 7.5 Data Leakage - L'Ennemi Silencieux

Le data leakage est la source n°1 de scores "trop beaux pour être vrais".

#### Type 1 : Target Leakage

Un feature contient une information qui ne serait **pas disponible au moment de la prédiction**.

**Exemple :** Prédire si un patient a une pneumonie. Le feature `a_pris_antibiotiques` est renseigné APRÈS le diagnostic → leakage.

**Comment détecter :**
```python
# Si un feature sépare presque parfaitement la target → suspect
for col in X.columns:
    positive = X[col][y == 1].mean()
    negative = X[col][y == 0].mean()
    if abs(positive - negative) > 0.5:  # Seuil arbitraire
        print(f"SUSPECT: {col} (pos={positive:.2f}, neg={negative:.2f})")
```

**Règle :** Excluez tout feature qui est mis à jour ou créé APRÈS la détermination de la target.

#### Type 2 : Train-Test Contamination

Le preprocessing "fuit" des informations de la validation dans le training.

**Exemple classique :** Faire `fit_transform` d'un `SimpleImputer` sur TOUT le dataset, puis splitter en train/valid. La moyenne d'imputation contient les valeurs de validation !

**Solution :** Utilisez des **pipelines** (voir 7.2). Le preprocessing est automatiquement appliqué uniquement sur les données de training pendant la cross-validation.

> Si votre modèle a une accuracy > 95% sur un problème non-trivial, suspectez du leakage en premier.

---

## Chapitre 8 : Comprendre ses Modèles (Explainability)

Comprendre POURQUOI un modèle prédit ce qu'il prédit est crucial pour : débuguer, améliorer les features, et gagner la confiance.

### 8.1 Permutation Importance - "Quels features comptent ?"

Principe : mélanger aléatoirement les valeurs d'un feature et mesurer combien le score baisse.

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())
```

**Interpréter :**
- Les features en haut sont les plus importants.
- Le chiffre = combien l'accuracy baisse quand on mélange ce feature.
- +/- = variabilité entre les shuffles.
- Valeurs négatives = le feature n'est pas utile (le mélanger améliore même le score par hasard).

### 8.2 Partial Dependence Plots (PDP) - "COMMENT un feature affecte les prédictions ?"

```python
from sklearn.inspection import PartialDependenceDisplay

# PDP pour un seul feature
fig, ax = plt.subplots(figsize=(8, 5))
PartialDependenceDisplay.from_estimator(model, val_X, ['Age'], ax=ax)
plt.title("Effet de l'âge sur la prédiction")

# PDP 2D (interaction entre 2 features)
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(model, val_X, [('Age', 'Fare')], ax=ax)
```

### 8.3 SHAP Values - "POURQUOI cette prédiction spécifique ?"

SHAP décompose chaque prédiction individuelle pour montrer la contribution de chaque feature.

```python
import shap

# Pour les modèles à base d'arbres (rapide et exact)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(val_X)

# Force plot pour UNE prédiction
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], val_X.iloc[0])

# Summary plot global (remplace la permutation importance)
shap.summary_plot(shap_values[1], val_X)
# - Chaque point = une observation
# - Position horizontale = impact sur la prédiction
# - Couleur = valeur du feature (rouge=élevé, bleu=faible)

# Dependence plot (remplace le PDP, avec interactions)
shap.dependence_plot('Age', shap_values[1], val_X, interaction_index='Fare')
```

**Les 3 types d'explainer SHAP :**
- `TreeExplainer` → arbres (XGBoost, LightGBM, Random Forest). Rapide et exact.
- `DeepExplainer` → réseaux de neurones.
- `KernelExplainer` → n'importe quel modèle (mais lent).

### Résumé Explainability

```
Permutation Importance → QUELS features comptent (global)
PDP                    → COMMENT un feature affecte en moyenne
SHAP Summary           → QUELS features + DANS QUEL SENS (global, mieux que PI)
SHAP Dependence        → COMMENT + INTERACTIONS (mieux que PDP)
SHAP Force Plot        → POURQUOI cette prédiction précise (local)
```

---

# PARTIE 4 : DEEP LEARNING

---

## Chapitre 9 : Introduction au Deep Learning

Le Deep Learning utilise des réseaux de neurones pour apprendre des patterns complexes. Il excelle sur les images, le texte, l'audio, et parfois les données tabulaires.

### 9.1 Le Neurone - Brique de Base

Un neurone calcule une fonction linéaire :

```
y = poids_1 * x_1 + poids_2 * x_2 + ... + biais
```

C'est exactement une régression linéaire. Le réseau "apprend" en ajustant les poids et le biais.

```python
from tensorflow import keras
from tensorflow.keras import layers

# Un seul neurone = régression linéaire
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])  # 3 inputs, 1 output
])
```

### 9.2 Réseau Profond (Deep Neural Network)

Empiler des couches de neurones avec une **fonction d'activation** entre chaque couche.

> **Sans activation, empiler des couches ne sert à rien** (la composition de fonctions linéaires reste linéaire).

**ReLU** (Rectified Linear Unit) est l'activation standard :
```
relu(x) = max(0, x)
```

```python
model = keras.Sequential([
    # Couches cachées avec ReLU
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    # Couche de sortie (linéaire pour la régression)
    layers.Dense(1),
])
```

**Règles d'architecture :**
- **Plus large** (plus d'unités) → apprend plus de relations linéaires.
- **Plus profond** (plus de couches) → apprend plus de relations non-linéaires.
- Sortie pour **régression** : `Dense(1)` sans activation.
- Sortie pour **classification binaire** : `Dense(1, activation='sigmoid')`.
- Sortie pour **classification multiclasse** : `Dense(n_classes, activation='softmax')`.

### 9.3 Entraîner un Réseau de Neurones

```python
# COMPILER : choisir l'optimizer et la fonction de perte
model.compile(
    optimizer='adam',           # Adam est le choix par défaut
    loss='mae',                # Pour la régression
    # loss='binary_crossentropy',  # Pour la classification binaire
    # metrics=['binary_accuracy'], # Métrique supplémentaire
)

# ENTRAÎNER
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,            # Taille des mini-batchs
    epochs=500,                # Nombre maximum d'époques
    verbose=0,
)

# VISUALISER la courbe d'apprentissage
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
plt.title("Courbe d'apprentissage")
plt.xlabel("Epoch")
plt.ylabel("Loss")
```

> **IMPORTANT :** Toujours normaliser les features avant d'entraîner un réseau de neurones ! Contrairement aux arbres, les NN sont très sensibles à l'échelle.

### 9.4 Overfitting et Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001,           # Amélioration minimum pour compter
    patience=20,               # Attendre 20 epochs sans amélioration
    restore_best_weights=True, # Revenir au meilleur modèle
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,                # Mettre haut, early stopping gère le reste
    callbacks=[early_stopping],
)
```

> **Stratégie recommandée :** Commencer avec un grand modèle + early stopping. Cela évite le sous-apprentissage tout en empêchant le sur-apprentissage.

### 9.5 Dropout et Batch Normalization

**Dropout** : désactive aléatoirement un pourcentage de neurones à chaque step d'entraînement. C'est comme entraîner un ensemble de sous-réseaux.

**Batch Normalization** : normalise les activations de chaque batch. Accélère et stabilise l'entraînement.

```python
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),           # Désactiver 30% des neurones
    layers.BatchNormalization(),    # Normaliser les activations

    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),

    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),

    layers.Dense(1),  # Sortie régression
])
```

**Pattern standard : `Dense → Dropout → BatchNorm`** à répéter pour chaque couche cachée.

### 9.6 Classification Binaire

| Aspect | Régression | Classification Binaire |
|--------|-----------|----------------------|
| Activation sortie | Aucune (linéaire) | `sigmoid` |
| Fonction de perte | `mae` ou `mse` | `binary_crossentropy` |
| Métrique | loss | `binary_accuracy` |

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[33]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid'),  # Sigmoid → probabilité [0, 1]
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

### Template Deep Learning Complet

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Architecture
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[num_features]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    # Régression :
    layers.Dense(1),
    # OU Classification binaire :
    # layers.Dense(1, activation='sigmoid'),
])

# Compilation
model.compile(
    optimizer='adam',
    loss='mae',  # ou 'binary_crossentropy'
)

# Entraînement avec early stopping
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[EarlyStopping(patience=20, min_delta=0.001, restore_best_weights=True)],
)
```

---

## Chapitre 10 : Computer Vision

La Computer Vision utilise des réseaux convolutifs (CNN) pour analyser les images.

### 10.1 Architecture d'un CNN

Un CNN a deux parties :
1. **Base convolutive** : extrait les features visuels (lignes, textures, formes).
2. **Tête dense** : classifie à partir des features extraits.

```
Image → [Conv2D → ReLU → MaxPool2D] × N → Flatten → Dense → Sortie
```

**Les 3 opérations fondamentales :**

| Opération | Rôle | Couche Keras |
|-----------|------|-------------|
| **Convolution** | Détecte des patterns (bords, textures) | `Conv2D(filters, kernel_size)` |
| **ReLU** | Introduit la non-linéarité | `activation='relu'` |
| **Max Pooling** | Réduit les dimensions, invariance locale | `MaxPool2D(pool_size)` |

### 10.2 Construire un CNN Custom

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Bloc 1
    layers.Conv2D(32, kernel_size=5, activation='relu', padding='same',
                  input_shape=[128, 128, 3]),  # [hauteur, largeur, canaux RGB]
    layers.MaxPool2D(),  # 128x128 → 64x64

    # Bloc 2 (doubler les filtres)
    layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),  # 64x64 → 32x32

    # Bloc 3
    layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),  # 32x32 → 16x16

    # Tête de classification
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
```

> **Pattern standard :** Doubler le nombre de filtres à chaque bloc (32 → 64 → 128) car les dimensions spatiales diminuent avec le pooling.

### 10.3 Transfer Learning (Recommandé)

Plutôt que d'entraîner un CNN from scratch, réutilisez un modèle pré-entraîné sur ImageNet (14M+ images). On remplace juste la tête de classification.

```python
from tensorflow.keras.applications import ResNet50

# Charger le modèle pré-entraîné SANS la tête
pretrained_base = ResNet50(
    include_top=False,    # Exclure la couche de classification originale
    pooling='avg',        # Global Average Pooling
    weights='imagenet'    # Poids pré-entraînés
)

# GELER la base (ne pas toucher aux poids pré-entraînés)
pretrained_base.trainable = False

# Construire le modèle final
model = keras.Sequential([
    pretrained_base,
    layers.Dense(2, activation='softmax'),  # Notre nouvelle classification
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
```

> **Le Transfer Learning est la méthode par défaut** pour la computer vision. Avec seulement 72 images d'entraînement, on peut atteindre 90% d'accuracy.

### 10.4 Data Augmentation

Appliquer des transformations aléatoires aux images d'entraînement pour augmenter artificiellement le dataset.

```python
from tensorflow.keras.layers.experimental import preprocessing

model = keras.Sequential([
    # Augmentation (active seulement pendant l'entraînement)
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomContrast(0.5),
    preprocessing.RandomRotation(0.1),

    # Base pré-entraînée
    pretrained_base,

    # Tête
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
```

### 10.5 Pipeline de Données Optimisé

```python
import tensorflow as tf

# Charger les images depuis un dossier
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'train/',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    batch_size=64,
    shuffle=True,
)

# Optimiser le pipeline (TOUJOURS faire ça)
def to_float(image, label):
    return tf.image.convert_image_dtype(image, tf.float32), label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(to_float).cache().prefetch(buffer_size=AUTOTUNE)
```

---

## Chapitre 11 : Séries Temporelles

Les séries temporelles sont des données ordonnées dans le temps. La prévision (forecasting) est l'un des problèmes les plus courants en ML.

### 11.1 Les 3 Composantes d'une Série Temporelle

```
Série Temporelle = Tendance + Saisonnalité + Cycles + Bruit
```

- **Tendance** : évolution à long terme (hausse, baisse, stable).
- **Saisonnalité** : patterns répétitifs liés au calendrier (jour/semaine/mois/année).
- **Cycles** : patterns de croissance/décroissance non liés au calendrier (économiques, épidémiques).

### 11.2 Features Temporelles

**Type 1 : Features de time-step (modélisent la tendance)**

```python
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess

# Time dummy simple
df['time'] = np.arange(len(df))

# Ou via DeterministicProcess (plus robuste)
dp = DeterministicProcess(
    index=df.index,
    constant=True,   # Intercept
    order=1,          # 1=linéaire, 2=quadratique
    drop=True,
)
X = dp.in_sample()
```

**Type 2 : Lag features (modélisent les cycles)**

```python
# Créer des lags
df['lag_1'] = df['ventes'].shift(1)
df['lag_7'] = df['ventes'].shift(7)
df['lag_28'] = df['ventes'].shift(28)

# Fonction générique
def make_lags(ts, lags):
    return pd.concat(
        {f'y_lag_{i}': ts.shift(i) for i in range(1, lags + 1)},
        axis=1
    ).fillna(0.0)
```

**Type 3 : Features saisonniers**

```python
from statsmodels.tsa.deterministic import CalendarFourier

# Indicateurs saisonniers (pour les courtes périodes)
# Automatiquement créés avec seasonal=True dans DeterministicProcess

# Fourier features (pour les longues périodes, ex: annuelle)
fourier = CalendarFourier(freq="A", order=10)  # 10 paires sin/cos

dp = DeterministicProcess(
    index=df.index,
    constant=True,
    order=1,                     # Tendance linéaire
    seasonal=True,               # Saisonnalité hebdomadaire (indicateurs)
    additional_terms=[fourier],  # Saisonnalité annuelle (Fourier)
    drop=True,
)
```

### 11.3 Modèle Hybride (La Stratégie Gagnante)

Les arbres de décision (XGBoost) **ne peuvent pas extrapoler** une tendance. La solution : un modèle hybride.

```python
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Étape 1 : Régression linéaire pour la tendance + saisonnalité
model_trend = LinearRegression(fit_intercept=False)
model_trend.fit(X_train_trend, y_train)
y_pred_trend = model_trend.predict(X_train_trend)

# Étape 2 : XGBoost pour les résidus (cycles, interactions)
residuals = y_train - y_pred_trend
model_residuals = XGBRegressor()
model_residuals.fit(X_train_lags, residuals)

# Étape 3 : Combiner pour la prédiction finale
y_final = model_trend.predict(X_test_trend) + model_residuals.predict(X_test_lags)
```

### 11.4 Prévision Multi-step

```python
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

# Créer les targets multi-step
def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i+1}': ts.shift(-i) for i in range(steps)},
        axis=1
    )

y = make_multistep_target(df['ventes'], steps=8)

# Stratégie "Direct" : un modèle par step
model = MultiOutputRegressor(XGBRegressor())
model.fit(X_train, y_train)

# Stratégie "DirRec" : chaque step utilise les prédictions précédentes
model = RegressorChain(XGBRegressor())
model.fit(X_train, y_train)
```

> **IMPORTANT :** Ne JAMAIS mélanger (shuffle) des données temporelles lors du split train/test !
> ```python
> train_test_split(X, y, shuffle=False)  # shuffle=False !
> ```

---

# PARTIE 5 : RÉUSSIR SUR KAGGLE

---

## Chapitre 12 : Stratégie de Compétition

### 12.1 Comprendre le Système Kaggle

**Les 2 Leaderboards :**
- **Public LB** : ne score qu'un **sous-ensemble** de vos prédictions test. Visible pendant la compétition.
- **Private LB** : score le reste. C'est le classement final. Révélé à la fin.

> **Règle d'or :** Faites confiance à votre CV local plus qu'au Public LB. Un petit écart entre CV et LB est sain (~1-3%).

**Sélection des soumissions finales :**
Vous choisissez **2 soumissions** avant la fin :
1. Une soumission **safe** : meilleur CV, score LB correct.
2. Une soumission **agressive** : meilleur score LB.

### 12.2 Timeline d'une Compétition

| Semaine | Focus |
|---------|-------|
| **1-2** | EDA, comprendre les données, baseline rapide |
| **2-3** | Feature engineering itératif |
| **3-4** | Modèles multiples, hyperparameter tuning |
| **Dernière** | Ensembling, sélection finale |
| **Dernier jour** | NE PAS changer de stratégie. Faire confiance au CV. |

### 12.3 Les 10 Commandements du Kaggler

1. **Toujours commencer par un baseline simple.** Obtenez une soumission sur le board avant d'optimiser.
2. **Le feature engineering bat l'hyperparameter tuning.** Investissez 70% de votre temps sur les features.
3. **Moins de features = mieux.** Préférez 30 features propres à 100 features bruyantes.
4. **Trackez vos expériences.** Notez chaque changement et son impact sur le CV et le LB.
5. **Validez correctement.** Si votre CV ne corrèle pas avec le LB, votre stratégie de validation est mauvaise.
6. **Commencez avec une forte régularisation.** Ajoutez de la complexité progressivement.
7. **Diversifiez vos modèles.** LightGBM + XGBoost + CatBoost + Random Forest = bon ensemble.
8. **Étudiez les solutions gagnantes** des compétitions passées similaires.
9. **Utilisez les forums.** La Discussion et les Notebooks publics sont des mines d'or.
10. **Ne surapprenez pas sur le LB.** Le Private LB punit ceux qui optimisent pour le Public LB.

### 12.4 Étudier les Solutions Gagnantes

Le meilleur moyen de progresser : après chaque compétition, lire les write-ups des gagnants.

Trouvez-les dans :
- L'onglet **Discussion** de la compétition (filtrer par "solution").
- Le notebook Kaggle ["Winning Solutions of Kaggle Competitions"](https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions).

---

## Chapitre 13 : Le Pipeline Complet de A à Z

Voici le template complet pour une compétition Kaggle, de la première ligne de code à la soumission finale.

### 13.1 Structure du Projet

```
competition_name/
├── data/
│   ├── raw/           # Données brutes (train.csv, test.csv)
│   └── processed/     # Données transformées
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   └── 03_modeling.ipynb
├── src/               # Code réutilisable
├── submissions/       # Fichiers de soumission
├── models/            # Modèles sauvegardés
└── experiments.md     # Journal des expériences
```

### 13.2 Le Template Complet

```python
# ============================================================
# IMPORTS & CONFIGURATION
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
import warnings
import gc

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
TARGET = 'target'  # ADAPTER

def seed_everything(seed=SEED):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()

# ============================================================
# CHARGER LES DONNÉES
# ============================================================
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
sample_sub = pd.read_csv('data/raw/sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"\nDistribution de la target:\n{train[TARGET].value_counts(normalize=True)}")

# ============================================================
# EDA RAPIDE (les vérifications minimales)
# ============================================================
# Valeurs manquantes
print("\nValeurs manquantes (train):")
missing = train.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# Types de colonnes
id_col = train.columns[0]  # ADAPTER
num_cols = train.select_dtypes(include=[np.number]).columns.drop(
    [TARGET, id_col], errors='ignore'
).tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

print(f"\nColonnes numériques: {len(num_cols)}")
print(f"Colonnes catégorielles: {len(cat_cols)}")

# Corrélation avec la target
print(f"\nCorrélation avec {TARGET}:")
print(train[num_cols + [TARGET]].corr()[TARGET].sort_values(ascending=False))

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineering(df):
    """Pipeline de feature engineering. Appliquer sur train ET test."""

    # --- Features numériques ---
    if len(num_cols) > 1:
        df['num_mean'] = df[num_cols].mean(axis=1)
        df['num_std'] = df[num_cols].std(axis=1)
        df['num_nulls'] = df[num_cols].isnull().sum(axis=1)

    # --- Features catégorielles ---
    for col in cat_cols:
        # Frequency encoding (safe, pas de leakage)
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

        # Label encoding
        le = LabelEncoder()
        df[f'{col}_le'] = le.fit_transform(df[col].astype(str))

    # --- AJOUTER VOS FEATURES CUSTOMS ICI ---
    # df['ratio_a_b'] = df['col_a'] / (df['col_b'] + 1e-8)
    # df['interaction'] = df['col_a'] * df['col_b']

    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Features finales
features = [c for c in train.columns if c not in [id_col, TARGET] + cat_cols]
print(f"\nNombre de features: {len(features)}")

# ============================================================
# MODÉLISATION - LightGBM
# ============================================================
lgb_params = {
    'objective': 'binary',          # ADAPTER: 'regression', 'multiclass'
    'metric': 'binary_logloss',     # ADAPTER selon la compétition
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
    'random_state': SEED,
}

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_importance = pd.DataFrame()

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train[TARGET])):
    print(f"\n{'='*40} Fold {fold+1}/{N_FOLDS} {'='*40}")

    X_tr = train.iloc[train_idx][features]
    X_val = train.iloc[val_idx][features]
    y_tr = train.iloc[train_idx][TARGET]
    y_val = train.iloc[val_idx][TARGET]

    model = lgb.LGBMClassifier(**lgb_params)  # ou LGBMRegressor
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )

    # Prédictions OOF (Out-of-Fold)
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Prédictions test (moyenne des folds)
    test_preds += model.predict_proba(test[features])[:, 1] / N_FOLDS

    # Feature importance
    imp = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_,
        'fold': fold
    })
    feature_importance = pd.concat([feature_importance, imp])

# Score CV
cv_score = roc_auc_score(train[TARGET], oof_preds)  # ADAPTER la métrique
print(f"\n{'='*60}")
print(f"CV Score (AUC): {cv_score:.6f}")
print(f"{'='*60}")

# ============================================================
# ANALYSE DES FEATURES IMPORTANTES
# ============================================================
mean_imp = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 12))
mean_imp.head(30).plot.barh()
plt.title("Top 30 Features par Importance")
plt.tight_layout()

# ============================================================
# SOUMISSION
# ============================================================
submission = pd.DataFrame({
    sample_sub.columns[0]: test[id_col],
    sample_sub.columns[1]: test_preds
})

# Pour la classification binaire avec seuil 0.5 :
# submission[sample_sub.columns[1]] = (test_preds > 0.5).astype(int)

submission.to_csv('submissions/submission.csv', index=False)
print(f"\nSoumission créée: {submission.shape}")
print(submission.head())
print(f"\nDistribution des prédictions:")
print(submission[sample_sub.columns[1]].describe())
```

### 13.3 Ensembling - Combiner Plusieurs Modèles

L'ensembling est la technique la plus utilisée par les gagnants Kaggle. L'idée : combiner des modèles différents qui font des erreurs différentes.

```python
# Après avoir entraîné LightGBM, XGBoost, et CatBoost séparément :

# Méthode 1 : Moyenne simple (safe, recommandé pour débuter)
final_preds = (test_lgb + test_xgb + test_catboost) / 3

# Méthode 2 : Moyenne pondérée (si un modèle est meilleur)
final_preds = 0.4 * test_lgb + 0.35 * test_xgb + 0.25 * test_catboost

# Méthode 3 : Rank averaging (robuste aux différences d'échelle)
from scipy.stats import rankdata
rank_lgb = rankdata(test_lgb) / len(test_lgb)
rank_xgb = rankdata(test_xgb) / len(test_xgb)
rank_cat = rankdata(test_catboost) / len(test_catboost)
final_preds = (rank_lgb + rank_xgb + rank_cat) / 3
```

### 13.4 Journal des Expériences

**TOUJOURS** tracker vos expériences. Exemple de format :

| Version | Description | Features | CV Score | LB Score | Gap |
|---------|-------------|----------|----------|----------|-----|
| V1 | Baseline RF | 15 | 0.7850 | 0.7600 | 0.025 |
| V2 | + freq encoding | 22 | 0.8010 | 0.7940 | 0.007 |
| V3 | + 30 interactions | 52 | 0.8100 | 0.7800 | 0.030 |
| V4 | V2 + target enc | 25 | 0.8080 | 0.8010 | 0.007 |

> Leçon de V3 : plus de features n'est pas toujours mieux. Le gap CV-LB de 0.030 indique de l'overfitting.

---

# ANNEXES

## Cheat Sheets

### Pandas en 30 Secondes

```python
# Charger
df = pd.read_csv("file.csv")

# Explorer
df.head(), df.shape, df.dtypes, df.describe(), df.info()
df.isnull().sum(), df['col'].value_counts()

# Filtrer
df.loc[(df.A > 5) & (df.B == 'x')]
df.loc[df.A.isin([1, 2, 3])]

# Transformer
df['new'] = df['A'] / df['B']
df['cat_encoded'] = df['cat'].map(mapping_dict)

# Grouper
df.groupby('cat')['num'].agg(['mean', 'count', 'std'])

# Combiner
pd.concat([df1, df2])
pd.merge(df1, df2, on='key', how='left')
```

### Scikit-Learn en 30 Secondes

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_val)

# Cross-validation
scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
```

### XGBoost / LightGBM en 30 Secondes

```python
# XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=100)

# LightGBM
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
```

### Seaborn en 30 Secondes

```python
sns.lineplot(data=df)                                  # Tendance
sns.barplot(x='cat', y='num', data=df)                 # Comparaison
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')    # Corrélations
sns.scatterplot(x='a', y='b', hue='c', data=df)       # Relation
sns.histplot(data=df, x='col', hue='group', bins=30)   # Distribution
sns.boxplot(x='cat', y='num', data=df)                 # Outliers
```

---

## Parcours d'Apprentissage Recommandés

### Parcours 1 : Débutant Complet (4-6 semaines)

```
Semaine 1-2 : Python → Pandas → Visualisation
Semaine 3   : Intro to Machine Learning
Semaine 4   : Intermediate ML (pipelines, CV, XGBoost)
Semaine 5-6 : Feature Engineering + première compétition (Titanic ou House Prices)
```

### Parcours 2 : Data Scientist Tabulaire (2-4 semaines supplémentaires)

```
Semaine 1   : Data Cleaning + Feature Engineering avancé
Semaine 2   : ML Explainability (SHAP, PDP)
Semaine 3   : Pratiquer sur 2-3 compétitions Getting Started
Semaine 4   : Étudier les solutions gagnantes
```

### Parcours 3 : Deep Learning (4-6 semaines supplémentaires)

```
Semaine 1-2 : Intro to Deep Learning
Semaine 3-4 : Computer Vision + Transfer Learning
Semaine 5-6 : NLP ou Time Series selon intérêt
```

### Parcours 4 : Spécialisation Compétition

```
- Données tabulaires : Feature Engineering + XGBoost/LightGBM + Ensembling
- Computer Vision    : Transfer Learning + Augmentation + EfficientNet
- NLP               : Transformers (BERT, RoBERTa) + Fine-tuning
- Time Series       : Hybrid Models + Fourier Features
```

---

## Correspondance Cours → Chapitres du Guide

| Cours Kaggle | Chapitre du Guide |
|-------------|-------------------|
| Python | Chapitre 1 |
| Pandas | Chapitre 2 |
| Data Visualization | Chapitre 3 |
| Data Cleaning | Chapitre 4 |
| Feature Engineering | Chapitre 5 |
| Intro to Machine Learning | Chapitre 6 |
| Intermediate Machine Learning | Chapitre 7 |
| Machine Learning Explainability | Chapitre 8 |
| Intro to Deep Learning | Chapitre 9 |
| Computer Vision | Chapitre 10 |
| Time Series | Chapitre 11 |
| Guides/Kaggle Competitions | Chapitres 12-13 |

---

> **Dernière règle :** La pratique bat la théorie. Après chaque chapitre, ouvrez un notebook et pratiquez sur un dataset réel. La meilleure façon d'apprendre est de participer à des compétitions Kaggle.

---

*Guide généré à partir de l'intégralité des cours Kaggle Learn (17 cours + 6 guides avancés, 161+ notebooks).*
