# LA BIBLE DE LA DATA SCIENCE & IA POUR KAGGLE

> Guide complet, étape par étape, pour maîtriser le Machine Learning et réussir les compétitions Kaggle.
> Basé sur l'ensemble des cours Kaggle Learn + les guides avancés.

---

## TABLE DES MATIÈRES

- [PARTIE 1 - Les Fondations Python](#partie-1---les-fondations-python)
- [PARTIE 2 - Maîtriser Pandas (Manipulation de Données)](#partie-2---maîtriser-pandas-manipulation-de-données)
- [PARTIE 3 - Exploration & Visualisation des Données (EDA)](#partie-3---exploration--visualisation-des-données-eda)
- [PARTIE 4 - Nettoyage & Préparation des Données](#partie-4---nettoyage--préparation-des-données)
- [PARTIE 5 - Ton Premier Modèle de Machine Learning](#partie-5---ton-premier-modèle-de-machine-learning)
- [PARTIE 6 - Machine Learning Intermédiaire](#partie-6---machine-learning-intermédiaire)
- [PARTIE 7 - Feature Engineering (Créer des Variables)](#partie-7---feature-engineering-créer-des-variables)
- [PARTIE 8 - Deep Learning (Réseaux de Neurones)](#partie-8---deep-learning-réseaux-de-neurones)
- [PARTIE 9 - Computer Vision (Images & CNN)](#partie-9---computer-vision-images--cnn)
- [PARTIE 10 - NLP (Traitement du Langage Naturel)](#partie-10---nlp-traitement-du-langage-naturel)
- [PARTIE 11 - Interpréter & Expliquer ses Modèles](#partie-11---interpréter--expliquer-ses-modèles)
- [PARTIE 12 - Stratégie Compétitions Kaggle](#partie-12---stratégie-compétitions-kaggle)
- [ANNEXES - Cheat Sheets & Templates](#annexes---cheat-sheets--templates)

---

# PARTIE 1 - Les Fondations Python

> **Objectif** : Maîtriser les bases de Python nécessaires pour la Data Science.
> **Sources** : `Python/`, `Intro to Programming/`

## 1.1 Variables et Types

Python est un langage **dynamiquement typé** : pas besoin de déclarer le type d'une variable.

```python
# Les variables se créent simplement par assignation
nom = "Alice"          # str (chaîne de caractères)
age = 25               # int (entier)
taille = 1.75          # float (décimal)
est_etudiant = True    # bool (booléen)

# Vérifier le type d'une variable
type(age)      # <class 'int'>
type(taille)   # <class 'float'>

# Conversion de types
float(10)      # 10.0
int(3.99)      # 3 (tronqué, pas arrondi)
str(42)        # "42"
int("807")     # 807
```

## 1.2 Opérateurs Arithmétiques

```python
a + b      # Addition
a - b      # Soustraction
a * b      # Multiplication
a / b      # Division (renvoie TOUJOURS un float)
a // b     # Division entière (arrondi vers le bas)
a % b      # Modulo (reste de la division)
a ** b     # Puissance

# Exemples
print(5 / 2)    # 2.5 (float)
print(5 // 2)   # 2   (int)
print(7 % 3)    # 1   (reste)
print(2 ** 10)  # 1024

# Fonctions numériques intégrées
min(1, 5, 3)    # 1
max(1, 5, 3)    # 5
abs(-42)        # 42
round(3.14159, 2)  # 3.14
```

> **Attention** : Respecte l'ordre des opérations (PEMDAS). Utilise des parenthèses pour être explicite :
> `(hat_cm + height_cm) / 100` et non `hat_cm + height_cm / 100`

## 1.3 Fonctions

```python
# Définir une fonction avec def + return + docstring
def calculer_ecart_min(a, b, c):
    """Retourne le plus petit écart entre deux nombres parmi a, b, c.

    >>> calculer_ecart_min(1, 5, -5)
    4
    """
    ecart1 = abs(a - b)
    ecart2 = abs(b - c)
    ecart3 = abs(a - c)
    return min(ecart1, ecart2, ecart3)

# Appel
resultat = calculer_ecart_min(10, 20, 15)  # 5

# Arguments par défaut
def saluer(nom="Monde"):
    print(f"Bonjour, {nom} !")

saluer()          # Bonjour, Monde !
saluer("Alice")   # Bonjour, Alice !
```

**`help()` est ton meilleur ami** :

```python
help(round)    # Affiche la documentation de la fonction round
help(len)      # Affiche la documentation de len
```

> **Règle d'or** : Toujours écrire des docstrings pour tes fonctions. `help()` affichera ta docstring.

## 1.4 Listes et Tuples

```python
# Créer une liste
planetes = ['Mercure', 'Vénus', 'Terre', 'Mars', 'Jupiter', 'Saturne']

# Indexation (commence à 0)
planetes[0]     # 'Mercure' (premier)
planetes[-1]    # 'Saturne' (dernier)
planetes[-2]    # 'Jupiter' (avant-dernier)

# Slicing (start:stop, stop EXCLU)
planetes[0:3]   # ['Mercure', 'Vénus', 'Terre']
planetes[:3]    # pareil
planetes[3:]    # ['Mars', 'Jupiter', 'Saturne']
planetes[-2:]   # ['Jupiter', 'Saturne']

# Fonctions utiles
len(planetes)         # 6
sorted(planetes)      # tri alphabétique (nouvelle liste)
sum([2, 3, 5, 7])     # 17

# Méthodes de liste
planetes.append('Neptune')       # ajouter à la fin
planetes.pop()                   # retirer et retourner le dernier
planetes.index('Terre')          # 2 (position)
'Mars' in planetes               # True (test d'appartenance)

# Tuples : comme les listes mais IMMUABLES
coordonnees = (48.8566, 2.3522)
lat, lon = coordonnees           # déballage de tuple

# Astuce : échanger deux variables
a, b = b, a
```

## 1.5 Boucles et List Comprehensions

```python
# Boucle for
for planete in planetes:
    print(planete)

# range() pour les séquences numériques
for i in range(5):
    print(i)   # 0, 1, 2, 3, 4

# Boucle while
compteur = 0
while compteur < 5:
    print(compteur)
    compteur += 1
```

**Les List Comprehensions** : syntaxe concise et puissante pour créer des listes.

```python
# Syntaxe : [expression FOR variable IN itérable IF condition]

# Carrés de 0 à 9
carres = [n**2 for n in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Filtrer : planètes avec un nom court
courtes = [p for p in planetes if len(p) < 5]
# ['Mars']

# Transformer + filtrer
majuscules = [p.upper() for p in planetes if len(p) < 6]

# Compter avec des booléens (True = 1, False = 0)
def compter_negatifs(nombres):
    return sum([n < 0 for n in nombres])

compter_negatifs([1, -3, 5, -2, 0])  # 2
```

## 1.6 Strings et Dictionnaires

```python
# Strings (chaînes de caractères)
message = "Bonjour le monde"
message.upper()           # "BONJOUR LE MONDE"
message.lower()           # "bonjour le monde"
message.split(" ")        # ["Bonjour", "le", "monde"]
", ".join(["a", "b"])     # "a, b"
message.replace("monde", "Kaggle")  # "Bonjour le Kaggle"

# f-strings (formatage moderne)
nom = "Alice"
score = 0.95
print(f"Le score de {nom} est {score:.2%}")  # "Le score de Alice est 95.00%"

# Dictionnaires (clé -> valeur)
params = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 6
}
params["n_estimators"]        # 100
params["min_samples"] = 5     # ajouter une entrée
"max_depth" in params         # True
params.keys()                 # dict_keys([...])
params.values()               # dict_values([...])
params.items()                # dict_items([(clé, val), ...])
```

## 1.7 Imports et Bibliothèques

```python
# Import standard
import math
math.sqrt(16)         # 4.0

# Import avec alias (convention Data Science)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import spécifique
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

---

# PARTIE 2 - Maîtriser Pandas (Manipulation de Données)

> **Objectif** : Savoir charger, explorer, filtrer, transformer et agréger des données.
> **Source** : `Pandas/`

## 2.1 Charger et Créer des Données

```python
import pandas as pd

# Lire un fichier CSV (la commande la plus utilisée)
df = pd.read_csv("train.csv")

# IMPORTANT : utiliser index_col pour éviter une colonne "Unnamed: 0"
df = pd.read_csv("train.csv", index_col=0)

# Premières vérifications (TOUJOURS faire ça en premier)
df.shape        # (8693, 14) -> 8693 lignes, 14 colonnes
df.head()       # 5 premières lignes
df.dtypes       # type de chaque colonne
df.info()       # résumé complet (types, non-null, mémoire)
df.describe()   # statistiques descriptives (count, mean, std, min, max...)
```

**Créer un DataFrame manuellement** :

```python
# Depuis un dictionnaire (clés = noms de colonnes)
pd.DataFrame({
    'Nom': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [0.95, 0.87, 0.92]
})

# Créer une Series (une seule colonne)
pd.Series([100, 200, 300], index=['Jan', 'Fev', 'Mar'], name='Ventes')
```

## 2.2 Sélectionner des Données

### Sélection de colonnes

```python
# Par nom (deux syntaxes)
df['Age']              # retourne une Series
df.Age                 # pareil (mais ne marche pas si le nom a des espaces)
df[['Nom', 'Age']]    # retourne un DataFrame (sous-ensemble de colonnes)
```

### iloc : sélection par POSITION (comme un tableau)

```python
df.iloc[0]             # première LIGNE (retourne une Series)
df.iloc[:, 0]          # première COLONNE
df.iloc[0:5, 0:3]      # lignes 0-4, colonnes 0-2
df.iloc[-5:]           # 5 dernières lignes

# IMPORTANT : iloc EXCLUT la borne de fin (comme Python)
df.iloc[0:3]           # lignes 0, 1, 2 (PAS 3)
```

### loc : sélection par LABEL (nom d'index/colonne)

```python
df.loc[0, 'Nom']                    # valeur à l'index 0, colonne 'Nom'
df.loc[:, ['Nom', 'Age', 'Score']]  # toutes les lignes, 3 colonnes

# IMPORTANT : loc INCLUT la borne de fin (différent de iloc !)
df.loc[0:3]   # lignes 0, 1, 2, 3 (INCLUT 3)
```

### Filtrage conditionnel (le plus utilisé)

```python
# Condition simple
df.loc[df.Age > 30]

# ET (les deux conditions doivent être vraies)
df.loc[(df.Age > 25) & (df.Score > 0.90)]

# OU (au moins une condition vraie)
df.loc[(df.Age < 25) | (df.Score > 0.95)]

# Appartenance à une liste
df.loc[df.Nom.isin(['Alice', 'Bob'])]

# Exclure les valeurs manquantes
df.loc[df.Age.notnull()]
```

## 2.3 Fonctions de Résumé et Transformations

```python
# Statistiques rapides
df.Age.mean()           # moyenne
df.Age.median()         # médiane
df.Age.std()            # écart-type
df.Age.min()            # minimum
df.Age.max()            # maximum
df.Nom.unique()         # valeurs uniques
df.Nom.nunique()        # nombre de valeurs uniques
df.Nom.value_counts()   # comptage de chaque valeur (trié décroissant)

# describe() : résumé automatique selon le type
df.Age.describe()       # count, mean, std, min, 25%, 50%, 75%, max
df.Nom.describe()       # count, unique, top, freq
```

**Transformer des colonnes** :

```python
# Vectorisé (RAPIDE, toujours préférer ça)
df['Age_double'] = df.Age * 2
df['Nom_majuscule'] = df.Nom.str.upper()
df['Combinaison'] = df.Nom + " - " + df.Age.astype(str)

# map() : transformer chaque élément d'une Series
df.Score.map(lambda x: "Bon" if x > 0.9 else "Moyen")

# apply() : transformer chaque LIGNE d'un DataFrame
def categoriser(row):
    if row.Age > 30 and row.Score > 0.9:
        return "Senior Expert"
    return "Autre"

df['Categorie'] = df.apply(categoriser, axis=1)  # axis=1 = par ligne
```

> **Règle de performance** : Utilise les opérations vectorisées (`df.col * 2`) plutôt que `map()`/`apply()` chaque fois que c'est possible. C'est beaucoup plus rapide.

## 2.4 Groupby et Tri

```python
# groupby : agréger par catégorie (split-apply-combine)
df.groupby('Categorie').Score.mean()      # score moyen par catégorie
df.groupby('Categorie').Age.count()       # nombre par catégorie

# Plusieurs agrégations à la fois
df.groupby('Categorie').Score.agg(['count', 'mean', 'min', 'max'])

# Grouper par plusieurs colonnes
df.groupby(['Categorie', 'Nom']).Score.mean()

# reset_index() : remettre les groupes en colonnes normales
resultat = df.groupby('Categorie').Score.mean().reset_index()

# Trier
df.sort_values(by='Score', ascending=False)            # tri décroissant
df.sort_values(by=['Categorie', 'Score'])               # tri multi-colonnes
```

## 2.5 Valeurs Manquantes et Types

```python
# Détecter les valeurs manquantes
df.isnull().sum()           # nombre de NaN par colonne
df.Age.isnull()             # booléen par ligne

# Remplir les valeurs manquantes
df.Age.fillna(0)                     # remplir par 0
df.Age.fillna(df.Age.median())       # remplir par la médiane
df.fillna(method='ffill')            # forward fill (propagation avant)

# Supprimer les lignes avec des NaN
df.dropna()                # supprime toute ligne avec au moins un NaN
df.dropna(subset=['Age'])  # supprime seulement si NaN dans Age

# Changer le type
df.Age.astype('float64')
df.Score.astype('int')

# Remplacer des valeurs
df.Nom.replace('Alice', 'Alicia')
```

## 2.6 Fusionner et Combiner

```python
# Concaténer (empiler des DataFrames)
pd.concat([df1, df2])                 # empiler verticalement
pd.concat([df1, df2], axis=1)         # coller côte à côte

# Merge (jointure comme en SQL)
pd.merge(df_gauche, df_droite, on='id_commun')
pd.merge(df_gauche, df_droite, on='id', how='left')   # left join
pd.merge(df_gauche, df_droite, on='id', how='inner')  # inner join
```

---

# PARTIE 3 - Exploration & Visualisation des Données (EDA)

> **Objectif** : Comprendre les données visuellement avant de modéliser.
> **Source** : `Data Visualization/`

## 3.1 Setup (À mettre en haut de chaque notebook)

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Style propre
sns.set_style("whitegrid")
```

## 3.2 Les Graphiques Essentiels

### Line Chart (tendances temporelles)

```python
plt.figure(figsize=(14, 6))
plt.title("Évolution des ventes")
sns.lineplot(data=df_ventes)  # trace une ligne par colonne
plt.xlabel("Date")
plt.ylabel("Ventes")
plt.show()

# Tracer des colonnes spécifiques
sns.lineplot(data=df['Produit_A'], label="Produit A")
sns.lineplot(data=df['Produit_B'], label="Produit B")
```

### Bar Chart (comparer des catégories)

```python
plt.figure(figsize=(10, 6))
plt.title("Score moyen par catégorie")
sns.barplot(x=df.index, y=df['Score'])
plt.ylabel("Score")
plt.show()
```

### Heatmap (matrice de valeurs)

```python
plt.figure(figsize=(14, 7))
plt.title("Corrélation entre les variables")
sns.heatmap(data=df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()
```

> **Astuce** : Les heatmaps de corrélation sont excellentes pour repérer les variables liées entre elles.

### Scatter Plot (relation entre 2 variables)

```python
# Nuage de points simple
sns.scatterplot(x=df['Age'], y=df['Revenu'])

# Avec ligne de régression
sns.regplot(x=df['Age'], y=df['Revenu'])

# Coloré par catégorie
sns.scatterplot(x=df['Age'], y=df['Revenu'], hue=df['Categorie'])

# Régression séparée par catégorie
sns.lmplot(x="Age", y="Revenu", hue="Categorie", data=df)
```

### Distribution (histogramme et KDE)

```python
# Histogramme
sns.histplot(df['Age'], bins=30)

# KDE (distribution lissée)
sns.kdeplot(data=df['Age'], fill=True)

# Comparer les distributions par catégorie
sns.kdeplot(data=df, x='Age', hue='Categorie', fill=True)
plt.title("Distribution de l'âge par catégorie")
```

> **Astuce** : Si les distributions de deux classes sont bien séparées sur une feature, cette feature est probablement un bon prédicteur.

### Quel graphique choisir ?

| Situation | Graphique |
|-----------|-----------|
| Tendance dans le temps | `sns.lineplot()` |
| Comparer des catégories | `sns.barplot()` |
| Matrice 2D (corrélation, etc.) | `sns.heatmap()` |
| Relation entre 2 variables continues | `sns.scatterplot()` ou `sns.regplot()` |
| Distribution d'une variable | `sns.histplot()` ou `sns.kdeplot()` |
| Distribution par catégorie | `sns.kdeplot(hue=...)` |

## 3.3 L'EDA Complète (Checklist)

Avant de modéliser, TOUJOURS faire cette analyse :

```python
# 1. Dimensions et premières lignes
print(f"Shape: {df.shape}")
df.head()

# 2. Types et valeurs manquantes
df.info()
print("\nValeurs manquantes :")
print(df.isnull().sum().sort_values(ascending=False))
print(f"\n% manquant total : {df.isnull().sum().sum() / df.size * 100:.1f}%")

# 3. Statistiques descriptives
df.describe()          # numériques
df.describe(include='object')  # catégorielles

# 4. Distribution de la target
sns.histplot(df['target'])
plt.title("Distribution de la variable cible")
plt.show()

# 5. Corrélation avec la target
correlations = df.corr()['target'].sort_values(ascending=False)
print(correlations)

# 6. Heatmap de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()
```

---

# PARTIE 4 - Nettoyage & Préparation des Données

> **Objectif** : Rendre les données propres et exploitables pour un modèle.
> **Source** : `Data Cleaning/`

## 4.1 Gérer les Valeurs Manquantes

**Première question** : La valeur manque parce qu'elle **n'a pas été enregistrée** ou parce qu'elle **n'existe pas** ?

- **Pas enregistrée** → imputer (remplir avec une estimation)
- **N'existe pas** → laisser NaN ou remplir avec "None" / 0

```python
import numpy as np

# 1. Diagnostic : combien de valeurs manquantes ?
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
pd.DataFrame({'Manquantes': missing, '%': missing_pct}).sort_values('%', ascending=False)

# 2. Stratégie : Suppression
df.dropna()                    # supprimer les LIGNES avec NaN (souvent trop agressif)
df.dropna(axis=1)              # supprimer les COLONNES avec NaN

# 3. Stratégie : Imputation simple
df['Age'].fillna(df['Age'].median(), inplace=True)  # médiane (numérique)
df['Ville'].fillna('Inconnu', inplace=True)          # constante (catégorielle)
df.fillna(method='ffill')                             # propagation avant
df.fillna(method='bfill')                             # propagation arrière

# 4. Stratégie : Imputation avec sklearn (RECOMMANDÉ pour le ML)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')  # ou 'mean', 'most_frequent', 'constant'
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[numerical_cols]),
    columns=numerical_cols
)
```

> **Règle** : Toujours `fit_transform()` sur les données d'entraînement, puis `transform()` sur les données de validation/test. JAMAIS fit sur le test.

## 4.2 Scaling et Normalisation

**Scaling** (mise à l'échelle) : change la PLAGE des valeurs (0-1). La forme de la distribution reste la même.

**Normalisation** : change la FORME de la distribution pour la rendre gaussienne (cloche).

```python
# SCALING : MinMaxScaler (ramène entre 0 et 1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['Age', 'Revenu']] = scaler.fit_transform(df[['Age', 'Revenu']])

# NORMALISATION : Box-Cox (rend la distribution gaussienne)
from scipy import stats
normalized_data, lambda_param = stats.boxcox(df['Revenu_positif'])
# ATTENTION : Box-Cox nécessite des valeurs STRICTEMENT positives
```

**Quand utiliser quoi ?**

| Algorithme | A besoin de... |
|-----------|----------------|
| KNN, SVM | **Scaling** (sensible aux distances) |
| Régression linéaire | Scaling aide |
| Naive Bayes Gaussien, LDA | **Normalisation** (suppose une distribution gaussienne) |
| Arbres (RF, XGBoost) | **Rien** (insensible à l'échelle) |

## 4.3 Parser les Dates

```python
# Convertir une colonne string en datetime
df['date'] = pd.to_datetime(df['date'], format="%m/%d/%Y")

# Auto-détection du format (plus lent, pas toujours correct)
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

# Extraire des composantes
df['jour'] = df['date'].dt.day
df['mois'] = df['date'].dt.month
df['annee'] = df['date'].dt.year
df['jour_semaine'] = df['date'].dt.dayofweek  # 0=lundi, 6=dimanche
```

## 4.4 Encodages de Caractères

```python
# Détecter l'encodage d'un fichier
import charset_normalizer

with open("fichier.csv", 'rb') as f:
    resultat = charset_normalizer.detect(f.read(10000))
print(resultat)  # {'encoding': 'Windows-1252', 'confidence': 0.73}

# Lire avec le bon encodage
df = pd.read_csv("fichier.csv", encoding='Windows-1252')

# Sauvegarder en UTF-8 (standard)
df.to_csv("fichier_utf8.csv", index=False)
```

## 4.5 Nettoyage de Texte

```python
# Étape 1 : Normaliser la casse et les espaces (résout 80% des problèmes)
df['pays'] = df['pays'].str.lower().str.strip()

# Étape 2 : Fuzzy matching pour les typos restantes
from fuzzywuzzy import process, fuzz

def corriger_valeurs(df, colonne, valeur_correcte, seuil=80):
    """Remplace les variantes proches d'une valeur par la valeur correcte."""
    valeurs_uniques = df[colonne].unique()
    matches = process.extract(valeur_correcte, valeurs_uniques,
                              limit=10, scorer=fuzz.token_sort_ratio)
    proches = [m[0] for m in matches if m[1] >= seuil]
    df.loc[df[colonne].isin(proches), colonne] = valeur_correcte

# Exemple : corriger "south korea", "s. korea", "south koera" → "south korea"
corriger_valeurs(df, 'pays', 'south korea', seuil=80)
```

---

# PARTIE 5 - Ton Premier Modèle de Machine Learning

> **Objectif** : Comprendre et construire un modèle ML de bout en bout.
> **Source** : `Intro to Machine Learning/`

## 5.1 Comment Fonctionne un Modèle

Un **arbre de décision** divise les données en groupes en posant des questions successives sur les features :

```
                    Combien de chambres ?
                     /              \
                  ≤ 3              > 3
                 /                   \
        Jardin ?                  Piscine ?
        /     \                   /       \
     Oui     Non              Oui       Non
    200k€   150k€            500k€     350k€
```

Chaque feuille (bout de l'arbre) donne une prédiction basée sur la moyenne historique du groupe.

## 5.2 Les 4 Étapes du Machine Learning

```
1. DÉFINIR  →  2. ENTRAÎNER  →  3. PRÉDIRE  →  4. ÉVALUER
(choisir       (fit sur les      (predict sur   (mesurer
l'algo)         données)          nouvelles      l'erreur)
                                  données)
```

## 5.3 Premier Modèle Complet

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# --- 1. CHARGER ET PRÉPARER ---
df = pd.read_csv('train.csv')
df = df.dropna()  # simple pour commencer

# Séparer target (y) et features (X)
y = df['SalePrice']                         # ce qu'on veut prédire
features = ['LotArea', 'YearBuilt', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df[features]

# Séparer en entraînement / validation (75% / 25%)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)

# --- 2. DÉFINIR ET ENTRAÎNER ---
model = DecisionTreeRegressor(random_state=42)
model.fit(train_X, train_y)

# --- 3. PRÉDIRE ---
predictions = model.predict(val_X)

# --- 4. ÉVALUER ---
mae = mean_absolute_error(val_y, predictions)
print(f"MAE: {mae:,.0f}")  # ex: MAE: 29,652
```

## 5.4 Pourquoi Valider ? (Ne JAMAIS évaluer sur les données d'entraînement)

```python
# MAUVAIS : évaluer sur les données d'entraînement
train_predictions = model.predict(train_X)
print(f"MAE train: {mean_absolute_error(train_y, train_predictions):,.0f}")
# Résultat : ~0 (le modèle a MÉMORISÉ les données, pas appris)

# BON : évaluer sur les données de validation
val_predictions = model.predict(val_X)
print(f"MAE validation: {mean_absolute_error(val_y, val_predictions):,.0f}")
# Résultat : ~30,000 (la VRAIE performance)
```

> Le modèle qui mémorise les données d'entraînement obtient un score parfait sur celles-ci, mais se plante sur de nouvelles données. C'est le **surapprentissage** (overfitting).

## 5.5 Overfitting vs Underfitting

| Problème | Symptôme | Cause | Solution |
|----------|----------|-------|----------|
| **Underfitting** | Mauvais score partout | Modèle trop simple | Plus de profondeur / features |
| **Overfitting** | Bon score train, mauvais score validation | Modèle trop complexe | Moins de profondeur / régularisation |

```python
# Trouver le bon compromis avec max_leaf_nodes
def evaluer_arbre(max_feuilles, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_feuilles, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y, preds)

# Tester plusieurs tailles
for n in [5, 50, 500, 5000]:
    mae = evaluer_arbre(n, train_X, val_X, train_y, val_y)
    print(f"max_leaf_nodes: {n:>5}  →  MAE: {mae:>10,.0f}")

# Résultat typique :
# max_leaf_nodes:     5  →  MAE:    347,380  (underfitting)
# max_leaf_nodes:    50  →  MAE:    258,171
# max_leaf_nodes:   500  →  MAE:    243,495  ← MEILLEUR
# max_leaf_nodes: 5,000  →  MAE:    254,983  (overfitting)
```

## 5.6 Random Forest : Le Premier Modèle Sérieux

Un **Random Forest** crée BEAUCOUP d'arbres de décision et fait la moyenne de leurs prédictions. Chaque arbre utilise un sous-ensemble aléatoire des données et des features.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)
preds = model.predict(val_X)
print(f"MAE: {mean_absolute_error(val_y, preds):,.0f}")
# Résultat : ~190,000 (bien meilleur que le meilleur arbre simple ~243,000)
```

> **Pourquoi c'est mieux** : Un seul arbre overfit facilement. En moyennant beaucoup d'arbres différents, les erreurs individuelles se compensent. C'est le principe de l'**ensemble**.

## 5.7 Soumettre sur Kaggle

```python
# Entraîner sur TOUTES les données d'entraînement
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)  # X et y complets, pas de split

# Prédire sur les données de test
test_data = pd.read_csv('test.csv')
test_X = test_data[features]
test_preds = model.predict(test_X)

# Créer le fichier de soumission
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_preds
})
submission.to_csv('submission.csv', index=False)
```

---

# PARTIE 6 - Machine Learning Intermédiaire

> **Objectif** : Gérer les vrais problèmes de données, utiliser XGBoost et les pipelines.
> **Source** : `Intermediate Machine Learning/`

## 6.1 Gérer les Valeurs Manquantes (3 Approches)

```python
from sklearn.impute import SimpleImputer
import pandas as pd

# Identifier les colonnes avec des NaN
cols_manquantes = [col for col in X_train.columns if X_train[col].isnull().any()]

# --- APPROCHE 1 : Supprimer les colonnes (simple mais perte d'info) ---
X_train_reduit = X_train.drop(cols_manquantes, axis=1)
X_valid_reduit = X_valid.drop(cols_manquantes, axis=1)

# --- APPROCHE 2 : Imputation (RECOMMANDÉ) ---
imputer = SimpleImputer(strategy='median')  # ou 'mean', 'most_frequent'
X_train_impute = pd.DataFrame(imputer.fit_transform(X_train))
X_valid_impute = pd.DataFrame(imputer.transform(X_valid))  # transform, PAS fit_transform !
X_train_impute.columns = X_train.columns
X_valid_impute.columns = X_valid.columns

# --- APPROCHE 3 : Imputation + indicateur de manquant ---
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()
for col in cols_manquantes:
    X_train_plus[col + '_manquant'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_manquant'] = X_valid_plus[col].isnull()

imputer = SimpleImputer(strategy='median')
X_train_plus = pd.DataFrame(imputer.fit_transform(X_train_plus))
X_valid_plus = pd.DataFrame(imputer.transform(X_valid_plus))
```

## 6.2 Variables Catégorielles (3 Approches)

```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Identifier les colonnes catégorielles
cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

# --- APPROCHE 1 : Supprimer (simple, souvent sous-optimal) ---
X_train_num = X_train.select_dtypes(exclude=['object'])

# --- APPROCHE 2 : Ordinal Encoding (pour les variables avec un ORDRE naturel) ---
# Ex: "Jamais" < "Rarement" < "Souvent" < "Toujours"
ordinal_encoder = OrdinalEncoder()
X_train[cat_cols] = ordinal_encoder.fit_transform(X_train[cat_cols])
X_valid[cat_cols] = ordinal_encoder.transform(X_valid[cat_cols])

# --- APPROCHE 3 : One-Hot Encoding (pour les variables SANS ordre) ---
# Ex: "Rouge", "Bleu", "Vert" → pas d'ordre logique
# NE PAS utiliser si > 15 catégories (cardinality trop haute)
low_card_cols = [col for col in cat_cols if X_train[col].nunique() < 10]

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_card_cols]))
OH_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_card_cols]))
OH_train.index = X_train.index
OH_valid.index = X_valid.index

# Remplacer les colonnes catégorielles par les one-hot
num_train = X_train.drop(cat_cols, axis=1)
X_train_final = pd.concat([num_train, OH_train], axis=1)
X_train_final.columns = X_train_final.columns.astype(str)
```

> **`handle_unknown='ignore'`** : évite les erreurs quand le test contient des catégories absentes de l'entraînement.

## 6.3 Pipelines (Le Standard Professionnel)

Un **Pipeline** empaquette le preprocessing et le modèle dans un seul objet. C'est la bonne façon de faire du ML.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Identifier les types de colonnes
numerical_cols = [c for c in X_train.columns if X_train[c].dtype in ['int64', 'float64']]
categorical_cols = [c for c in X_train.columns
                    if X_train[c].dtype == 'object' and X_train[c].nunique() < 10]

# Étape 1 : Définir le preprocessing
numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Étape 2 : Créer le pipeline complet
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# Étape 3 : Entraîner et évaluer (UNE seule ligne !)
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)
print(f"MAE: {mean_absolute_error(y_valid, preds):,.0f}")
```

**Avantages du Pipeline** :
1. Code plus propre et plus court
2. Pas de fuite de données (preprocessing fit seulement sur le train)
3. Facile à déployer (un seul objet à sauvegarder)
4. Compatible avec la cross-validation

## 6.4 Cross-Validation

Au lieu d'un seul split train/validation, on fait **K splits** rotatifs pour une évaluation plus fiable.

```python
from sklearn.model_selection import cross_val_score

# Le pipeline gère automatiquement le preprocessing à chaque fold
scores = -1 * cross_val_score(
    pipeline, X, y,
    cv=5,                           # 5 folds
    scoring='neg_mean_absolute_error'
)

print("MAE par fold:", scores)
print(f"MAE moyenne: {scores.mean():,.0f} (+/- {scores.std():,.0f})")
```

> **Quand utiliser la cross-validation ?**
> - Petit dataset (< 10,000 lignes) : **OUI** (un seul split est trop bruité)
> - Grand dataset : un simple split suffit, la CV prend trop de temps
> - Si les scores varient beaucoup entre folds → tu as BESOIN de la CV

## 6.5 XGBoost (Le Roi du Tabular ML)

**XGBoost** (eXtreme Gradient Boosting) construit des arbres **séquentiellement**, chaque nouvel arbre corrigeant les erreurs des précédents.

```python
from xgboost import XGBRegressor  # ou XGBClassifier pour la classification

# Modèle de base
model = XGBRegressor(random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
print(f"MAE: {mean_absolute_error(y_valid, preds):,.0f}")
```

**La recette gagnante** :

```python
model = XGBRegressor(
    n_estimators=1000,       # beaucoup d'arbres
    learning_rate=0.05,      # petits pas (meilleure généralisation)
    n_jobs=4,                # parallélisme (vitesse)
    random_state=0
)

model.fit(
    X_train, y_train,
    early_stopping_rounds=5,                    # arrêter si pas d'amélioration
    eval_set=[(X_valid, y_valid)],              # surveiller la validation
    verbose=False
)
```

**Les paramètres clés de XGBoost** :

| Paramètre | Description | Valeur typique |
|-----------|-------------|----------------|
| `n_estimators` | Nombre d'arbres | 500 - 2000 |
| `learning_rate` | Taille du pas | 0.01 - 0.1 |
| `max_depth` | Profondeur des arbres | 3 - 8 |
| `early_stopping_rounds` | Patience avant arrêt | 5 - 50 |
| `subsample` | % de données par arbre | 0.6 - 1.0 |
| `colsample_bytree` | % de features par arbre | 0.6 - 1.0 |
| `reg_alpha` | Régularisation L1 | 0 - 10 |
| `reg_lambda` | Régularisation L2 | 0 - 10 |

## 6.6 Data Leakage (Fuite de Données)

La fuite de données fait que ton modèle semble fantastique en local mais se plante en production. **C'est le piège #1.**

### Type 1 : Target Leakage

Une feature contient une info qui n'est disponible qu'APRÈS la target.

```
Exemple : Prédire "a eu une pneumonie"
Feature "a pris des antibiotiques" → LEAKAGE ! (prescrit APRÈS le diagnostic)
```

**Règle** : Pour chaque feature, demande-toi "Est-ce que cette info serait disponible AU MOMENT de la prédiction ?"

### Type 2 : Train-Test Contamination

Le preprocessing est fit sur les données incluant le test.

```python
# MAUVAIS : fit l'imputer sur TOUT le dataset
imputer.fit_transform(tout_le_dataset)
X_train, X_valid = split(tout_le_dataset)

# BON : fit seulement sur le train
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_valid = imputer.transform(X_valid)

# MEILLEUR : utiliser un Pipeline (fait ça automatiquement)
```

**Signal d'alarme** : Si ton score est > 98%, vérifie la fuite de données.

```python
# Détecter : si une feature sépare parfaitement les classes → suspect
for col in X.columns:
    if X[col].nunique() < 50:
        print(f"{col}: {X.groupby(y)[col].mean()}")
```

---

# PARTIE 7 - Feature Engineering (Créer des Variables)

> **Objectif** : Créer les features qui font la différence entre un bon et un excellent score.
> **Source** : `Feature Engineering/`

## 7.1 Mutual Information (Sélection de Features)

La **Mutual Information (MI)** mesure la dépendance entre une feature et la target. Contrairement à la corrélation, elle détecte les relations **non-linéaires**.

```python
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt

X = df.drop('target', axis=1)
y = df['target']

# Encoder les catégorielles en numérique pour le MI
for col in X.select_dtypes("object"):
    X[col], _ = X[col].factorize()

# Identifier les features discrètes
discrete_features = X.dtypes == int

# Calculer les scores MI
mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
# Pour une classification : mutual_info_classif(X, y, ...)

mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print(mi_scores)

# Visualiser
plt.figure(figsize=(10, 6))
mi_scores.sort_values().plot.barh()
plt.title("Mutual Information Scores")
plt.show()
```

> **Important** : Le MI est univarié (une feature à la fois). Une feature avec un MI faible peut quand même être utile en **interaction** avec d'autres features.

## 7.2 Créer des Features (Les 5 Techniques)

### 1. Transformations Mathématiques

```python
# Ratios (très utiles, difficiles à apprendre pour les modèles)
df['revenu_par_personne'] = df['Revenu'] / df['Taille_menage']
df['prix_m2'] = df['Prix'] / df['Surface']
df['ratio_chambre_surface'] = df['NbChambres'] / df['Surface']

# Log (pour les distributions asymétriques)
df['log_revenu'] = np.log1p(df['Revenu'])  # log1p gère les 0

# Polynômes
df['age_carre'] = df['Age'] ** 2
```

### 2. Comptages

```python
# Compter le nombre de features actives
features_binaires = ['Garage', 'Piscine', 'Jardin', 'Cave', 'Terrasse']
df['nb_equipements'] = df[features_binaires].sum(axis=1)

# Compter les non-zéro
composants = ['Ciment', 'Sable', 'Gravier', 'Eau', 'Adjuvant']
df['nb_composants'] = df[composants].gt(0).sum(axis=1)
```

### 3. Décomposer des Features

```python
# Séparer "Corporate L3" en Type et Level
df[['Type', 'Level']] = df['Policy'].str.split(' ', expand=True)

# Extraire d'un format complexe (ex: "0001_01" → Groupe et Numéro)
df['Groupe'] = df['PassengerId'].str.split('_').str[0]
df['Numero'] = df['PassengerId'].str.split('_').str[1]
```

### 4. Combiner des Features

```python
# Interaction catégorielle
df['marque_style'] = df['Marque'] + "_" + df['Style']

# Interaction numérique
df['surface_qualite'] = df['Surface'] * df['QualiteGenerale']
```

### 5. Agrégations par Groupe

```python
# Statistiques de groupe
df['revenu_moyen_ville'] = df.groupby('Ville')['Revenu'].transform('mean')
df['revenu_median_ville'] = df.groupby('Ville')['Revenu'].transform('median')
df['nb_personnes_ville'] = df.groupby('Ville')['Ville'].transform('count')

# Fréquence d'encodage
df['freq_ville'] = df.groupby('Ville')['Ville'].transform('count') / len(df)

# ATTENTION pour le train/valid split : calculer sur le TRAIN uniquement
train_agg = train.groupby('Ville')['Revenu'].mean().reset_index()
train_agg.columns = ['Ville', 'revenu_moyen_ville']
valid = valid.merge(train_agg, on='Ville', how='left')
```

## 7.3 Target Encoding

Remplace chaque catégorie par la **moyenne de la target** pour cette catégorie.

```python
from category_encoders import MEstimateEncoder

# IMPORTANT : utiliser un split séparé pour éviter le leakage
X_encode = X.sample(frac=0.25, random_state=42)  # 25% pour l'encodage
y_encode = y[X_encode.index]
X_train_restant = X.drop(X_encode.index)
y_train_restant = y[X_train_restant.index]

# Encoder avec smoothing (m contrôle le lissage)
encoder = MEstimateEncoder(cols=["Ville"], m=5.0)
encoder.fit(X_encode, y_encode)
X_train_encoded = encoder.transform(X_train_restant)
```

> **Quand utiliser le Target Encoding** :
> - Features avec BEAUCOUP de catégories (> 15, trop pour le one-hot)
> - Features avec une relation probable avec la target
> - **Toujours** avec du smoothing (paramètre `m`) et un split séparé

## 7.4 Clustering (K-Means comme Feature)

Utiliser le clustering pour créer des "groupes naturels" dans les données.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardiser (K-Means est sensible à l'échelle)
features_cluster = ['Latitude', 'Longitude', 'RevenuMedian']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_cluster])

# Créer les clusters
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Les labels sont catégoriels
df['Cluster'] = df['Cluster'].astype('category')
```

## 7.5 PCA (Analyse en Composantes Principales)

La PCA trouve les axes de **variation maximale** dans les données. Utile pour :
- Réduire la dimensionnalité
- Créer des features décorrélées
- Découvrir des structures cachées

```python
from sklearn.decomposition import PCA

features_pca = ['highway_mpg', 'engine_size', 'horsepower', 'curb_weight']
X_pca_input = df[features_pca]

# IMPORTANT : toujours standardiser avant la PCA
X_scaled = (X_pca_input - X_pca_input.mean()) / X_pca_input.std()

# Appliquer la PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Examiner la variance expliquée
print("Variance expliquée par composante:", pca.explained_variance_ratio_)
# ex: [0.68, 0.22, 0.07, 0.03] → PC1 explique 68% de la variation

# Examiner les loadings (ce que chaque composante signifie)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(features_pca))],
    index=features_pca
)
print(loadings)

# Créer des features inspirées par la PCA
# Si PC3 oppose horsepower et curb_weight → créer le ratio
df['puissance_poids'] = df['horsepower'] / df['curb_weight']
```

## 7.6 Checklist Feature Engineering

```
□ 1. Explorer et comprendre les données (EDA)
□ 2. Calculer les scores MI pour classer les features
□ 3. Gérer les valeurs manquantes (imputation)
□ 4. Encoder les catégorielles (ordinal / one-hot / target)
□ 5. Créer des features : ratios, log, comptages, interactions
□ 6. Agréger par groupe (mean, count, std par catégorie)
□ 7. Clustering (K-Means sur features spatiales ou économiques)
□ 8. PCA si beaucoup de features corrélées
□ 9. Valider avec cross-validation à chaque étape
□ 10. Vérifier l'absence de data leakage
```

---

# PARTIE 8 - Deep Learning (Réseaux de Neurones)

> **Objectif** : Comprendre et construire des réseaux de neurones avec Keras/TensorFlow.
> **Source** : `Intro to Deep Learning/`

## 8.1 Un Neurone = Régression Linéaire

Un neurone calcule : `y = w₁·x₁ + w₂·x₂ + ... + b`

```python
from tensorflow import keras
from tensorflow.keras import layers

# Un seul neurone linéaire (3 inputs, 1 output)
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```

## 8.2 Réseau Profond (Deep Neural Network)

Sans **fonction d'activation**, empiler des couches ne sert à rien (tout reste linéaire).
La fonction **ReLU** (`max(0, x)`) introduit la non-linéarité.

```python
model = keras.Sequential([
    # Couches cachées avec activation ReLU
    layers.Dense(units=512, activation='relu', input_shape=[11]),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    # Couche de sortie : PAS d'activation pour la régression
    layers.Dense(units=1),
])
```

## 8.3 Entraîner un Réseau

```python
# 1. Préparer les données (TOUJOURS normaliser les features)
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train_scaled = (df_train - min_) / (max_ - min_)
df_valid_scaled = (df_valid - min_) / (max_ - min_)  # même min/max !

X_train = df_train_scaled.drop('target', axis=1)
y_train = df_train_scaled['target']
X_valid = df_valid_scaled.drop('target', axis=1)
y_valid = df_valid_scaled['target']

# 2. Compiler (choisir la loss et l'optimiseur)
model.compile(
    optimizer='adam',      # Adam est le choix par défaut (auto-tuning)
    loss='mae',            # ou 'mse', 'huber'
)

# 3. Entraîner
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    verbose=0,
)

# 4. Visualiser les courbes d'apprentissage
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
plt.title("Courbes d'apprentissage")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```

## 8.4 Combattre l'Overfitting

### Early Stopping (TOUJOURS l'utiliser)

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001,          # amélioration minimale requise
    patience=20,              # nombre d'epochs sans amélioration avant arrêt
    restore_best_weights=True # revenir au meilleur modèle
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,               # met un nombre élevé, early stopping gère
    callbacks=[early_stopping],
    verbose=0,
)
```

### Dropout (Régularisation)

Éteint aléatoirement une fraction des neurones à chaque étape d'entraînement.

```python
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),                    # 30% des neurones désactivés
    layers.BatchNormalization(),             # normalise les activations
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])
```

### Batch Normalization

Normalise les sorties de chaque couche. Accélère l'entraînement et stabilise le réseau.

> **Règle** : Quand tu ajoutes du Dropout, augmente le nombre de neurones pour compenser.

## 8.5 Classification Binaire

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[33]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid'),   # Sigmoid → probabilité [0, 1]
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',              # PAS 'mae' pour la classification
    metrics=['binary_accuracy'],
)
```

| Type de problème | Loss | Activation finale | Unités finales |
|-----------------|------|-------------------|----------------|
| Régression | `mae` ou `mse` | Aucune (linéaire) | 1 |
| Classification binaire | `binary_crossentropy` | `sigmoid` | 1 |
| Classification multi-classe | `categorical_crossentropy` | `softmax` | N classes |

---

# PARTIE 9 - Computer Vision (Images & CNN)

> **Objectif** : Construire des modèles de classification d'images.
> **Source** : `Computer Vision/`, `Guides/Transfer Learning for CV Guide/`

## 9.1 Architecture CNN

Un CNN a deux parties :
1. **Base convolutionnelle** : extrait les features (lignes, textures, formes, objets)
2. **Tête dense** : classifie à partir des features extraites

```
Image → [Conv → ReLU → Pool] × N → Flatten → Dense → Prediction
         Base convolutionnelle            Tête dense
```

## 9.2 Transfer Learning (L'Approche Recommandée)

Au lieu de tout entraîner from scratch, on réutilise un modèle pré-entraîné sur ImageNet.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. Charger le modèle pré-entraîné SANS sa tête
base_model = ResNet50(
    include_top=False,        # retirer la couche de classification
    pooling='avg',            # global average pooling → vecteur 1D
    weights='imagenet'        # poids pré-entraînés sur ImageNet
)

# 2. GELER la base (on ne modifie pas les poids pré-entraînés)
base_model.trainable = False

# 3. Ajouter une nouvelle tête
model = Sequential([
    base_model,
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')    # 2 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Charger les images** :

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = data_gen.flow_from_directory(
    'data/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_gen = data_gen.flow_from_directory(
    'data/valid/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.fit(train_gen, validation_data=valid_gen, epochs=10)
```

## 9.3 Data Augmentation

Crée des variations des images d'entraînement pour améliorer la généralisation.

```python
from tensorflow.keras.layers.experimental import preprocessing

model = Sequential([
    # Augmentation (active seulement pendant l'entraînement)
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomContrast(0.3),
    preprocessing.RandomZoom(0.2),
    # Base pré-entraînée
    base_model,
    # Tête
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])
```

> **Modèles pré-entraînés populaires** :
> - **ResNet50** : bon compromis vitesse/précision
> - **EfficientNetB0-B7** : état de l'art en efficacité
> - **VGG16** : simple, bon pour apprendre

---

# PARTIE 10 - NLP (Traitement du Langage Naturel)

> **Objectif** : Travailler avec du texte (classification, sentiment, etc.).
> **Source** : `Guides/Natural Language Processing Guide/`

## 10.1 Preprocessing du Texte

```python
import re, string, nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

def nettoyer_texte(text):
    """Pipeline de nettoyage standard pour du texte."""
    text = str(text).lower()                                    # minuscules
    text = re.sub(r'\[.*?\]', '', text)                        # retirer [crochets]
    text = re.sub(r'https?://\S+|www\.\S+', '', text)         # retirer URLs
    text = re.sub(r'<.*?>', '', text)                          # retirer HTML
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # retirer ponctuation
    text = re.sub(r'\w*\d\w*', '', text)                       # retirer mots avec chiffres
    # Retirer stopwords et stemmer
    text = ' '.join(stemmer.stem(word) for word in text.split()
                    if word not in stop_words)
    return text

df['texte_propre'] = df['texte'].apply(nettoyer_texte)
```

## 10.2 Approche Classique : TF-IDF + Modèle ML

**TF-IDF** (Term Frequency - Inverse Document Frequency) : donne plus de poids aux mots rares et discriminants.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Pipeline TF-IDF + Naive Bayes (excellent baseline rapide)
pipe_nb = Pipeline([
    ('bow', CountVectorizer()),        # texte → matrice de comptages
    ('tfidf', TfidfTransformer()),     # comptages → poids TF-IDF
    ('model', MultinomialNB())         # classification
])

pipe_nb.fit(X_train_text, y_train)
predictions = pipe_nb.predict(X_test_text)

# Pipeline TF-IDF + XGBoost (souvent meilleur)
import xgboost as xgb

pipe_xgb = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', xgb.XGBClassifier(
        learning_rate=0.1, max_depth=7, n_estimators=80,
        use_label_encoder=False, eval_metric='auc'
    ))
])
```

## 10.3 Approche Deep Learning : Embeddings + LSTM

```python
from keras.models import Sequential
from keras.layers import (Embedding, Bidirectional, LSTM,
                          GlobalMaxPool1D, Dense, Dropout, BatchNormalization)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tokeniser et padder
max_length = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_text)
vocab_size = len(tokenizer.word_index) + 1

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text),
                            maxlen=max_length, padding='post')

# Modèle BiLSTM
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
    Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.2)),
    GlobalMaxPool1D(),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_seq, y_train, epochs=7, batch_size=32,
          validation_split=0.2)
```

## 10.4 Approche Moderne : Transformers (BERT / DeBERTa)

L'approche la plus performante aujourd'hui : fine-tuner un modèle pré-entraîné.

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import Dataset

# 1. Choisir le modèle
model_name = 'microsoft/deberta-v3-small'  # excellent rapport qualité/taille
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Tokeniser les données
def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

dataset = Dataset.from_pandas(df)
tokenized = dataset.map(tokenize, batched=True)

# 3. Split train/validation
split = tokenized.train_test_split(0.2, seed=42)

# 4. Configurer l'entraînement
args = TrainingArguments(
    output_dir='outputs',
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    fp16=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
)

# 5. Entraîner
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
trainer = Trainer(
    model, args,
    train_dataset=split['train'],
    eval_dataset=split['test'],
    tokenizer=tokenizer,
)
trainer.train()
```

> **Hiérarchie de performance NLP** :
> 1. **BERT / DeBERTa** (meilleur mais lent) : ~98% accuracy
> 2. **GloVe + BiLSTM** (bon compromis) : ~97% accuracy
> 3. **TF-IDF + XGBoost** (rapide, bon baseline) : ~85% accuracy
> 4. **TF-IDF + Naive Bayes** (le plus rapide) : ~80% accuracy

---

# PARTIE 11 - Interpréter & Expliquer ses Modèles

> **Objectif** : Comprendre POURQUOI un modèle fait ses prédictions.
> **Source** : `Machine Learning Explainability/`

## 11.1 Permutation Importance ("Quelles features comptent ?")

Principe : mélanger les valeurs d'une feature et mesurer la chute de performance.

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())
```

> Toujours calculer sur les données de **validation**, pas d'entraînement.

## 11.2 Partial Dependence Plots ("Comment une feature affecte les prédictions ?")

```python
from sklearn.inspection import PartialDependenceDisplay

# Effet d'une seule feature
PartialDependenceDisplay.from_estimator(model, val_X, ['Age'])
plt.show()

# Interaction entre 2 features
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(model, val_X, [('Age', 'Revenu')], ax=ax)
plt.show()
```

## 11.3 SHAP Values ("Pourquoi CETTE prédiction spécifique ?")

```python
import shap

# Pour les modèles à base d'arbres (rapide et exact)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(val_X)

# Décomposition d'une prédiction individuelle
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], val_X.iloc[0])

# Vue globale : importance de chaque feature
shap.summary_plot(shap_values, val_X)
```

> **Pipeline d'explicabilité** :
> 1. **Quoi ?** → Permutation Importance (features globalement importantes)
> 2. **Comment ?** → PDP (direction et forme de l'effet)
> 3. **Pourquoi ?** → SHAP (explication d'une prédiction individuelle)

---

# PARTIE 12 - Stratégie Compétitions Kaggle

> **Objectif** : Maximiser ton score dans les compétitions Kaggle.
> **Source** : `Guides/Kaggle Competitions Guide/`, expérience accumulée

## 12.1 Workflow d'une Compétition

```
Jour 1-2:                    Jours 3-5:                   Jours 6+:
┌─────────────┐             ┌─────────────┐              ┌─────────────┐
│ 1. Lire les │             │ 4. Feature   │              │ 7. Ensemble  │
│    règles   │             │    Engineering│              │    de modèles│
│ 2. EDA      │      →      │ 5. Modèles   │      →       │ 8. Post-     │
│    complète │             │    avancés   │              │    processing│
│ 3. Baseline │             │ 6. Tuning    │              │ 9. Sélection │
│    simple   │             │              │              │    finale    │
└─────────────┘             └─────────────┘              └─────────────┘
```

## 12.2 Étape par Étape

### Étape 1 : Comprendre le Problème

```python
# 1. Lire la description de la compétition
# 2. Comprendre la MÉTRIQUE d'évaluation (c'est ce qu'il faut optimiser !)
# 3. Lire les données : data description, train.csv, test.csv
# 4. Regarder le sample_submission.csv pour le format attendu

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target: {train['target'].describe()}")
print(f"Format soumission: {sample_sub.head()}")
```

### Étape 2 : EDA Rapide

```python
# Distribution de la target
sns.histplot(train['target'])
plt.title("Distribution de la variable cible")
plt.show()

# Valeurs manquantes
print(train.isnull().sum().sort_values(ascending=False).head(20))

# Types de colonnes
print(f"Numériques: {train.select_dtypes(include='number').columns.tolist()}")
print(f"Catégorielles: {train.select_dtypes(include='object').columns.tolist()}")

# Corrélations
top_corr = train.corr()['target'].abs().sort_values(ascending=False).head(20)
print(top_corr)
```

### Étape 3 : Baseline Simple (Soumettre en < 1 heure)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # ou Regressor
from sklearn.metrics import accuracy_score

# Preprocessing minimal
features = train.select_dtypes(include='number').columns.drop('target').tolist()
X = train[features].fillna(-999)
y = train['target']
X_test = test[features].fillna(-999)

# Train/valid split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle simple
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Validation: {accuracy_score(y_val, model.predict(X_val)):.4f}")

# Soumission
predictions = model.predict(X_test)
submission = pd.DataFrame({'Id': test['Id'], 'target': predictions})
submission.to_csv('submission_baseline.csv', index=False)
```

### Étape 4 : Améliorer Itérativement

```python
# Template de pipeline complète pour itérer rapidement
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Colonnes
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = [c for c in X.columns if X[c].dtype == 'object' and X[c].nunique() < 15]

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols),
])

# Pipeline complète
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Étape 5 : Ensemble de Modèles

La technique la plus fiable pour gagner quelques points.

```python
import numpy as np

# Entraîner plusieurs modèles différents
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    'rf': RandomForestClassifier(n_estimators=300, random_state=42),
    'et': ExtraTreesClassifier(n_estimators=300, random_state=42),
    'xgb': XGBClassifier(n_estimators=1000, learning_rate=0.05, random_state=42),
    'lgbm': LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42),
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]  # probabilité classe 1
    predictions[name] = preds
    print(f"{name}: entraîné")

# Moyenne simple (souvent la meilleure approche)
ensemble_preds = np.mean(list(predictions.values()), axis=0)
ensemble_labels = (ensemble_preds > 0.5).astype(int)

# Soumission
submission = pd.DataFrame({'Id': test['Id'], 'target': ensemble_labels})
submission.to_csv('submission_ensemble.csv', index=False)
```

## 12.3 Les Règles d'Or

### Suivi CV vs LB

```python
# TOUJOURS tracker le rapport entre ton score CV et ton score Leaderboard
# Sain : CV ≈ LB (petit écart)
# Danger : CV >> LB (overfitting sur le CV)
# Danger : CV << LB (overfitting sur le LB, ou chance)

historique = pd.DataFrame({
    'Version': ['V1', 'V2', 'V3', 'V4'],
    'CV': [0.800, 0.810, 0.820, 0.815],
    'LB': [0.795, 0.807, 0.808, 0.812],
    'Gap': [0.005, 0.003, 0.012, 0.003],
})
print(historique)
# V3 a un gap de 0.012 → overfit probable, se méfier
```

### Régularisation

- **Commencer fort** : learning_rate bas, max_depth limité, régularisation haute
- Ajouter de la complexité **graduellement**
- "Less is more" : 29 features bien choisies > 56 features bruitées

### Features vs Hyperparamètres

- Le **feature engineering** a plus d'impact que le **tuning** des hyperparamètres
- Passe 70% de ton temps sur les features, 20% sur le modèle, 10% sur le tuning
- Un bon feature engineering peut valoir plus que changer de modèle

### Ensemble

- La diversité est clé : mélange différents TYPES de modèles (XGBoost + LightGBM + RF + ExtraTrees)
- Moyenne simple > optimisation des poids (surtout sur petits datasets)
- Vérifie la corrélation entre tes modèles : < 0.97 = bonne diversité

## 12.4 Les Pièges à Éviter

| Piège | Solution |
|-------|----------|
| Overfitting au LB public | Se fier au CV, le LB privé est le vrai test |
| Trop de features | Commencer simple, ajouter chirurgicalement |
| Optuna sur petit dataset | Les params auto-tunés overfittent. Préférer des params manuels |
| Ne pas lire les discussions | Les top solutions partagent souvent des insights précieux |
| Group features entre train/test | Vérifier si les groupes sont partagés avant de créer des features |
| Ignorer la métrique | Optimiser la métrique de la compétition, pas l'accuracy par défaut |

---

# ANNEXES - Cheat Sheets & Templates

## A. Imports Standards

```python
# === DATA ===
import numpy as np
import pandas as pd

# === VISUALISATION ===
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline

# === PREPROCESSING ===
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   OrdinalEncoder, OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === MODÈLES ===
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# === MÉTRIQUES ===
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)

# === FEATURE ENGINEERING ===
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from category_encoders import MEstimateEncoder
```

## B. Template Pipeline Complète (Copier-Coller)

```python
"""
Template pour compétition Kaggle - Classification binaire
Remplacer : 'target', les noms de colonnes, et les paramètres
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# === 1. CHARGER ===
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y = train['target']
X = train.drop(['target', 'Id'], axis=1)
X_test = test.drop(['Id'], axis=1)

# === 2. IDENTIFIER LES COLONNES ===
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = [c for c in X.columns
            if X[c].dtype == 'object' and X[c].nunique() < 15]

# === 3. PIPELINE ===
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols),
])

model = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# === 4. CROSS-VALIDATION ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# === 5. ENTRAÎNER SUR TOUT ET SOUMETTRE ===
pipeline.fit(X, y)
predictions = pipeline.predict(X_test)

submission = pd.DataFrame({
    'Id': test['Id'],
    'target': predictions
})
submission.to_csv('submission.csv', index=False)
print(f"Soumission créée : {submission.shape}")
```

## C. Métriques Courantes

| Métrique | Type | Usage | Sklearn |
|----------|------|-------|---------|
| MAE | Régression | Erreur moyenne absolue | `mean_absolute_error` |
| RMSE | Régression | Pénalise les grosses erreurs | `np.sqrt(mean_squared_error(...))` |
| Accuracy | Classification | % correct (classes équilibrées) | `accuracy_score` |
| F1-Score | Classification | Équilibre précision/rappel (classes déséquilibrées) | `f1_score` |
| AUC-ROC | Classification | Qualité du ranking (probabilités) | `roc_auc_score` |
| Log Loss | Classification | Qualité des probabilités | `log_loss` |

## D. Quand Utiliser Quel Modèle

| Situation | Modèle recommandé |
|-----------|-------------------|
| Baseline rapide | Random Forest |
| Données tabulaires (le plus courant sur Kaggle) | XGBoost ou LightGBM |
| Petit dataset (< 1000 lignes) | Random Forest + forte régularisation |
| Images | CNN + Transfer Learning (ResNet, EfficientNet) |
| Texte | Transformers (BERT, DeBERTa) |
| Séries temporelles | XGBoost avec features de lag + features calendaires |
| Ensemble final | Moyenne de XGBoost + LightGBM + ExtraTrees |

## E. Checklist Avant Soumission

```
□ EDA complète (distributions, corrélations, valeurs manquantes)
□ Pas de data leakage (features disponibles au moment de la prédiction)
□ Preprocessing dans un Pipeline
□ Cross-validation stable (faible variance entre folds)
□ Score CV cohérent avec le score LB
□ Essayé au moins 3 types de modèles différents
□ Feature engineering itératif (ajouter une feature → valider → garder ou jeter)
□ Ensemble de modèles diversifiés
□ Fichier submission.csv au bon format
□ 2 soumissions sélectionnées pour le classement final
```

---

> **Rappel final** : Le Machine Learning, c'est 80% de préparation des données et 20% de modélisation. Le feature engineering fait la différence entre un bon score et un excellent score. Commence simple, itère souvent, et fais confiance à ta cross-validation.

---

*Guide généré à partir des cours Kaggle Learn et des guides avancés du dossier `learn-kaggle/`.*
