# learn-kaggle - Guide de structure complet

> Ce document decrit l'ensemble des dossiers et fichiers du repertoire `learn-kaggle`.
> Il est concu pour permettre a un agent de code de localiser rapidement les ressources dont il a besoin.

---

## Vue d'ensemble

Le dossier `learn-kaggle` contient l'ensemble des cours Kaggle Learn organises par theme. Chaque dossier de premier niveau correspond a un cours ou une categorie. Les notebooks (`.ipynb`) sont les fichiers principaux contenant le code, les explications et les exercices.

### Arborescence de premier niveau

```
learn-kaggle/
├── Python/                                  # Cours Python fondamental
├── Intro to Programming/                    # Introduction a la programmation
├── Pandas/                                  # Manipulation de donnees avec Pandas
├── Intro to SQL/                            # SQL fondamental avec BigQuery
├── Advanced SQL/                            # SQL avance
├── Data Cleaning/                           # Nettoyage de donnees
├── Data Visualization/                      # Visualisation de donnees (Seaborn)
├── Intro to Machine Learning/               # ML fondamental
├── Intermediate Machine Learning/           # ML intermediaire
├── Intro to Deep Learning/                  # Deep Learning fondamental
├── Computer Vision/                         # Vision par ordinateur (CNN)
├── Feature Engineering/                     # Ingenierie de features
├── Machine Learning Explainability/         # Interpretabilite des modeles
├── Geospatial Analysis/                     # Analyse geospatiale
├── Time Series/                             # Series temporelles
├── Intro to Game AI and Reinforcement Learning/  # IA de jeu et RL
├── Intro to AI Ethics/                      # Ethique de l'IA
├── Guides/                                  # Guides avances et specialises
│   ├── JAX Guide/
│   ├── TensorFlow Guide/
│   ├── Natural Language Processing Guide/
│   ├── R Guide/
│   ├── Transfer Learning for CV Guide/
│   └── Kaggle Competitions Guide/
└── GUIDE-STRUCTURE.md                       # Ce fichier
```

---

## 1. Python/

**Theme** : Bases du langage Python
**Contenu** : 7 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Hello, Python/` | `hello-python.ipynb` | Syntaxe Python de base : variables, operateurs, types primitifs, print |
| `Functions and Getting Help/` | `functions-and-getting-help.ipynb` | Definition de fonctions, parametres, valeurs de retour, `help()`, docstrings |
| `Booleans and Conditionals/` | `hello-python.ipynb` | Booleens, operateurs de comparaison, `if/elif/else`, operateurs logiques |
| `Lists and Tuples/` | `lists.ipynb` | Listes (creation, indexation, slicing, methodes), tuples, differences |
| `Loops and List Comprehensions/` | `loops-and-list-comprehensions.ipynb` | Boucles `for/while`, `range()`, list comprehensions, syntaxe concise |
| `Strings and Dictionaries/` | `strings-and-dictionaries.ipynb` | Manipulation de chaines, formatage, methodes string, dictionnaires |
| `Working with External Libraries/` | `working-with-external-libraries.ipynb` | `import`, alias, sous-modules, exploration de bibliotheques externes |

**Utiliser quand** : besoin de rappels sur la syntaxe Python pure, sans librairie specifique.

---

## 2. Intro to Programming/

**Theme** : Introduction a la programmation pour debutants absolus
**Contenu** : 6 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Arithmetic and Variables/` | `arithmetic-and-variables.ipynb` | Operations mathematiques de base (+, -, *, /), affectation de variables |
| `Functions/` | `functions.ipynb` | Concept de fonction, appels, parametres simples |
| `Data Types/` | `data-types.ipynb` | Types int, float, str, bool, conversions de types |
| `Conditions and Conditional Statements/` | `conditions-and-conditional-statements.ipynb` | Conditions if/else, comparaisons, logique booleenne |
| `Intro to Lists/` | `intro-to-lists.ipynb` | Listes : creation, acces, modification, longueur |
| `Bonus Lessons/` | `titanic-tutorial.ipynb` | Tutoriel bonus applique sur le dataset Titanic |

**Utiliser quand** : besoin d'explications tres basiques pour quelqu'un qui n'a jamais programme.

---

## 3. Pandas/

**Theme** : Manipulation et analyse de donnees avec la librairie Pandas
**Contenu** : 6 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Creating, Reading and Writing/` | `creating-reading-and-writing.ipynb` | Creer des DataFrame/Series, lire CSV/Excel, ecrire des fichiers |
| `Indexing, Selecting & Assigning/` | `indexing-selecting-assigning.ipynb` | `.loc`, `.iloc`, selection conditionnelle, assignation de valeurs |
| `Summary Functions and Maps/` | `summary-functions-and-maps.ipynb` | `describe()`, `mean()`, `value_counts()`, `map()`, `apply()` |
| `Grouping and Sorting/` | `grouping-and-sorting.ipynb` | `groupby()`, `agg()`, `sort_values()`, `sort_index()` |
| `Data Types and Missing Values/` | `data-types-and-missing-values.ipynb` | `dtypes`, `astype()`, `isnull()`, `fillna()`, `dropna()` |
| `Renaming and Combining/` | `renaming-and-combining.ipynb` | `rename()`, `concat()`, `merge()`, `join()` |

**Utiliser quand** : besoin de manipuler, filtrer, agreger ou transformer des DataFrames.

---

## 4. Intro to SQL/

**Theme** : Fondamentaux SQL avec Google BigQuery
**Contenu** : 6 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Getting Started With SQL and BigQuery/` | `getting-started-with-sql-and-bigquery.ipynb` | Configuration BigQuery, `Client()`, structure des datasets/tables |
| `Select, From & Where/` | `select-from-where.ipynb` | Requetes SELECT de base, filtrage WHERE, operateurs de comparaison |
| `Group By, Having & Count/` | `group-by-having-count.ipynb` | Agregation GROUP BY, filtrage HAVING, COUNT, SUM, AVG |
| `Order By/` | `order-by.ipynb` | Tri des resultats, ASC/DESC, tri multi-colonnes |
| `As & With/` | `as-with.ipynb` | Alias AS, Common Table Expressions (CTE) avec WITH |
| `Joining Data/` | `joining-data.ipynb` | JOIN (INNER, LEFT, RIGHT), cles de jointure, relations entre tables |

**Utiliser quand** : besoin d'ecrire des requetes SQL basiques ou d'utiliser BigQuery.

---

## 5. Advanced SQL/

**Theme** : Techniques SQL avancees
**Contenu** : 4 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `JOINs and UNIONs/` | `joins-and-unions.ipynb` | Jointures avancees (FULL, CROSS), UNION, UNION ALL |
| `Analytic Functions/` | `analytic-functions.ipynb` | Fonctions fenetres (OVER, PARTITION BY, ROW_NUMBER, RANK, LAG, LEAD) |
| `Nested and Repeated Data/` | `nested-and-repeated-data.ipynb` | Donnees imbriquees (STRUCT), donnees repetees (ARRAY), UNNEST |
| `Writing Efficient Queries/` | `writing-efficient-queries.ipynb` | Optimisation de requetes, estimation de couts, bonnes pratiques |

**Utiliser quand** : besoin de fonctions fenetres, donnees imbriquees, ou optimisation SQL.

---

## 6. Data Cleaning/

**Theme** : Nettoyage et preparation des donnees
**Contenu** : 5 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Handling Missing Values/` | `handling-missing-values.ipynb` | Detection de NaN, strategies (suppression, imputation), `fillna()`, `dropna()` |
| `Scaling and Normalization/` | `scaling-and-normalization.ipynb` | MinMaxScaler, StandardScaler, normalisation, quand utiliser chacun |
| `Parsing Dates/` | `parsing-dates.ipynb` | Conversion de chaines en datetime, `pd.to_datetime()`, extraction jour/mois/annee |
| `Character Encodings/` | `character-encodings.ipynb` | UTF-8, ASCII, Latin-1, detection d'encodage avec `chardet`, lecture de fichiers |
| `Inconsistent Data Entry/` | `inconsistent-data-entry.ipynb` | Nettoyage de texte (espaces, casse), fuzzy matching avec `fuzzywuzzy` |

**Utiliser quand** : les donnees sont sales (valeurs manquantes, encodages, dates mal formatees, doublons).

---

## 7. Data Visualization/

**Theme** : Visualisation de donnees avec Seaborn et Matplotlib
**Contenu** : 8 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Hello, Seaborn/` | `hello-seaborn.ipynb` | Introduction a Seaborn, configuration de base, premiers graphiques |
| `Line Charts/` | `line-charts.ipynb` | Graphiques en lignes, `sns.lineplot()`, tendances temporelles |
| `Bar Charts and Heatmaps/` | `bar-charts-and-heatmaps.ipynb` | Diagrammes a barres (`sns.barplot()`), heatmaps (`sns.heatmap()`) |
| `Scatter Plots/` | `scatter-plots.ipynb` | Nuages de points, `sns.scatterplot()`, `sns.regplot()`, correlations |
| `Distributions/` | `distributions.ipynb` | Histogrammes, KDE, `sns.histplot()`, `sns.kdeplot()` |
| `Choosing Plot Types and Custom Styles/` | `choosing-plot-types-and-custom-styles.ipynb` | Choix du bon type de graphique, themes Seaborn, personnalisation |
| `Final Project/` | `final-project.ipynb` | Projet de synthese combinant plusieurs types de visualisations |
| `Creating Your Own Notebooks/` | `creating-your-own-notebooks.ipynb` | Creation et configuration de notebooks Kaggle |

**Utiliser quand** : besoin de creer des graphiques, explorer visuellement des donnees, ou choisir un type de visualisation.

---

## 8. Intro to Machine Learning/

**Theme** : Fondamentaux du Machine Learning
**Contenu** : 7 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `How Models Work/` | `how-models-work.ipynb` | Concept d'arbre de decision, comment un modele fait des predictions |
| `Basic Data Exploration/` | `basic-data-exploration.ipynb` | EDA : `shape`, `describe()`, `head()`, `dtypes`, comprehension des donnees |
| `Your First Machine Learning Model/` | `your-first-machine-learning-model.ipynb` | Pipeline complet : selection features, `DecisionTreeRegressor`, `.fit()`, `.predict()` |
| `Model Validation/` | `model-validation.ipynb` | Train/test split, MAE, pourquoi valider, `train_test_split()` |
| `Underfitting and Overfitting/` | `underfitting-and-overfitting.ipynb` | Compromis biais-variance, `max_leaf_nodes`, complexite du modele |
| `Random Forests/` | `random-forests.ipynb` | `RandomForestRegressor`, ensembles, pourquoi les forets sont meilleures |
| `Machine Learning Competitions/` | `machine-learning-competitions.ipynb` | Soumission Kaggle, workflow de competition, fichier submission.csv |

**Utiliser quand** : besoin de construire un premier modele ML, comprendre train/test split, ou utiliser scikit-learn basique.

---

## 9. Intermediate Machine Learning/

**Theme** : Techniques ML intermediaires
**Contenu** : 7 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Introduction/` | `introduction.ipynb` | Vue d'ensemble du cours, rappels, configuration |
| `Missing Values/` | `missing-values.ipynb` | 3 approches : suppression, imputation simple, imputation avancee (`SimpleImputer`) |
| `Categorical Variables/` | `categorical-variables.ipynb` | Ordinal Encoding, One-Hot Encoding, `OrdinalEncoder`, `OneHotEncoder` |
| `Pipelines/` | `pipelines.ipynb` | `Pipeline`, `ColumnTransformer`, workflows reproductibles avec sklearn |
| `Cross-Validation/` | `cross-validation.ipynb` | K-Fold CV, `cross_val_score()`, choix du nombre de folds |
| `XGBoost/` | `xgboost.ipynb` | `XGBRegressor`, gradient boosting, `n_estimators`, `learning_rate`, `early_stopping_rounds` |
| `Data Leakage/` | `data-leakage.ipynb` | Target leakage, train-test contamination, comment les identifier |

**Utiliser quand** : besoin de gerer des variables categoriques, construire des pipelines sklearn, utiliser XGBoost, ou faire de la cross-validation.

---

## 10. Intro to Deep Learning/

**Theme** : Fondamentaux du Deep Learning avec Keras/TensorFlow
**Contenu** : 7 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `A Single Neuron/` | `a-single-neuron.ipynb` | Perceptron, poids, biais, `keras.Sequential`, couche `Dense` |
| `Deep Neural Networks/` | `deep-neural-networks.ipynb` | Reseaux multi-couches, couches cachees, representation hierarchique |
| `Stochastic Gradient Descent/` | `stochastic-gradient-descent.ipynb` | Fonction de perte, SGD, learning rate, epochs, batches |
| `Overfitting and Underfitting/` | `overfitting-and-underfitting.ipynb` | Early stopping, capacite du reseau, courbes d'apprentissage |
| `Dropout and Batch Normalization/` | `dropout-and-batch-normalization.ipynb` | `Dropout`, `BatchNormalization`, regularisation des reseaux |
| `Binary Classification/` | `binary-classification.ipynb` | Sigmoid, cross-entropy, classification binaire avec reseaux de neurones |
| `Detecting the Higgs Boson With TPUs/` | `detecting-the-higgs-boson-with-tpus.ipynb` | Entrainement sur TPU, donnees de physique des particules |

**Utiliser quand** : besoin de comprendre les reseaux de neurones, construire un modele Keras, ou gerer overfitting/underfitting en DL.

---

## 11. Computer Vision/

**Theme** : Vision par ordinateur avec CNN (Convolutional Neural Networks)
**Contenu** : 9 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `The Convolutional Classifier/` | `the-convolutional-classifier.ipynb` | Architecture CNN complete, classification d'images, base + tete |
| `Convolution and ReLU/` | `convolution-and-relu.ipynb` | Couches de convolution, filtres/noyaux, activation ReLU, detection de motifs |
| `Maximum Pooling/` | `maximum-pooling.ipynb` | MaxPooling, reduction de dimensions, invariance a la translation |
| `The Sliding Window/` | `the-sliding-window.ipynb` | Champ receptif, fenetre glissante, stride, padding |
| `Custom Convnets/` | `custom-convnets.ipynb` | Construction de CNN personnalises, empilage de couches conv |
| `Data Augmentation/` | `data-augmentation.ipynb` | Augmentation d'images (rotation, flip, zoom), `ImageDataGenerator` |
| `Transfer Learning/` | `transfer-learning.ipynb` | Modeles pre-entraines (VGG16, ResNet), fine-tuning, feature extraction |
| `Create Your First Submission/` | `create-your-first-submission.ipynb` | Soumission d'une competition de vision Kaggle, workflow complet |
| `Getting Started TPUs + Cassava Leaf Disease/` | `getting-started-tpus-cassava-leaf-disease.ipynb` | Classification d'images sur TPU, maladie des feuilles de manioc |

**Utiliser quand** : besoin de travailler avec des images, construire un CNN, ou faire du transfer learning pour la vision.

---

## 12. Feature Engineering/

**Theme** : Creation et selection de features pour ameliorer les modeles
**Contenu** : 6 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `What Is Feature Engineering/` | `what-is-feature-engineering.ipynb` | Concept, importance, impact sur les performances, mutual information |
| `Mutual Information/` | `mutual-information.ipynb` | Score MI, selection de features, `mutual_info_regression`, `mutual_info_classif` |
| `Creating Features/` | `creating-features.ipynb` | Transformations mathematiques, interactions, comptages, decompositions |
| `Clustering With K-Means/` | `clustering-with-k-means.ipynb` | K-Means comme outil de feature engineering, distance aux centroides |
| `Principal Component Analysis/` | `principal-component-analysis.ipynb` | PCA, reduction de dimensionnalite, variance expliquee, composantes |
| `Target Encoding/` | `target-encoding.ipynb` | Encodage par la cible, smoothing, risques de leakage, `TargetEncoder` |

**Utiliser quand** : besoin de creer de nouvelles features, selectionner les meilleures, ou reduire la dimensionnalite.

---

## 13. Machine Learning Explainability/

**Theme** : Interpretabilite et explicabilite des modeles ML
**Contenu** : 5 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Use Cases for Model Insights/` | `use-cases-for-model-insights.ipynb` | Pourquoi l'interpretabilite est importante, cas d'usage concrets |
| `Permutation Importance/` | `permutation-importance.ipynb` | Importance par permutation, `eli5`, identification des features cles |
| `Partial Plots/` | `partial-plots.ipynb` | Partial Dependence Plots (PDP), effet marginal d'une feature |
| `SHAP Values/` | `shap-values.ipynb` | Valeurs SHAP, contribution de chaque feature a une prediction |
| `Advanced Uses of SHAP Values/` | `advanced-uses-of-shap-values.ipynb` | Summary plots, dependence plots, force plots, analyses avancees |

**Utiliser quand** : besoin d'expliquer les predictions d'un modele, comprendre quelles features comptent, ou generer des visualisations d'interpretabilite.

---

## 14. Geospatial Analysis/

**Theme** : Analyse de donnees geospatiales avec GeoPandas et Folium
**Contenu** : 6 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Your First Map/` | `your-first-map.ipynb` | GeoPandas basique, `GeoDataFrame`, lecture de shapefiles, premiere carte |
| `Coordinate Reference Systems/` | `coordinate-reference-systems.ipynb` | Systemes de coordonnees (CRS), EPSG, re-projection, `to_crs()` |
| `Interactive Maps/` | `interactive-maps.ipynb` | Cartes interactives avec Folium, marqueurs, choroplethes |
| `Manipulating Geospatial Data/` | `manipulating-geospatial-data.ipynb` | Operations sur GeoDataFrame, geocodage, jointures spatiales |
| `Proximity Analysis/` | `proximity-analysis.ipynb` | Calcul de distances, buffers, analyses de proximite |
| `US Vaccine Tracker/` | `us-vaccine-tracker.ipynb` | Projet applique : suivi de vaccination US sur carte |

**Utiliser quand** : besoin de travailler avec des coordonnees GPS, creer des cartes, ou faire des analyses spatiales.

---

## 15. Time Series/

**Theme** : Analyse et prevision de series temporelles
**Contenu** : 6 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Linear Regression With Time Series/` | `linear-regression-with-time-series.ipynb` | Regression lineaire sur series temporelles, features de temps, lag features |
| `Trend/` | `trend.ipynb` | Detection de tendance, detrending, `DeterministicProcess` |
| `Seasonality/` | `seasonality.ipynb` | Saisonnalite, indicateurs saisonniers, series de Fourier |
| `Time Series as Features/` | `time-series-as-features.ipynb` | Lags, moyennes mobiles, features auto-regressives |
| `Hybrid Models/` | `hybrid-models.ipynb` | Combinaison modeles statistiques + ML, residus, ensembles |
| `Forecasting With Machine Learning/` | `forecasting-with-machine-learning.ipynb` | Prevision avec XGBoost/RandomForest, validation temporelle |

**Utiliser quand** : besoin de prevoir des valeurs futures, detecter des tendances/saisonnalite, ou creer des features temporelles.

---

## 16. Intro to Game AI and Reinforcement Learning/

**Theme** : Intelligence artificielle pour les jeux et apprentissage par renforcement
**Contenu** : 4 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Play the Game/` | `play-the-game.ipynb` | Introduction au jeu Connect Four, environnement Kaggle, regles |
| `One-Step Lookahead/` | `one-step-lookahead.ipynb` | Agent minimax a 1 coup, evaluation heuristique, strategie gloutonne |
| `N-Step Lookahead/` | `n-step-lookahead.ipynb` | Minimax a N coups, elagage alpha-beta, profondeur de recherche |
| `Deep Reinforcement Learning/` | `deep-reinforcement-learning.ipynb` | DQN, reseaux de neurones pour RL, entrainement d'agents |

**Utiliser quand** : besoin de creer des agents de jeu, comprendre minimax, ou implementer du reinforcement learning.

---

## 17. Intro to AI Ethics/

**Theme** : Ethique et responsabilite en intelligence artificielle
**Contenu** : 5 sous-dossiers, chacun avec 1 notebook

| Sous-dossier | Fichier | Description |
|---|---|---|
| `Introduction to AI Ethics/` | `introduction-to-ai-ethics.ipynb` | Vue d'ensemble de l'ethique IA, enjeux, cadre de reflexion |
| `Human-Centered Design for AI/` | `human-centered-design-for-ai.ipynb` | Conception centree sur l'humain, impact social, considerations design |
| `Identifying Bias in AI/` | `identifying-bias-in-ai.ipynb` | Types de biais (historique, representation, mesure), detection |
| `AI Fairness/` | `ai-fairness.ipynb` | Metriques d'equite, trade-offs, equite de groupe vs individuelle |
| `Model Cards/` | `model-cards.ipynb` | Documentation de modeles, limites, cas d'usage, transparence |

**Utiliser quand** : besoin de reflexion ethique sur un projet ML, documentation de modeles, ou analyse de biais.

---

## 18. Guides/

Le dossier `Guides/` contient des ressources avancees et specialisees organisees par technologie ou domaine.

### 18.1. Guides/JAX Guide/

**Theme** : Framework JAX pour le calcul numerique et le ML
**Structure** : `Beginner/` + `Advanced/`

| Niveau | Fichier | Description |
|---|---|---|
| `Beginner/` | `tf-jax-tutorials-part1.ipynb` | Introduction a JAX, comparaison avec TensorFlow, bases de JAX |
| `Advanced/` | `bert-tpus-jax-huggingface.ipynb` | BERT sur TPU avec JAX et HuggingFace Transformers |
| `Advanced/` | `getting-started-with-nlp-using-jax.ipynb` | NLP avec JAX, traitement de texte, modeles de langue |
| `Advanced/` | `sentiment-clf-jax-flax-on-tpus-w-b.ipynb` | Classification de sentiments avec JAX/Flax sur TPU, tracking W&B |

**Utiliser quand** : besoin d'utiliser JAX au lieu de TensorFlow/PyTorch, ou entrainer sur TPU avec JAX.

---

### 18.2. Guides/TensorFlow Guide/

**Theme** : Approfondissement TensorFlow/Keras
**Structure** : `Beginner/` (2 sous-cours) + `Advanced/`

#### Beginner/Intro to Deep Learning/ (7 notebooks)

Memes notebooks que le cours `Intro to Deep Learning/` (section 10) mais organises dans le contexte TensorFlow :
- `a-single-neuron.ipynb` - Neurone unique avec Keras
- `deep-neural-networks.ipynb` - Reseaux profonds
- `stochastic-gradient-descent.ipynb` - SGD
- `overfitting-and-underfitting.ipynb` - Surapprentissage/sous-apprentissage
- `dropout-and-batch-normalization.ipynb` - Regularisation
- `binary-classification.ipynb` - Classification binaire
- `detecting-the-higgs-boson-with-tpus.ipynb` - TPU

#### Beginner/Computer Vision/ (9 notebooks)

Memes notebooks que le cours `Computer Vision/` (section 11) dans le contexte TensorFlow :
- `the-convolutional-classifier.ipynb` - Classificateur CNN
- `convolution-and-relu.ipynb` - Convolution et ReLU
- `maximum-pooling.ipynb` - MaxPooling
- `the-sliding-window.ipynb` - Fenetre glissante
- `custom-convnets.ipynb` - CNN personnalises
- `data-augmentation.ipynb` - Augmentation de donnees
- `transfer-learning.ipynb` - Transfer learning
- `create-your-first-submission.ipynb` - Premiere soumission
- `getting-started-tpus-cassava-leaf-disease.ipynb` - TPU + Cassava

#### Advanced/ (4 notebooks)

| Fichier | Description |
|---|---|
| `tensorflow-roberta-0-705.ipynb` | Modele RoBERTa en TensorFlow, NLP avance, score 0.705 |
| `yolo-v3-object-detection-in-tensorflow.ipynb` | Detection d'objets avec YOLO v3 en TensorFlow |
| `deep-learning-for-time-series-forecasting.ipynb` | Prevision de series temporelles avec deep learning |
| `seti-bl-channelwise-alignment-tf-tpu.ipynb` | Alignement de canaux pour le projet SETI, TPU |

**Utiliser quand** : besoin de code TensorFlow/Keras specifique, YOLO, RoBERTa, ou DL pour series temporelles.

---

### 18.3. Guides/Natural Language Processing Guide/

**Theme** : Traitement automatique du langage naturel (NLP)
**Contenu** : 4 notebooks (pas de sous-dossiers)

| Fichier | Description |
|---|---|
| `getting-started-with-nlp-for-absolute-beginners.ipynb` | Introduction NLP : tokenisation, bag-of-words, premiers modeles texte |
| `nlp-glove-bert-tf-idf-lstm-explained.ipynb` | Tour d'horizon : TF-IDF, GloVe, LSTM, BERT, comparaisons |
| `how-to-preprocessing-when-using-embeddings.ipynb` | Preprocessing texte pour embeddings : nettoyage, normalisation, padding |
| `deep-learning-for-nlp-zero-to-transformers-bert.ipynb` | Du zero aux Transformers : RNN, attention, BERT, fine-tuning |

**Utiliser quand** : besoin de travailler avec du texte, embeddings, BERT, ou tout modele NLP.

---

### 18.4. Guides/R Guide/

**Theme** : Langage R pour la data science
**Structure** : `Beginner/` + `Advanced/`

| Niveau | Fichier | Description |
|---|---|---|
| `Beginner/` | `getting-started-in-r-first-steps.ipynb` | Bases de R : vecteurs, dataframes, fonctions, syntaxe |
| `Beginner/` | `cheatsheet-70-ggplot-charts.ipynb` | 70 types de graphiques ggplot2 avec code, reference visuelle |
| `Advanced/` | `eda-prophet-mlp-neural-network-forecasting.ipynb` | Previsions avec Prophet et MLP en R |
| `Advanced/` | `back-to-predict-the-future-interactive-m5-eda.Rmd` | EDA interactive sur la competition M5 (fichier R Markdown) |
| `Advanced/` | `house-prices-lasso-xgboost-and-a-detailed-eda.Rmd` | EDA detaillee + Lasso + XGBoost pour House Prices (R Markdown) |

**Utiliser quand** : besoin de travailler en R, creer des graphiques ggplot, ou utiliser Prophet.

---

### 18.5. Guides/Transfer Learning for CV Guide/

**Theme** : Applications du Transfer Learning en vision par ordinateur
**Structure** : 7 sous-dossiers par type de tache

#### Multiclass Classification/ (3 notebooks)

| Fichier | Description |
|---|---|
| `cats-or-dogs-using-cnn-with-transfer-learning.ipynb` | Classification chats/chiens avec transfer learning |
| `brain-tumor-mri-classification-tensorflow-cnn.ipynb` | Classification de tumeurs cerebrales sur IRM avec CNN |
| `efficientnetb5-with-keras-aptos-2019.ipynb` | EfficientNet-B5 pour la retinopathie diabetique (APTOS 2019) |

#### Multilabel Classification/ (1 notebook)

| Fichier | Description |
|---|---|
| `pretrained-resnet34-with-rgby-0-460-public-lb.ipynb` | ResNet34 pre-entraine pour classification multi-label d'images proteines |

#### Object Detection/ (2 notebooks)

| Fichier | Description |
|---|---|
| `end-to-end-object-detection-with-transformers-detr.ipynb` | Detection d'objets avec DETR (Transformers) de bout en bout |
| `airbus-mask-rcnn-and-coco-transfer-learning.ipynb` | Mask R-CNN + transfer COCO pour detection de navires (Airbus) |

#### Segmentation/ (2 notebooks)

| Fichier | Description |
|---|---|
| `mask-rcnn-and-medical-transfer-learning-siim-acr.ipynb` | Mask R-CNN pour segmentation medicale (pneumothorax SIIM-ACR) |
| `sartorius-starter-torch-mask-r-cnn-lb-0-273.ipynb` | Mask R-CNN PyTorch pour segmentation de cellules (Sartorius) |

#### Emotion Recognition/ (2 notebooks)

| Fichier | Description |
|---|---|
| `emotion-detection.ipynb` | Detection d'emotions faciales avec CNN |
| `facial-emotion-recognition-using-transfer-learning.ipynb` | Reconnaissance d'emotions par transfer learning |

#### Pose Estimation/ (1 notebook)

| Fichier | Description |
|---|---|
| `centernet-baseline.ipynb` | CenterNet pour l'estimation de pose (baseline) |

#### Anomaly Detection/ (1 notebook)

| Fichier | Description |
|---|---|
| `video-anomaly-detection.ipynb` | Detection d'anomalies dans des videos |

**Utiliser quand** : besoin d'un exemple de transfer learning pour une tache CV specifique (detection, segmentation, classification, etc.).

---

### 18.6. Guides/Kaggle Competitions Guide/

**Theme** : Guide complet pour les competitions Kaggle avec donnees et solutions
**Structure** : `Beginner/` + `Advanced/`

#### Beginner/Getting Started with Kaggle Competitions/

| Fichier | Description |
|---|---|
| `getting-started-with-kaggle-competitions.ipynb` | Guide d'introduction : inscription, soumission, workflow competition |

#### Beginner/Titanic - Machine Learning from Disaster/

| Fichier/Dossier | Description |
|---|---|
| `titanic-tutorial.ipynb` | Tutoriel complet Titanic : EDA, feature engineering, modele, soumission |
| `data/train.csv` | Donnees d'entrainement Titanic (891 passagers, survie, classe, age, etc.) |
| `data/test.csv` | Donnees de test Titanic (418 passagers, sans la colonne Survived) |
| `data/gender_submission.csv` | Exemple de soumission (format attendu) |

#### Beginner/Kaggle Getting Started Competitions/

Contient 12 competitions avec notebooks + donnees :

##### connectx/ (Jeu Connect Four)
- `connectxbot.ipynb` - Bot pour le jeu Connect Four
- `kaggle-environments-0.1.4/` - Librairie Kaggle Environments complete (moteur de jeu, bots, tests)
  - `kaggle_environments/` - Code source Python du moteur
  - `kaggle_environments/envs/connectx/` - Environnement Connect Four
  - `kaggle_environments/envs/tictactoe/` - Environnement TicTacToe
  - `kaggle_environments/envs/identity/` - Environnement Identity (test)

##### contradictory-my-dear-watson/ (NLP - Inference textuelle)
- `kerasnlp-starter-notebook-contradictory-dearwatson.ipynb` - Classification NLI avec KerasNLP
- `data/train.csv` - Paires de phrases avec labels (entailment, contradiction, neutral)
- `data/test.csv` - Paires de test
- `data/sample_submission.csv` - Format de soumission

##### digit-recognizer/ (Classification d'images - MNIST)
- `a-beginner-s-approach-to-classification.ipynb` - Approche debutant pour la classification de chiffres
- `introduction-to-cnn-keras-0-997-top-6.ipynb` - CNN Keras avec score 0.997 (top 6%)
- `tensorflow-deep-nn.ipynb` - Reseau profond TensorFlow pour MNIST
- `data/train.csv` - 42000 images de chiffres (28x28 pixels aplatis)
- `data/test.csv` - 28000 images de test
- `data/sample_submission.csv` - Format de soumission

##### gan-getting-started/ (Generation d'images - GAN)
- `fortunec-dds8150-7.ipynb` - GAN pour generer des peintures style Monet
- `data/monet_jpg/` - ~300 images JPG de peintures de Monet
- `data/photo_jpg/` - ~7000 photos JPG reelles
- `data/monet_tfrec/` - Images Monet au format TFRecord
- `data/photo_tfrec/` - Photos au format TFRecord

##### house-prices-advanced-regression-techniques/ (Regression)
- `house-prices-prediction-using-tfdf.ipynb` - Prediction de prix avec TensorFlow Decision Forests
- `data/train.csv` - 1460 maisons avec 79 features + prix de vente
- `data/test.csv` - 1459 maisons de test
- `data/sample_submission.csv` - Format de soumission

##### Housing Prices Competition for Kaggle Learn Users/ (Regression simplifiee)
- `exercise-underfitting-and-overfitting-testing.ipynb` - Exercice sur le compromis biais-variance
- `data/train.csv` - Donnees d'entrainement simplifiees
- `data/test.csv` - Donnees de test
- `data/sample_submission.csv` - Format de soumission

##### LLM Classification Finetuning/ (Fine-tuning LLM)
- `lmsys-kerasnlp-starter.ipynb` - Starter notebook pour fine-tuning de LLM avec KerasNLP
- `data/train.csv` - Donnees d'entrainement pour classification de conversations LLM
- `data/test.csv` - Donnees de test
- `data/sample_submission.csv` - Format de soumission

##### Natural Language Processing with Disaster Tweets/ (Classification de texte)
- `kerasnlp-starter-notebook-disaster-tweets.ipynb` - Classification de tweets (catastrophe ou non)
- `data/train.csv` - Tweets avec labels (1 = catastrophe, 0 = non)
- `data/test.csv` - Tweets de test
- `data/sample_submission.csv` - Format de soumission

##### Petals to the Metal - Flower Classification on TPU/ (Classification d'images sur TPU)
- `flower-classification.ipynb` - Classification de fleurs sur TPU
- `data/tfrecords-jpeg-192x192/` - Images de fleurs en TFRecord (192x192)
- `data/tfrecords-jpeg-224x224/` - Images de fleurs en TFRecord (224x224)
- `data/tfrecords-jpeg-331x331/` - Images de fleurs en TFRecord (331x331)
- `data/tfrecords-jpeg-512x512/` - Images de fleurs en TFRecord (512x512)

##### spaceship-titanic/ (Classification)
- `spaceship-titanic-with-tfdf.ipynb` - Classification avec TensorFlow Decision Forests
- `data/train.csv` - Passagers avec features et label de transport
- `data/test.csv` - Donnees de test
- `data/sample_submission.csv` - Format de soumission

##### store-sales-time-series-forecasting/ (Series temporelles)
- `store-sales-forecasting-using-time-series-analysis.ipynb` - Prevision de ventes de magasins
- `data/train.csv` - Ventes historiques par magasin/produit/jour
- `data/test.csv` - Periodes a predire
- `data/sample_submission.csv` - Format de soumission
- `data/oil.csv` - Prix du petrole quotidien (feature externe)
- `data/stores.csv` - Metadata des magasins (ville, type, cluster)
- `data/holidays_events.csv` - Jours feries et evenements
- `data/transactions.csv` - Nombre de transactions par magasin/jour

##### Titanic - Machine Learning from Disaster/ (Classification)
- `titanic-competition-w-tensorflow-decision-forests.ipynb` - Titanic avec TF Decision Forests
- `titanic-tutorial.ipynb` - Tutoriel classique Titanic
- `data/train.csv` - Donnees d'entrainement
- `data/test.csv` - Donnees de test
- `data/gender_submission.csv` - Exemple de soumission

#### Advanced/

| Fichier | Description |
|---|---|
| `winning-solutions-of-kaggle-competitions.ipynb` | Analyse des solutions gagnantes de competitions Kaggle, strategies, techniques |

**Utiliser quand** : besoin d'un exemple complet de competition Kaggle avec donnees, ou d'une reference pour un type de probleme specifique (regression, classification, NLP, vision, series temporelles, GAN).

---

## Index rapide par besoin

| Je veux... | Aller a... |
|---|---|
| Apprendre Python | `Python/` ou `Intro to Programming/` |
| Manipuler des DataFrames | `Pandas/` |
| Ecrire des requetes SQL | `Intro to SQL/` ou `Advanced SQL/` |
| Nettoyer des donnees | `Data Cleaning/` |
| Creer des graphiques | `Data Visualization/` |
| Construire un premier modele ML | `Intro to Machine Learning/` |
| Utiliser XGBoost ou des pipelines | `Intermediate Machine Learning/` |
| Creer un reseau de neurones | `Intro to Deep Learning/` |
| Travailler avec des images (CNN) | `Computer Vision/` |
| Creer de nouvelles features | `Feature Engineering/` |
| Expliquer un modele | `Machine Learning Explainability/` |
| Travailler avec des coordonnees GPS | `Geospatial Analysis/` |
| Prevoir des series temporelles | `Time Series/` |
| Creer un agent de jeu | `Intro to Game AI and Reinforcement Learning/` |
| Comprendre l'ethique IA | `Intro to AI Ethics/` |
| Utiliser JAX | `Guides/JAX Guide/` |
| Code TensorFlow avance | `Guides/TensorFlow Guide/` |
| Travailler avec du texte/NLP | `Guides/Natural Language Processing Guide/` |
| Programmer en R | `Guides/R Guide/` |
| Transfer learning pour vision | `Guides/Transfer Learning for CV Guide/` |
| Exemples de competitions Kaggle | `Guides/Kaggle Competitions Guide/` |
| Donnees CSV pour s'entrainer | `Guides/Kaggle Competitions Guide/Beginner/Kaggle Getting Started Competitions/` |

---

## Index par type de fichier

| Type | Localisation | Usage |
|---|---|---|
| `.ipynb` (Jupyter Notebook) | Tous les dossiers | Code + explications, fichier principal de chaque lecon |
| `.csv` | `Guides/Kaggle Competitions Guide/.../data/` | Datasets de competition (train, test, submission) |
| `.jpg` | `Guides/.../gan-getting-started/data/` | Images Monet et photos pour GAN |
| `.tfrec` (TFRecord) | `Guides/.../gan-getting-started/data/` et `.../Petals to the Metal/data/` | Donnees d'images au format TensorFlow |
| `.Rmd` (R Markdown) | `Guides/R Guide/Advanced/` | Notebooks R Markdown |
| `.py` | `Guides/.../connectx/kaggle-environments-0.1.4/` | Code source du moteur de jeu Kaggle Environments |
| `.json` | `Guides/.../connectx/kaggle-environments-0.1.4/` | Schemas et configs du moteur de jeu |

---

## Parcours d'apprentissage recommandes

### Debutant complet
1. `Intro to Programming/` → 2. `Python/` → 3. `Pandas/` → 4. `Data Visualization/` → 5. `Intro to Machine Learning/`

### Data Science
1. `Python/` → 2. `Pandas/` → 3. `Data Cleaning/` → 4. `Data Visualization/` → 5. `Feature Engineering/` → 6. `Intermediate Machine Learning/`

### Deep Learning
1. `Intro to Deep Learning/` → 2. `Computer Vision/` → 3. `Guides/TensorFlow Guide/` → 4. `Guides/Transfer Learning for CV Guide/`

### NLP
1. `Python/` → 2. `Guides/Natural Language Processing Guide/` → 3. `Guides/TensorFlow Guide/Advanced/` (RoBERTa)

### Competitions Kaggle
1. `Intro to Machine Learning/` → 2. `Intermediate Machine Learning/` → 3. `Feature Engineering/` → 4. `Guides/Kaggle Competitions Guide/`
