# Claude Code - Expert Kaggle Gold Medal

34 skills + 4 agents pour transformer Claude Code en expert Kaggle. Couvre tout le workflow : de l'exploration des données à la soumission finale.

Tu n'as pas besoin d'être un expert en data science. Suis ce guide étape par étape, lance les commandes, et Claude fait le travail.

---

## Vocabulaire (pour les débutants)

Avant de commencer, voici les termes que tu vas croiser :

| Terme | Signification |
|-------|--------------|
| **Skill** (`/kaggle-*`) | Une commande que tu tapes dans Claude Code. Elle fait une tâche précise (nettoyer les données, entraîner un modèle, etc.) |
| **Agent** | Un assistant autonome qui fait une mission longue (analyser une compétition, optimiser un modèle). Plus puissant qu'un skill. |
| **CV (Cross-Validation)** | Le score de ton modèle mesuré en local, sur tes propres données. C'est ta boussole. |
| **LB (Leaderboard)** | Le score affiché par Kaggle quand tu soumets. C'est le vrai score. |
| **Gap CV-LB** | La différence entre ton score local (CV) et le score Kaggle (LB). Si le gap est petit (<3%), ton CV est fiable. |
| **OOF (Out-Of-Fold)** | Les prédictions de ton modèle sur les données d'entraînement, faites fold par fold (le modèle ne voit jamais les données qu'il prédit). Sert à calculer le CV et à construire les ensembles. |
| **Feature** | Une colonne dans tes données. "Feature engineering" = créer de nouvelles colonnes utiles. |
| **Baseline** | Un premier modèle très simple, sans optimisation, juste pour avoir un score de départ. |
| **Ensemble** | Combiner les prédictions de plusieurs modèles pour avoir un meilleur score. |
| **Overfitting** | Le modèle a "mémorisé" les données d'entraînement au lieu d'apprendre des patterns. Score de train excellent mais score de validation mauvais. |
| **Leakage** | Le modèle a accès à de l'information qu'il ne devrait pas avoir (ex: la réponse est cachée dans une feature). Le score semble très bon mais c'est faux. |
| **SHAP** | Une technique pour comprendre quelles features sont importantes pour le modèle et pourquoi. |
| **GBDT** | Gradient Boosted Decision Trees : LightGBM, XGBoost, CatBoost. Les modèles qui gagnent la majorité des compétitions tabulaires. |

---

## Installation

```bash
# Installation globale (disponible dans tous tes projets)
cp -r skills/* ~/.claude/skills/
cp -r agents/* ~/.claude/agents/
```

Puis **redémarre Claude Code** pour activer les skills et agents.

> **Perdu à n'importe quel moment ?** Tape `/kaggle-guide`. Il analyse ton projet et te dit exactement quoi faire ensuite, avec la commande à copier-coller.

---

## Le Workflow complet — De zéro à la médaille d'or

La clé pour gagner sur Kaggle, c'est l'**itération**. Tu ne fais pas tout une seule fois dans l'ordre. Tu fais des boucles : tu crées des features, tu entraînes, tu évalues, tu ajustes, tu recommences. Chaque boucle améliore ton score un petit peu.

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  ÉTAPE 1 — Comprendre la compétition                        │
│  ÉTAPE 2 — Première soumission (baseline)                   │
│                                                              │
│         ╔════════════════════════════════════╗                │
│         ║       BOUCLE D'ITÉRATION          ║                │
│         ║  (tu passes 80% de ton temps ici) ║                │
│         ║                                    ║                │
│         ║  ÉTAPE 3 — Améliorer les features  ║                │
│         ║  ÉTAPE 4 — Entraîner le modèle     ║                │
│         ║  ÉTAPE 5 — Évaluer et comparer     ║                │
│         ║             ↓                      ║                │
│         ║  Le score monte ? → Continuer.     ║                │
│         ║  Le score stagne ? → Changer.      ║                │
│         ║             ↓                      ║                │
│         ║     Retour à l'étape 3             ║                │
│         ╚════════════════════════════════════╝                │
│                                                              │
│  ÉTAPE 6 — Combiner les modèles (ensemble)                  │
│  ÉTAPE 7 — Vérifier et soumettre                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## ÉTAPE 1 — Comprendre la compétition

**Objectif :** Savoir à quoi tu as affaire avant d'écrire du code.

### Option A : Demander un plan stratégique (recommandé pour les débutants)

```
→ Agent kaggle-strategist
  "Analyse la compétition Spaceship Titanic et crée un plan d'attaque"
```

**Ce que ça fait :** L'agent va :
- Lire les données de la compétition
- Rechercher sur le web les solutions gagnantes de compétitions similaires
- Identifier le type de problème, la métrique, les pièges
- Produire un plan multi-phases avec des scores attendus

**Ce que tu obtiens :** Un rapport complet dans `reports/strategy/` avec un plan à suivre.

### Option B : Explorer les données directement

```
→ /kaggle-eda data/train.csv
```

**Ce que ça fait :** Analyse exploratoire complète en 10 étapes :
- Quels types de colonnes (nombres, texte, dates) ?
- Combien de valeurs manquantes ?
- Comment est distribué le target (ce qu'on veut prédire) ?
- Quelles colonnes sont corrélées entre elles ?
- Y a-t-il des outliers (valeurs anormales) ?
- Les données de train et test sont-elles similaires ?

**Ce que tu obtiens :** Un rapport dans `reports/eda/` qui te dit tout ce qu'il faut savoir sur tes données.

### Si tu as besoin de recherches

```
→ Agent kaggle-researcher
  "Cherche les meilleures techniques pour la classification tabulaire binaire"
```

**Ce que ça fait :** Recherche sur le web les techniques, notebooks populaires, et solutions gagnantes. Rapport dans `reports/research/`.

---

## ÉTAPE 2 — Première soumission (le baseline)

**Objectif :** Avoir un premier score sur le Leaderboard le plus vite possible. Ce score sert de référence : tout ce que tu fais ensuite doit le battre.

### Option A : Pipeline complet d'un coup (le plus rapide)

```
→ /kaggle-pipeline Spaceship Titanic
```

**Ce que ça fait :** Tout d'un coup :
- Crée la structure du projet (dossiers data/, models/, submissions/...)
- Fait une EDA rapide
- Nettoie les données
- Crée quelques features de base
- Entraîne un modèle LightGBM
- Prépare une première soumission

**Ce que tu obtiens :** Un projet complet prêt à itérer + une soumission à envoyer.

### Option B : Baseline minimaliste

```
→ /kaggle-baseline
```

**Ce que ça fait :** Crée un modèle ultra simple en moins de 30 minutes. Pas de feature engineering, paramètres par défaut. Juste le strict minimum pour avoir un score.

### Soumettre ton baseline

```
→ /kaggle-submit
```

**Ce que ça fait :** Vérifie que ton fichier de soumission est correct (bonnes colonnes, bon nombre de lignes, pas de NaN) et le prépare.

### Comparer CV et LB (très important !)

Après ta première soumission, note les deux scores :
- **CV** : le score calculé en local (affiché pendant l'entraînement)
- **LB** : le score affiché par Kaggle après soumission

| Gap CV-LB | Ce que ça veut dire | Quoi faire |
|-----------|--------------------|-----------|
| < 3% | Ton CV est fiable. Tu peux l'utiliser comme boussole. | Continue normalement |
| 3-5% | Prudence. Le CV est approximatif. | Lance `/kaggle-validation` pour améliorer ta stratégie de CV |
| > 5% | Problème. Ton CV n'est pas fiable du tout. | Lance `/kaggle-leakage` (tu as peut-être du leakage) puis `/kaggle-validation` |

---

## ÉTAPE 3 — Améliorer les features

**Objectif :** Créer de nouvelles colonnes dans tes données qui aident le modèle à mieux prédire. C'est l'étape qui a le **plus d'impact** sur ton score.

### Nettoyer les données (si pas encore fait)

```
→ /kaggle-cleaning
```

**Ce que ça fait :**
- Corrige les types de données (un nombre stocké comme texte, etc.)
- Traite les valeurs manquantes (les remplace intelligemment)
- Détecte les NaN déguisés (des "N/A", "?", "-", -999 qui cachent des valeurs manquantes)
- Supprime les doublons
- Gère les outliers (valeurs extrêmes)
- Regroupe les catégories rares
- Supprime les colonnes constantes (qui ne servent à rien)

**Ce que tu obtiens :** Des données propres + un rapport dans `reports/cleaning/`.

### Créer des features

```
→ /kaggle-feature
```

**Ce que ça fait :**
- Crée des interactions entre colonnes (A × B, A / B, A - B)
- Crée des agrégations (moyenne par groupe, count, etc.)
- Encode les catégories (frequency encoding, target encoding)
- Crée des features temporelles si applicable (jour, mois, heure)
- Mesure l'impact de chaque feature sur le CV

**Ce que tu obtiens :** De nouvelles features + un rapport avec l'impact de chacune.

### Si tu veux visualiser

```
→ /kaggle-viz
```

**Ce que ça fait :** Crée des graphiques avancés (distributions, corrélations, heatmaps) pour mieux comprendre les données.

---

## ÉTAPE 4 — Entraîner le modèle

**Objectif :** Entraîner un modèle avec tes nouvelles features et mesurer si le score s'est amélioré.

```
→ /kaggle-model
```

**Ce que ça fait :**
- Entraîne un ou plusieurs modèles (LightGBM, XGBoost, CatBoost)
- Utilise la cross-validation (5 folds par défaut) pour un score fiable
- Sauvegarde les prédictions OOF et test dans `artifacts/`
- Affiche l'importance des features

**Pour des données spécifiques :**

| Tes données sont... | Lance... |
|---------------------|----------|
| Un tableau CSV classique | `/kaggle-model` ou `/kaggle-tabular` |
| Du texte (reviews, articles, tweets) | `/kaggle-nlp` |
| Des images (photos, scans, satellite) | `/kaggle-cv` |
| Des séries temporelles (ventes, météo, finance) | `/kaggle-timeseries` |
| Un tableau mais tu veux du deep learning | `/kaggle-deeplearning` |
| Un jeu ou une simulation | `/kaggle-rl` |

---

## ÉTAPE 5 — Évaluer et comparer

**Objectif :** Savoir si ton changement a amélioré le score ou non. Ne garder que ce qui marche.

```
→ /kaggle-experiments
```

**Ce que ça fait :**
- Log ton run dans `runs.csv` (CV score, features utilisées, paramètres)
- Compare avec les runs précédents
- Identifie quel changement a eu le plus d'impact

### Décider quoi faire ensuite

| Résultat | Ce que ça veut dire | Quoi faire |
|----------|--------------------|-----------|
| CV monte de +0.005 ou plus | Ta feature ou ton changement marche | Garde-le ! Retour à l'étape 3 pour ajouter d'autres features |
| CV ne bouge pas | Le changement n'apporte rien | Supprime-le. Essaie une autre approche |
| CV baisse | Bug ou bruit | Lance `/kaggle-debug` pour comprendre pourquoi |
| CV monte mais LB baisse | Overfitting probable | Lance `/kaggle-validation` pour revoir le CV |
| Tu ne sais pas quelles features sont utiles | Besoin d'analyse | Lance `/kaggle-explain` (SHAP values) |

### Comprendre ton modèle

```
→ /kaggle-explain
```

**Ce que ça fait :** Montre quelles features sont importantes, lesquelles le modèle utilise vraiment, et lesquelles sont du bruit. Te montre aussi les Partial Dependence Plots (comment chaque feature influence la prédiction).

### Quand itérer ? Quand passer à l'ensemble ?

**Continue d'itérer (étapes 3-4-5) tant que :**
- Tu trouves des features qui améliorent le score
- Tu n'as pas encore testé les approches évidentes
- Tu as des idées à explorer

**Passe à l'étape 6 quand :**
- Le score ne monte plus malgré tes efforts
- Tu as au moins 2-3 modèles différents (LightGBM, XGBoost, CatBoost)
- Tu as essayé les principales features

---

## Quand ça va mal (à utiliser pendant les itérations)

### Le score a baissé

```
→ Agent kaggle-debugger
  "Mon score est passé de 0.81 à 0.79 après avoir ajouté des features d'interaction"
```

**Ce que ça fait :** Le debugger analyse ton code, tes données, et tes résultats. Il produit un **diagnostic** (la cause du problème) et un **patch plan** (les fichiers à modifier avec le diff exact ligne par ligne). Rapport dans `reports/debug/`.

### Le score stagne

```
→ /kaggle-debug
  "Le score stagne à 0.81 depuis 5 itérations"
```

**Ce que ça fait :** Analyse les prédictions du modèle, identifie les observations les plus mal prédites, et propose des pistes d'amélioration.

Ensuite :
```
→ Agent kaggle-researcher
  "Cherche des techniques avancées pour améliorer une classification binaire tabulaire"
```

**Ce que ça fait :** Recherche sur le web des nouvelles idées et techniques.

### Le CV et le LB ne matchent pas

```
→ /kaggle-validation
```

**Ce que ça fait :** Diagnostique ta stratégie de CV. Peut-être que tu as besoin de GroupKFold (données groupées), TimeSeriesCV (données temporelles), ou adversarial validation.

### Score suspicieusement bon

```
→ /kaggle-leakage
```

**Ce que ça fait :** Audit complet de data leakage en 7 points. Si une feature a un score trop bon, c'est peut-être parce qu'elle contient la réponse.

### C'est trop lent ou RAM saturée

```
→ /kaggle-efficiency
```

**Ce que ça fait :** Réduit l'usage mémoire, accélère le training, propose des alternatives (Polars au lieu de Pandas, GPU, etc.).

### Pas assez de données ou classes déséquilibrées

```
→ /kaggle-augmentation
```

**Ce que ça fait :** Augmente tes données avec SMOTE (pour le déséquilibre), Mixup, pseudo-labeling, etc.

### Optimiser les hyperparamètres

```
→ Agent kaggle-optimizer
  "Optimise les hyperparamètres de mon LightGBM"
```

**Ce que ça fait :** Crée un script Optuna, lance une recherche automatique des meilleurs paramètres, et sauvegarde les résultats dans `configs/` et `reports/optimizer/`.

> **Important :** Ne fais le tuning que quand tes features sont stables. Les features ont plus d'impact que les hyperparamètres.

---

## ÉTAPE 6 — Combiner les modèles (ensemble)

**Objectif :** Combiner les prédictions de plusieurs modèles pour obtenir un meilleur score que chaque modèle individuellement.

**Prérequis :** Tu dois avoir au moins 2-3 modèles différents (ex: LightGBM + XGBoost + CatBoost).

```
→ /kaggle-ensemble
```

**Ce que ça fait :**
- Calcule la corrélation entre tes modèles (s'ils prédisent tous la même chose, l'ensemble n'aide pas)
- Teste différentes méthodes : moyenne simple, moyenne pondérée, rank average, stacking
- Trouve les poids optimaux
- Sauvegarde les prédictions de l'ensemble dans `artifacts/`

| Corrélation entre modèles | Ce que ça veut dire | Quoi faire |
|--------------------------|--------------------|-----------|
| > 0.98 | Les modèles sont trop similaires | Diversifie : `/kaggle-deeplearning`, features différentes, seeds différents |
| 0.93 - 0.97 | Bonne diversité | L'ensemble va bien marcher |
| < 0.93 | Très différents (attention) | Vérifie que chaque modèle est bon individuellement |

---

## ÉTAPE 7 — Vérifier et soumettre

### 7a. Polish (pour gratter les derniers points)

```
→ /kaggle-metrics
```
Vérifie que ta métrique locale correspond exactement à celle de Kaggle.

```
→ /kaggle-calibration
```
Calibre les probabilités. Utile si la métrique est Log Loss ou Brier Score.

```
→ /kaggle-postprocess
```
Optimise les seuils de décision (pour F1, Accuracy), le clipping, l'arrondi.

```
→ /kaggle-inference
```
Pipeline d'inférence optimisé. Utile pour les "code competitions" avec contrainte de temps.

### 7b. Vérification finale

```
→ /kaggle-sanity
```

**Ce que ça fait :** Suite complète de tests de sanité :
- Le fichier de soumission a le bon format ?
- Pas de NaN dans les prédictions ?
- Les features sont-elles vraiment utiles (vs features aléatoires) ?
- Le preprocessing est-il bien dans le fold (pas de leakage) ?

**Lance toujours `/kaggle-sanity` avant une soumission importante.**

### 7c. Choisir les 2 soumissions finales

```
→ /kaggle-leaderboard
```

**Ce que ça fait :** Analyse le risque de "shake-up" (quand le classement change entre le public LB et le private LB) et t'aide à choisir tes 2 soumissions finales.

**Règle simple pour choisir :**
| Soumission 1 (conservative) | Soumission 2 (aggressive) |
|-----------------------------|--------------------------|
| Le modèle avec le meilleur CV et le plus stable | Le meilleur ensemble ou la meilleure expérimentation |

### 7d. Soumettre

```
→ /kaggle-submit
```

**Ce que ça fait :** Valide le format et prépare le fichier `submission.csv`.

---

## Récapitulatif — Toutes les commandes

### Les 4 Agents (missions longues et autonomes)

| Agent | Ce qu'il fait | Quand l'utiliser |
|-------|---------------|-----------------|
| `kaggle-strategist` | Analyse la compétition, recherche les solutions gagnantes similaires sur le web, produit un plan multi-phases | Tu commences une compétition |
| `kaggle-researcher` | Recherche des techniques, notebooks populaires, et solutions gagnantes sur le web | Tu cherches de nouvelles idées |
| `kaggle-optimizer` | Crée des scripts Optuna et optimise les hyperparamètres automatiquement | Tes features sont stables, tu veux tuner |
| `kaggle-debugger` | Diagnostique un problème et produit un patch plan (fichiers + lignes + diff) | Le score a baissé ou quelque chose ne va pas |

### Les 34 Skills (commandes rapides)

#### Navigation
| Commande | Ce qu'elle fait |
|----------|----------------|
| `/kaggle-guide` | **T'aide quand tu es perdu.** Analyse ton projet et te dit quoi faire avec la commande exacte. |
| `/kaggle-knowledge` | Base de connaissances chargée automatiquement. Tu n'as rien à faire. |

#### Démarrage
| Commande | Ce qu'elle fait |
|----------|----------------|
| `/kaggle-pipeline` | Crée un projet complet de A à Z : structure + EDA + cleaning + features + modèle + soumission. |
| `/kaggle-baseline` | Crée un modèle ultra simple en <30 min. Juste pour avoir un premier score. |

#### Données
| Commande | Ce qu'elle fait |
|----------|----------------|
| `/kaggle-eda` | Analyse exploratoire complète des données (types, distributions, corrélations, missing values). |
| `/kaggle-cleaning` | Nettoie les données : types, missing values, outliers, doublons, NaN déguisés, catégories rares. |
| `/kaggle-feature` | Crée des features : interactions, agrégations, encodages, features temporelles. |
| `/kaggle-viz` | Crée des graphiques avancés avec Seaborn, Matplotlib ou Plotly. |
| `/kaggle-leakage` | Détecte le data leakage en 7 points. Lance ça si ton score semble trop beau. |
| `/kaggle-augmentation` | Augmente les données : SMOTE, Mixup, pseudo-labeling. Pour les petits datasets ou classes déséquilibrées. |

#### Modélisation
| Commande | Ce qu'elle fait |
|----------|----------------|
| `/kaggle-model` | Entraîne des modèles (LightGBM, XGBoost, CatBoost) avec cross-validation. |
| `/kaggle-validation` | Choisit la bonne stratégie de CV. Diagnostique les problèmes CV vs LB. |
| `/kaggle-ensemble` | Combine plusieurs modèles (moyenne, stacking, blending). Trouve les poids optimaux. |
| `/kaggle-experiments` | Tracker d'expériences : log chaque run, compare les scores, identifie ce qui marche. |

#### Diagnostic
| Commande | Ce qu'elle fait |
|----------|----------------|
| `/kaggle-debug` | Diagnostique les problèmes : overfitting, score qui baisse, erreurs de prédiction. |
| `/kaggle-explain` | Explique le modèle : SHAP, permutation importance. Montre quelles features comptent. |
| `/kaggle-sanity` | Tests de sanité avant soumission : format, NaN, leakage résiduel, features utiles. |
| `/kaggle-metrics` | Vérifie que ta métrique locale = métrique Kaggle. Implémente des métriques custom. |

#### Finalisation
| Commande | Ce qu'elle fait |
|----------|----------------|
| `/kaggle-calibration` | Calibre les probabilités (Platt, Isotonic). Pour Log Loss / Brier Score. |
| `/kaggle-postprocess` | Optimise les seuils, clipping, arrondi. Pour F1, Accuracy, QWK. |
| `/kaggle-inference` | Pipeline d'inférence optimisé (TTA, batch, multi-model). Pour les code competitions. |
| `/kaggle-leaderboard` | Analyse le risque de shake-up. Aide à choisir les 2 soumissions finales. |
| `/kaggle-submit` | Valide le format de soumission et prépare le fichier final. |

#### Domaines spécialisés
| Commande | Spécialité |
|----------|-----------|
| `/kaggle-tabular` | Données tabulaires (CSV classique) — pipeline complet optimisé |
| `/kaggle-nlp` | Texte — fine-tuning BERT/DeBERTa, classification, NER |
| `/kaggle-cv` | Images — EfficientNet, ViT, augmentations, segmentation |
| `/kaggle-timeseries` | Séries temporelles — lag features, Prophet, validation temporelle |
| `/kaggle-deeplearning` | Deep learning sur tableaux — TabNet, FT-Transformer, SAINT |
| `/kaggle-sql` | SQL et BigQuery — requêtes, window functions, optimisation |
| `/kaggle-geospatial` | Données géographiques — GeoPandas, Folium, distance features |
| `/kaggle-rl` | Jeux et simulation — minimax, MCTS, PPO, self-play |
| `/kaggle-tpu` | TPU et TensorFlow — tf.distribute, TFRecords |
| `/kaggle-efficiency` | Vitesse et mémoire — reduce_mem, Polars, GPU, caching |
| `/kaggle-ethics` | Biais et fairness — audit éthique, Model Cards |

---

## Convention d'artefacts

Chaque skill produit des fichiers dans une structure standardisée :

```
mon-projet/
├── data/
│   ├── raw/                    # Données brutes (train.csv, test.csv)
│   └── processed/              # Données transformées
├── reports/                    # Rapports de chaque skill/agent
│   ├── strategy/               #   → kaggle-strategist
│   ├── research/               #   → kaggle-researcher
│   ├── eda/                    #   → /kaggle-eda
│   ├── cleaning/               #   → /kaggle-cleaning
│   ├── debug/                  #   → kaggle-debugger
│   ├── optimizer/              #   → kaggle-optimizer
│   └── ...                     #   → autres skills
├── artifacts/                  # Prédictions OOF et test
│   ├── oof_lgbm_v1.parquet     #   → prédictions OOF du LightGBM v1
│   ├── test_lgbm_v1.parquet    #   → prédictions test du LightGBM v1
│   └── oof_ensemble_v1.parquet #   → OOF de l'ensemble
├── models/                     # Modèles sauvegardés (.pkl, .cbm)
├── configs/                    # Paramètres optimisés (.yaml)
├── submissions/                # Fichiers de soumission (.csv)
├── notebooks/                  # Notebooks Jupyter
└── runs.csv                    # Historique de toutes tes expériences
```

---

## Definition of Done — Contrats de sortie

Chaque skill clé a des artefacts obligatoires qui garantissent que le travail est complet :

| Skill | C'est fini quand... |
|-------|---------------------|
| `/kaggle-baseline` | `runs.csv` (1re ligne) + OOF + soumission + score CV noté |
| `/kaggle-model` | 5 folds OU 2+ seeds + OOF + test predictions + params dans `runs.csv` |
| `/kaggle-ensemble` | Matrice de corrélation + méthode documentée + OOF/test ensemble |
| `/kaggle-metrics` | Fonction métrique + test mini-sample + vérification vs LB |
| `/kaggle-postprocess` | Fit sur OOF uniquement + gain mesuré + postprocessor sauvegardé |
| `/kaggle-sanity` | Tous les checks passés (OK/KO par item) |
| `/kaggle-calibration` | Reliability diagram + ECE avant/après + calibrateur sauvegardé |
| `kaggle-debugger` | Rapport + patch plan (fichier + ligne + diff) + vérifications |
