# Claude Code - Expert Kaggle Gold Medal

Ensemble de skills et agents pour transformer Claude Code en expert data science et compétitions Kaggle.

## Installation

Copier le contenu dans ton dossier `.claude` :

```bash
# Copier les skills
cp -r skills/* ~/.claude/skills/

# Copier les agents
cp -r agents/* ~/.claude/agents/
```

Ou pour un projet spécifique :
```bash
cp -r skills/* .claude/skills/
cp -r agents/* .claude/agents/
```

Puis redémarrer Claude Code.

## Skills Disponibles (28)

### Skill Auto-chargé (Knowledge)
| Skill | Description |
|-------|-------------|
| `kaggle-knowledge` | Base de connaissances experte chargée automatiquement quand tu travailles sur du data science |

### Skills du Workflow Principal
| Commande | Description | Phase |
|----------|-------------|-------|
| `/kaggle-pipeline` | Pipeline complet de A à Z pour une compétition | Démarrage |
| `/kaggle-baseline` | Baseline ultra rapide (<30 min) avec première soumission | Démarrage |
| `/kaggle-eda` | Analyse exploratoire complète d'un dataset | Exploration |
| `/kaggle-cleaning` | Nettoyage et preprocessing des données | Nettoyage |
| `/kaggle-leakage` | Détection systématique de data leakage (7 types) | Sécurité |
| `/kaggle-feature` | Feature engineering avancé | Features |
| `/kaggle-validation` | Stratégie de validation croisée robuste | Validation |
| `/kaggle-model` | Construction et optimisation de modèles | Modélisation |
| `/kaggle-debug` | Diagnostic et debugging de modèles | Diagnostic |
| `/kaggle-ensemble` | Stratégies d'ensembling avancées | Ensemble |
| `/kaggle-augmentation` | Augmentation de données (tabulaire, image, texte, time series) | Augmentation |
| `/kaggle-experiments` | Tracking d'expériences, comparaison de runs, ablation | Tracking |
| `/kaggle-efficiency` | Optimisation vitesse, RAM, GPU, caching | Performance |
| `/kaggle-leaderboard` | Stratégie de LB, shake-up risk, sélection finale | Soumission |
| `/kaggle-submit` | Préparation et validation de soumission | Soumission |

### Skills par Domaine
| Commande | Description |
|----------|-------------|
| `/kaggle-tabular` | Stratégie experte pour compétitions tabulaires |
| `/kaggle-cv` | Stratégie experte pour compétitions Computer Vision |
| `/kaggle-nlp` | Stratégie experte pour compétitions NLP |
| `/kaggle-timeseries` | Stratégie experte pour compétitions Time Series |
| `/kaggle-deeplearning` | Deep Learning pour tabulaire (TabNet, FT-Transformer, SAINT, entity embeddings) |
| `/kaggle-sql` | SQL & BigQuery : extraction de données, window functions, optimisation |
| `/kaggle-geospatial` | Analyse géospatiale : GeoPandas, Folium, CRS, spatial joins, proximité |
| `/kaggle-rl` | Game AI & Reinforcement Learning : minimax, MCTS, PPO, self-play |
| `/kaggle-tpu` | TPU & TensorFlow/Keras : tf.distribute, tf.data, TFRecords, TTA, LR schedules |

### Skills d'Analyse
| Commande | Description |
|----------|-------------|
| `/kaggle-explain` | ML Explainability : SHAP, LIME, PDP, Permutation Importance |
| `/kaggle-viz` | Visualisation avancée : Seaborn, Matplotlib, Plotly, subplots, compositions |
| `/kaggle-ethics` | AI Ethics : détection de biais, métriques de fairness, Model Cards, audit éthique |

### Agents (4)
| Agent | Modèle | Description |
|-------|--------|-------------|
| `kaggle-researcher` | Sonnet | Analyse de compétition, recherche de solutions gagnantes (read-only) |
| `kaggle-optimizer` | Sonnet | Optimisation de modèles et hyperparamètres (peut éditer) |
| `kaggle-strategist` | Opus | Stratège de compétition, plan d'attaque multi-phases (read-only + web) |
| `kaggle-debugger` | Sonnet | Diagnostic de problèmes de performance, analyse d'erreurs (read-only + bash) |

## Guide Pas à Pas — Nouvelle Compétition

### Jour 1 : Comprendre et Démarrer

```
1. /kaggle-pipeline <nom_competition>
   → Crée la structure de dossiers et le template de notebook
   → Tu obtiens : data/, notebooks/, src/, submissions/, models/

2. /kaggle-eda <chemin_du_train.csv>
   → Analyse exploratoire complète (10 étapes)
   → Tu obtiens : types, missing, distributions, corrélations, drift train/test
   → DÉCISION : quel type de problème ? quelle métrique ? quelles features semblent prometteuses ?

3. /kaggle-baseline
   → Baseline end-to-end en <30 min, params par défaut, 0 feature engineering
   → Tu obtiens : CV score de référence + première soumission
   → NOTE le CV et le LB → c'est ta boussole pour la suite
```

**Que faire selon les résultats de l'EDA :**

| Tu observes... | Lance... |
|---|---|
| Beaucoup de missing values (>10%) | `/kaggle-cleaning` |
| Données géographiques (lat/lon) | `/kaggle-geospatial` |
| Colonnes de dates/timestamps | `/kaggle-timeseries` |
| Colonnes de texte libre | `/kaggle-nlp` |
| Fichiers images (jpg, png, dicom) | `/kaggle-cv` |
| Dataset > 500 MB ou RAM saturée | `/kaggle-efficiency` |
| Données SQL/BigQuery | `/kaggle-sql` |
| Compétition de jeu/simulation | `/kaggle-rl` |

### Jour 2-3 : Sécuriser et Nettoyer

```
4. /kaggle-leakage
   → Audit de leakage AVANT d'itérer (7 checks)
   → Tu obtiens : rapport avec CRITICAL / WARNING / OK
   → SI leakage trouvé → corriger AVANT de continuer

5. /kaggle-cleaning
   → Nettoyage complet (missing, outliers, types, doublons, encoding)
   → Tu obtiens : train et test nettoyés + rapport de nettoyage

6. /kaggle-validation
   → Choisir la bonne stratégie de CV
   → Tu obtiens : le bon splitter (Stratified/Group/TimeSeries/Purged)
```

**Que faire selon les résultats du leakage :**

| Tu observes... | Fais... |
|---|---|
| Feature avec corr > 0.95 au target | Supprime-la, c'est du leakage |
| Adversarial AUC > 0.70 | Train/test drift → utilise `/kaggle-validation` pour adapter le CV |
| IDs séquentiels corrélés au target | Utilise GroupKFold ou l'ID comme feature temporelle |
| CV >>> LB (gap > 5%) | Leakage dans le preprocessing → vérifie que tout est DANS le fold |

### Jour 3-7 : Feature Engineering

```
7. /kaggle-feature
   → Feature engineering itératif
   → Tu obtiens : nouvelles features + comparaison CV avant/après

8. /kaggle-experiments (init)
   → Initialiser le tracker d'expériences
   → Chaque run sera loggé : CV, LB, features, params
```

**Que faire selon les résultats du feature engineering :**

| Tu observes... | Lance... |
|---|---|
| Le CV monte de +0.005 ou plus | Bonne feature, garde-la. Log avec `/kaggle-experiments` |
| Le CV ne bouge pas | Feature inutile, supprime-la. Essaie une autre approche |
| Le CV baisse | Bug ou bruit. Lance `/kaggle-debug` pour diagnostiquer |
| Tu veux comprendre quelles features comptent | `/kaggle-explain` |
| Tu veux visualiser les patterns | `/kaggle-viz` |
| Classes très déséquilibrées | `/kaggle-augmentation` (SMOTE, class weights) |
| Dataset très petit (<5000 lignes) | `/kaggle-augmentation` (pseudo-labeling, Mixup) |

### Jour 7-14 : Modélisation

```
9. /kaggle-model
    → LightGBM + XGBoost + CatBoost avec OOF propre
    → Tu obtiens : 3+ modèles avec OOF et test predictions sauvés

10. kaggle-optimizer (agent) → Optuna tuning
    → Hyperparameter optimization contrôlée
    → Tu obtiens : params optimaux par modèle
```

**Que faire selon les résultats de la modélisation :**

| Tu observes... | Lance... |
|---|---|
| Train >>> Val (gap > 5%) | `/kaggle-debug` → overfitting → régulariser |
| Train ≈ Val, tous les deux bas | `/kaggle-debug` → underfitting → plus de features/complexité |
| Score qui stagne malgré le tuning | `/kaggle-deeplearning` (TabNet, FT-Transformer pour diversité) |
| CV ne corrèle pas avec LB | `/kaggle-validation` → revoir la stratégie de split |
| Score baisse vs version précédente | `/kaggle-debug` + `/kaggle-experiments` (compare les runs) |
| Entraînement trop lent | `/kaggle-efficiency` (GPU, reduce_mem, Polars) |
| Tu veux comprendre les erreurs | `/kaggle-debug` → error analysis sur les worst predictions |

### Jour 14-21 : Ensemble

```
11. /kaggle-ensemble
    → Analyser la diversité, combiner les modèles
    → Tu obtiens : matrice de corrélation + ensemble optimisé

12. /kaggle-experiments (report)
    → Tableau récap de tous les runs
    → Tu obtiens : historique complet CV/LB + meilleur run identifié
```

**Que faire selon les résultats de l'ensemble :**

| Tu observes... | Fais... |
|---|---|
| Corrélation entre modèles > 0.98 | Ajouter de la diversité : `/kaggle-deeplearning`, features différentes, seeds différents |
| Corrélation entre 0.93-0.97 | Bonne diversité. Tester rank average, weighted average, stacking |
| Le stacking overfit (stacking < simple avg) | Revenir au rank average simple. Meta-model trop complexe |
| L'ensemble n'apporte rien | Les modèles sont trop corrélés. Changer d'approche, pas juste de params |

### Derniers jours : Soumission Finale

```
13. /kaggle-leaderboard
    → Analyse shake-up risk + sélection des 2 soumissions finales
    → Tu obtiens : risk score + recommandation conservative/aggressive

14. /kaggle-submit
    → Validation finale du fichier de soumission
    → Tu obtiens : submission.csv validé et prêt
```

**Que faire pour choisir les 2 soumissions finales :**

| Situation | Soumission 1 | Soumission 2 |
|---|---|---|
| CV et LB bien corrélés | Meilleur CV | Meilleur ensemble |
| CV >>> LB (overfit probable) | Le plus stable (lowest CV std) | Meilleur CV avec le moins de features |
| LB >>> CV (CV trop strict) | Meilleur LB | Meilleur CV |
| Risque de shake-up élevé | Meilleur CV (conservative) | Modèle le plus simple/stable |

### En Cas de Blocage

| Problème | Solution |
|---|---|
| "Je ne sais pas quoi faire ensuite" | `/kaggle-experiments` → regarde le tableau, identifie le plus gros gap |
| "Mon score ne monte plus" | `/kaggle-debug` → error analysis → trouver les segments d'erreur |
| "J'ai trop de features" | `/kaggle-explain` → SHAP/permutation importance → supprimer le bruit |
| "Je ne comprends pas mes données" | `/kaggle-eda` + `/kaggle-viz` → revenir aux fondamentaux |
| "Mon CV est instable" | `/kaggle-validation` → changer de split, augmenter n_folds |
| "C'est trop lent" | `/kaggle-efficiency` → reduce_mem, Polars, GPU, caching |
| "Score suspicieusement bon" | `/kaggle-leakage` → audit complet immédiat |
| "Je veux l'avis d'un expert" | kaggle-strategist (agent) → plan d'attaque complet |
| "Pourquoi mon score a baissé" | kaggle-debugger (agent) → diagnostic automatisé |

## Workflow Recommandé - Gold Medal

```
Phase 0 : Stratégie
└── kaggle-strategist (agent) → Plan d'attaque multi-phases
    └── kaggle-researcher (agent) → Recherche solutions gagnantes similaires

Phase 1 : Démarrage Rapide
├── /kaggle-pipeline → Structure du projet
├── /kaggle-baseline → Baseline en <30 min + première soumission
├── /kaggle-eda → Analyse exploratoire complète
├── /kaggle-viz → Visualisations avancées
└── /kaggle-leakage → Audit leakage AVANT d'itérer

Phase 2 : Préparation des Données
├── /kaggle-cleaning → Nettoyage et preprocessing
├── /kaggle-feature → Feature engineering
├── /kaggle-augmentation → Augmentation si nécessaire
├── /kaggle-validation → Définir la stratégie de CV
└── /kaggle-efficiency → Optimiser RAM et vitesse

Phase 3 : Modélisation
├── /kaggle-model → Modèles GBDT (LightGBM, XGBoost, CatBoost)
├── /kaggle-deeplearning → Modèles DL (TabNet, FT-Transformer)
├── kaggle-optimizer (agent) → Hyperparameter tuning
├── /kaggle-experiments → Tracker chaque run (CV, LB, params, features)
└── /kaggle-debug → Diagnostiquer si score baisse

Phase 4 : Ensemble & Soumission
├── /kaggle-ensemble → Combiner les modèles
├── /kaggle-leaderboard → Stratégie de LB et shake-up risk
└── /kaggle-submit → Préparer la soumission finale

En continu :
├── kaggle-debugger (agent) → Diagnostiquer les problèmes
├── /kaggle-experiments → Tracker et comparer les runs
├── /kaggle-explain → Comprendre les prédictions
└── /kaggle-leaderboard → Tracker CV vs LB
```

## Technologies Couvertes

- **ML** : scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **DL** : PyTorch, TensorFlow/Keras, timm, Transformers (HuggingFace)
- **DL Tabulaire** : TabNet, FT-Transformer, SAINT, Entity Embeddings, Autoencoders
- **Data** : Pandas, NumPy, Polars
- **SQL** : Google BigQuery, window functions, CTEs, nested/repeated data
- **Visualisation** : Matplotlib, Seaborn, Plotly, Albumentations
- **Explainability** : SHAP, LIME, eli5, Permutation Importance, PDP/ICE
- **CV** : Albumentations, torchvision, EfficientNet, ViT, Swin, Mask R-CNN, RLE, FocalLoss, DICOM, RGBY
- **NLP** : HuggingFace Transformers, DeBERTa, LoRA/PEFT, back-translation
- **Time Series** : statsmodels, Prophet, features lag/rolling
- **Géospatial** : GeoPandas, Folium, Shapely, geopy, H3, spatial joins
- **Game AI / RL** : minimax, alpha-beta, MCTS, PPO, Stable-Baselines3, self-play
- **Éthique** : 6 types de biais, 4 métriques de fairness, Model Cards, audit éthique, HCD
- **TPU/TF** : tf.distribute, tf.data, TFRecords, TTA, LR schedules Keras, Functional API, Wide & Deep
- **Augmentation** : SMOTE, Mixup, CutMix, pseudo-labeling, back-translation, time series warping
- **Validation** : StratifiedKFold, GroupKFold, StratifiedGroupKFold, PurgedTimeSeriesCV, adversarial validation
- **Ensembling** : weighted/rank average, stacking, blending, hill climbing, multi-seed
- **Leakage** : 7 types de détection (target, contamination, temporal, group, ID, postprocess, external)
- **Tracking** : ExperimentTracker, ablation study, seed stability, config management
- **Performance** : reduce_mem_usage, Polars, feature caching, chunked processing, GPU acceleration
