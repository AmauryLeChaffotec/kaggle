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

## Skills Disponibles (24)

### Skill Auto-chargé (Knowledge)
| Skill | Description |
|-------|-------------|
| `kaggle-knowledge` | Base de connaissances experte chargée automatiquement quand tu travailles sur du data science |

### Skills du Workflow Principal
| Commande | Description | Phase |
|----------|-------------|-------|
| `/kaggle-pipeline` | Pipeline complet de A à Z pour une compétition | Démarrage |
| `/kaggle-eda` | Analyse exploratoire complète d'un dataset | Exploration |
| `/kaggle-cleaning` | Nettoyage et preprocessing des données | Nettoyage |
| `/kaggle-feature` | Feature engineering avancé | Features |
| `/kaggle-validation` | Stratégie de validation croisée robuste | Validation |
| `/kaggle-model` | Construction et optimisation de modèles | Modélisation |
| `/kaggle-debug` | Diagnostic et debugging de modèles | Diagnostic |
| `/kaggle-ensemble` | Stratégies d'ensembling avancées | Ensemble |
| `/kaggle-augmentation` | Augmentation de données (tabulaire, image, texte, time series) | Augmentation |
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

## Workflow Recommandé - Gold Medal

```
Phase 0 : Stratégie
└── kaggle-strategist (agent) → Plan d'attaque multi-phases
    └── kaggle-researcher (agent) → Recherche solutions gagnantes similaires

Phase 1 : Exploration
├── /kaggle-pipeline → Structure du projet
├── /kaggle-eda → Analyse exploratoire complète
└── /kaggle-viz → Visualisations avancées

Phase 2 : Préparation des Données
├── /kaggle-cleaning → Nettoyage et preprocessing
├── /kaggle-feature → Feature engineering
├── /kaggle-augmentation → Augmentation si nécessaire
└── /kaggle-validation → Définir la stratégie de CV

Phase 3 : Modélisation
├── /kaggle-model → Modèles GBDT (LightGBM, XGBoost, CatBoost)
├── /kaggle-deeplearning → Modèles DL (TabNet, FT-Transformer)
├── kaggle-optimizer (agent) → Hyperparameter tuning
└── /kaggle-debug → Diagnostiquer si score baisse

Phase 4 : Ensemble & Soumission
├── /kaggle-ensemble → Combiner les modèles
├── /kaggle-leaderboard → Stratégie de LB et shake-up risk
└── /kaggle-submit → Préparer la soumission finale

En continu :
├── kaggle-debugger (agent) → Diagnostiquer les problèmes
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
