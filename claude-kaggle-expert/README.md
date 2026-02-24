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

## Skills Disponibles

### Skill Auto-chargé (Knowledge)
| Skill | Description |
|-------|-------------|
| `kaggle-knowledge` | Base de connaissances experte chargée automatiquement quand tu travailles sur du data science |

### Skills Invocables (commandes)
| Commande | Description |
|----------|-------------|
| `/kaggle-eda` | Analyse exploratoire complète d'un dataset |
| `/kaggle-feature` | Feature engineering avancé |
| `/kaggle-model` | Construction et optimisation de modèles |
| `/kaggle-submit` | Préparation et validation de soumission |
| `/kaggle-pipeline` | Pipeline complet de A à Z pour une compétition |
| `/kaggle-tabular` | Stratégie experte pour compétitions tabulaires |
| `/kaggle-cv` | Stratégie experte pour compétitions Computer Vision |
| `/kaggle-nlp` | Stratégie experte pour compétitions NLP |
| `/kaggle-timeseries` | Stratégie experte pour compétitions Time Series |
| `/kaggle-sql` | SQL & BigQuery : extraction de données, window functions, optimisation |
| `/kaggle-explain` | ML Explainability : SHAP, LIME, PDP, Permutation Importance |
| `/kaggle-viz` | Visualisation avancée : Seaborn, Matplotlib, Plotly, subplots, compositions |
| `/kaggle-geospatial` | Analyse géospatiale : GeoPandas, Folium, CRS, spatial joins, proximité |
| `/kaggle-rl` | Game AI & Reinforcement Learning : minimax, MCTS, PPO, self-play |

### Agents
| Agent | Description |
|-------|-------------|
| `kaggle-researcher` | Analyse de compétition, recherche de solutions gagnantes |
| `kaggle-optimizer` | Optimisation de modèles et hyperparamètres |

## Workflow Recommandé

1. `/kaggle-pipeline` pour démarrer une nouvelle compétition
2. `/kaggle-eda` pour analyser les données
3. `/kaggle-feature` pour créer des features
4. `/kaggle-model` pour construire et entraîner des modèles
5. `/kaggle-submit` pour préparer la soumission

## Technologies Couvertes

- **ML** : scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **DL** : PyTorch, TensorFlow/Keras, timm, Transformers (HuggingFace)
- **Data** : Pandas, NumPy, Polars
- **SQL** : Google BigQuery, window functions, CTEs, nested/repeated data
- **Visualisation** : Matplotlib, Seaborn, Plotly, Albumentations
- **Explainability** : SHAP, LIME, eli5, Permutation Importance, PDP/ICE
- **CV** : Albumentations, torchvision, EfficientNet, ViT, Swin
- **NLP** : HuggingFace Transformers, DeBERTa, LoRA/PEFT
- **Time Series** : statsmodels, Prophet, features lag/rolling
- **Géospatial** : GeoPandas, Folium, Shapely, geopy, H3, spatial joins
- **Game AI / RL** : minimax, alpha-beta, MCTS, PPO, Stable-Baselines3, self-play
