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
- **Visualisation** : Matplotlib, Seaborn, Plotly
- **CV** : Albumentations, torchvision, EfficientNet, ViT, Swin
- **NLP** : HuggingFace Transformers, DeBERTa, LoRA/PEFT
- **Time Series** : statsmodels, Prophet, features lag/rolling
