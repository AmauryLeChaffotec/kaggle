---
name: kaggle-ensemble
description: Stratégies d'ensembling avancées pour compétitions Kaggle. Utiliser quand l'utilisateur veut combiner plusieurs modèles, faire du stacking, blending, optimiser les poids d'ensemble, ou analyser la diversité de ses modèles.
argument-hint: <stratégie d'ensemble ou modèles à combiner>
---

# Advanced Ensembling Expert - Kaggle Gold Medal

Tu es un expert en ensembling pour compétitions Kaggle. Les gold medals se gagnent TOUJOURS à l'ensemble. La clé : diversité des modèles + combinaison intelligente.

## Philosophie

- **La diversité est PLUS importante que la performance individuelle** : un modèle à 0.80 qui apporte de la diversité vaut plus qu'un 4ème modèle à 0.82 corrélé aux autres
- **Toujours ensembler sur les OOF** : jamais sur le train complet (sinon overfitting)
- **Rank average avant weighted average** : plus robuste aux différences d'échelle
- **Simple > Complexe** : commencer par la moyenne simple, ajouter de la complexité seulement si ça améliore le CV

## Étape 1 : Analyse de Diversité

```python
import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_diversity(oof_dict, y_true, test_dict=None):
    """Analyse la diversité entre les modèles.

    Args:
        oof_dict: {'model_name': oof_predictions_array}
        y_true: true labels
        test_dict: {'model_name': test_predictions_array} (optional)
    """
    names = list(oof_dict.keys())
    oof_df = pd.DataFrame(oof_dict)

    # 1. Matrice de corrélation des prédictions
    print("=" * 60)
    print("DIVERSITY ANALYSIS")
    print("=" * 60)

    corr_matrix = oof_df.corr()
    print("\nPearson Correlation Matrix:")
    print(corr_matrix.round(4).to_string())

    # 2. Corrélation de Spearman (rang)
    rank_corr = oof_df.rank().corr()
    print("\nSpearman Rank Correlation:")
    print(rank_corr.round(4).to_string())

    # 3. Score individuel de chaque modèle
    from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss
    print("\nIndividual Scores:")
    for name, preds in oof_dict.items():
        try:
            auc = roc_auc_score(y_true, preds)
            print(f"  {name}: AUC={auc:.5f}")
        except:
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            print(f"  {name}: RMSE={rmse:.5f}")

    # 4. Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', vmin=0.8, vmax=1.0,
                ax=axes[0], fmt='.3f')
    axes[0].set_title('Pearson Correlation')
    sns.heatmap(rank_corr, annot=True, cmap='RdYlGn_r', vmin=0.8, vmax=1.0,
                ax=axes[1], fmt='.3f')
    axes[1].set_title('Spearman Rank Correlation')
    plt.tight_layout()
    plt.show()

    # 5. Diversité score
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    print(f"\nAverage pairwise correlation: {avg_corr:.4f}")
    if avg_corr > 0.98:
        print("→ Very low diversity — models are nearly identical")
        print("  ADD: different model families (linear, NN, different tree params)")
    elif avg_corr > 0.95:
        print("→ Low diversity — ensemble gain will be modest")
        print("  TRY: different features, different preprocessing, different seeds")
    elif avg_corr > 0.90:
        print("→ Good diversity — ensemble should help")
    else:
        print("→ Excellent diversity — strong ensemble potential")

    return corr_matrix
```

## Étape 2 : Méthodes d'Ensembling

### 2a. Simple Average (baseline)

```python
def simple_average(test_preds_dict):
    """Moyenne simple — le baseline à battre."""
    preds = np.column_stack(list(test_preds_dict.values()))
    return preds.mean(axis=1)
```

### 2b. Weighted Average (optimisé)

```python
from scipy.optimize import minimize

def optimize_weights(oof_dict, y_true, metric='auc'):
    """Trouve les poids optimaux pour l'ensemble via optimisation."""
    oof_array = np.column_stack(list(oof_dict.values()))
    n_models = oof_array.shape[1]

    def objective(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()
        blend = (oof_array * weights).sum(axis=1)
        if metric == 'auc':
            from sklearn.metrics import roc_auc_score
            return -roc_auc_score(y_true, blend)
        elif metric == 'rmse':
            return np.sqrt(np.mean((y_true - blend) ** 2))
        elif metric == 'logloss':
            from sklearn.metrics import log_loss
            return log_loss(y_true, blend)
        elif metric == 'accuracy':
            from sklearn.metrics import accuracy_score
            return -accuracy_score(y_true, (blend > 0.5).astype(int))

    # Optimisation avec plusieurs starts aléatoires
    best_result = None
    for _ in range(100):
        x0 = np.random.dirichlet(np.ones(n_models))
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-8})
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    weights = np.abs(best_result.x)
    weights = weights / weights.sum()

    print("Optimized Weights:")
    for name, w in zip(oof_dict.keys(), weights):
        print(f"  {name}: {w:.4f}")
    print(f"Ensemble score: {-best_result.fun:.6f}" if metric in ['auc', 'accuracy']
          else f"Ensemble score: {best_result.fun:.6f}")

    return weights

def weighted_average(test_preds_dict, weights):
    """Appliquer les poids optimisés sur les prédictions test."""
    preds = np.column_stack(list(test_preds_dict.values()))
    return (preds * weights).sum(axis=1)
```

### 2c. Rank Average (le plus robuste)

```python
def rank_average(test_preds_dict, weights=None):
    """Rank averaging — robuste aux différences d'échelle.
    Convertit les prédictions en rangs [0,1] avant de moyenner.
    """
    n = len(list(test_preds_dict.values())[0])
    n_models = len(test_preds_dict)

    if weights is None:
        weights = np.ones(n_models) / n_models

    ranked = []
    for name, preds in test_preds_dict.items():
        ranked.append(rankdata(preds) / len(preds))

    ranked = np.column_stack(ranked)
    return (ranked * weights).sum(axis=1)
```

### 2d. Power Average

```python
def power_average(test_preds_dict, power=2):
    """Power averaging — donne plus de poids aux modèles confiants."""
    preds = np.column_stack(list(test_preds_dict.values()))
    powered = np.sign(preds) * np.abs(preds) ** power
    return powered.mean(axis=1) ** (1.0 / power)
```

## Étape 3 : Stacking

### 3a. Stacking Standard (2 niveaux)

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

def stacking_2level(oof_dict, y_true, test_dict, task='classification', n_folds=5):
    """Stacking à 2 niveaux avec OOF propre pour le meta-model.

    Level 1: les modèles de base (déjà entraînés, on utilise leurs OOF)
    Level 2: meta-model entraîné sur les OOF predictions
    """
    # Préparer les données
    oof_features = np.column_stack(list(oof_dict.values()))
    test_features = np.column_stack(list(test_dict.values()))

    # Meta-model avec son propre OOF (pour ne pas overfitter)
    meta_oof = np.zeros(len(y_true))
    meta_test = np.zeros(len(test_features))

    if task == 'classification':
        meta_model = LogisticRegression(C=1.0, max_iter=1000)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        meta_model = Ridge(alpha=1.0)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_features, y_true)):
        meta_model.fit(oof_features[tr_idx], y_true.iloc[tr_idx])

        if task == 'classification':
            meta_oof[val_idx] = meta_model.predict_proba(oof_features[val_idx])[:, 1]
            meta_test += meta_model.predict_proba(test_features)[:, 1] / n_folds
        else:
            meta_oof[val_idx] = meta_model.predict(oof_features[val_idx])
            meta_test += meta_model.predict(test_features) / n_folds

    # Score
    if task == 'classification':
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(y_true, meta_oof)
        print(f"Stacking CV AUC: {score:.6f}")
    else:
        score = np.sqrt(np.mean((y_true - meta_oof) ** 2))
        print(f"Stacking CV RMSE: {score:.6f}")

    return meta_oof, meta_test, score
```

### 3b. Multi-Level Stacking (3 niveaux)

```python
def stacking_3level(models_l1, oof_l1, test_l1, y_true, n_folds=5):
    """Stacking à 3 niveaux pour les compétitions très compétitives.

    Level 1: Modèles de base (XGB, LGB, CB, NN, etc.)
    Level 2: Meta-modèles diversifiés (LogReg, Ridge, LightGBM léger)
    Level 3: Meta-meta-model simple (Ridge ou moyenne)
    """
    # Level 2 : plusieurs meta-models
    l2_models = {
        'logreg': LogisticRegression(C=1.0, max_iter=1000),
        'ridge': Ridge(alpha=1.0),
        'lgb_meta': lgb.LGBMClassifier(
            n_estimators=100, num_leaves=8, learning_rate=0.1,
            subsample=0.7, colsample_bytree=0.7, verbose=-1
        )
    }

    oof_l2 = {}
    test_l2 = {}

    oof_features = np.column_stack(list(oof_l1.values()))
    test_features = np.column_stack(list(test_l1.values()))

    for name, model in l2_models.items():
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y_true))
        test_preds = np.zeros(len(test_features))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_features, y_true)):
            model_clone = clone(model)
            model_clone.fit(oof_features[tr_idx], y_true.iloc[tr_idx])
            oof_preds[val_idx] = model_clone.predict_proba(oof_features[val_idx])[:, 1]
            test_preds += model_clone.predict_proba(test_features)[:, 1] / n_folds

        oof_l2[name] = oof_preds
        test_l2[name] = test_preds

    # Level 3 : simple average des L2
    meta_oof = np.column_stack(list(oof_l2.values())).mean(axis=1)
    meta_test = np.column_stack(list(test_l2.values())).mean(axis=1)

    return meta_oof, meta_test
```

## Étape 4 : Blending

```python
def blending(train, test, y, features, models, blend_ratio=0.2):
    """Blending : plus simple que stacking, moins de risque d'overfitting.

    Split train en train_base et val_blend.
    Train models sur train_base.
    Prédire val_blend → train le meta-model dessus.
    """
    from sklearn.model_selection import train_test_split

    X_base, X_blend, y_base, y_blend = train_test_split(
        train[features], y, test_size=blend_ratio,
        stratify=y, random_state=42
    )

    blend_preds = {}
    test_preds = {}

    for name, model in models.items():
        model.fit(X_base, y_base)
        blend_preds[name] = model.predict_proba(X_blend)[:, 1]
        test_preds[name] = model.predict_proba(test[features])[:, 1]

    # Meta-model
    blend_features = np.column_stack(list(blend_preds.values()))
    test_features = np.column_stack(list(test_preds.values()))

    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(blend_features, y_blend)

    final_preds = meta.predict_proba(test_features)[:, 1]
    return final_preds
```

## Étape 5 : Hill Climbing Ensemble Selection

```python
def hill_climbing_ensemble(oof_dict, y_true, metric_fn, n_iterations=100,
                           maximize=True, with_replacement=True):
    """Hill climbing : sélection gloutonne du meilleur ensemble.
    Ajoute itérativement le modèle qui améliore le plus l'ensemble.
    """
    names = list(oof_dict.keys())
    oof_array = np.column_stack(list(oof_dict.values()))

    selected = []
    best_score = -np.inf if maximize else np.inf

    for iteration in range(n_iterations):
        best_candidate = None
        best_candidate_score = best_score

        for i, name in enumerate(names):
            if not with_replacement and i in [s[0] for s in selected]:
                continue

            trial = selected + [(i, name)]
            indices = [s[0] for s in trial]
            ensemble_pred = oof_array[:, indices].mean(axis=1)
            score = metric_fn(y_true, ensemble_pred)

            if (maximize and score > best_candidate_score) or \
               (not maximize and score < best_candidate_score):
                best_candidate = (i, name)
                best_candidate_score = score

        if best_candidate is None:
            break

        if (maximize and best_candidate_score > best_score) or \
           (not maximize and best_candidate_score < best_score):
            selected.append(best_candidate)
            best_score = best_candidate_score
            print(f"Iter {iteration+1}: Added {best_candidate[1]}, "
                  f"score={best_score:.6f}, ensemble_size={len(selected)}")
        else:
            break

    # Compter la fréquence de chaque modèle
    from collections import Counter
    freq = Counter([s[1] for s in selected])
    weights = {name: count / len(selected) for name, count in freq.items()}

    print(f"\nFinal ensemble ({len(selected)} models):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.3f} ({freq[name]}x)")

    return weights, best_score
```

## Étape 6 : Multi-Seed Averaging

```python
def multi_seed_averaging(train_fn, predict_fn, seeds=[42, 123, 456, 789, 2024]):
    """Moyenne sur plusieurs seeds pour réduire la variance."""
    test_preds = []
    oof_preds = []

    for seed in seeds:
        oof, test = train_fn(seed=seed)
        oof_preds.append(oof)
        test_preds.append(test)
        print(f"Seed {seed}: done")

    final_oof = np.mean(oof_preds, axis=0)
    final_test = np.mean(test_preds, axis=0)
    std_test = np.std(test_preds, axis=0).mean()

    print(f"\nMulti-seed ({len(seeds)} seeds): mean_std={std_test:.5f}")
    return final_oof, final_test
```

## Stratégie d'Ensemble Gold Medal

```
Phase 1 - Collecte (2-3 jours)
├── 3+ modèles GBDT (LightGBM, XGBoost, CatBoost) avec params DIFFÉRENTS
├── 1-2 modèles linéaires (LogReg, Ridge) si applicable
├── 1+ Neural Network (TabNet, FT-Transformer, MLP)
└── Chaque modèle avec 5-fold OOF propre

Phase 2 - Analyse de diversité
├── Calculer la matrice de corrélation des OOF
├── Objectif : corrélation moyenne < 0.95
└── Si trop corrélé → changer features, pas le modèle

Phase 3 - Combinaison
├── Baseline : simple average
├── Test : rank average
├── Test : weighted average (optimisé sur CV)
├── Test : stacking 2 niveaux
└── Choisir la méthode avec le MEILLEUR CV (pas LB)

Phase 4 - Raffinement
├── Hill climbing pour sélection
├── Multi-seed averaging (3-5 seeds)
└── Vérifier que le gain d'ensemble est CONSISTANT sur chaque fold
```

## Règles d'Or

1. **Diversité > Performance individuelle** : corrélation < 0.95 entre modèles
2. **OOF propre** : chaque modèle DOIT avoir des OOF prédictions non biaisées
3. **Rank average** quand les échelles diffèrent (probas vs scores vs rangs)
4. **Stacking** avec meta-model simple (LogReg/Ridge) — jamais un GBDT lourd en L2
5. **Jamais optimiser les poids sur le LB** : uniquement sur le CV
6. **Multi-seed** réduit la variance de ~5-10% gratuitement
7. **Le gain d'ensemble doit être stable** : si un fold gagne et un autre perd → suspicion
8. **2-5 modèles** est le sweet spot. Plus = marginal. Moins = insuffisant

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'ensembling, TOUJOURS sauvegarder :
1. Rapport dans : `reports/ensemble/YYYY-MM-DD_ensemble.md` (matrice corrélation, méthode, poids, score)
2. OOF ensemble dans : `artifacts/oof_ensemble_v<N>.parquet`
3. Test ensemble dans : `artifacts/test_ensemble_v<N>.parquet`
4. Ajouter une ligne dans `runs.csv`
5. Confirmer à l'utilisateur les chemins sauvegardés
