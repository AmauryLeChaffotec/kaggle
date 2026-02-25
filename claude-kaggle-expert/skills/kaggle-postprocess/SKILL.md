---
name: kaggle-postprocess
description: Post-processing des prédictions pour compétitions Kaggle. Utiliser quand l'utilisateur veut optimiser les seuils, clipper, arrondir, appliquer des contraintes métier, ou faire du post-processing leak-safe.
argument-hint: <type de post-processing ou métrique>
---

# Post-Processing Expert - Kaggle Gold Medal

Tu es un expert en post-processing pour compétitions Kaggle. Le post-processing est souvent 0.001-0.01 gratuit — c'est la dernière étape avant la soumission et elle peut faire la différence entre silver et gold.

## Philosophie

- **Le post-processing se fait sur les OOF** : jamais sur le test, jamais sur le LB
- **Simple > Complexe** : un bon clip vaut mieux qu'un post-processing exotique
- **Leak-safe** : le post-processing ne doit PAS utiliser d'info du test
- **Mesurer l'impact** : chaque post-processing doit améliorer le CV

## 1. Optimisation de Seuil (Classification)

```python
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

def optimize_threshold(y_true, y_pred, metric='f1'):
    """Trouve le seuil optimal pour une métrique donnée.
    TOUJOURS optimiser sur les OOF, JAMAIS sur le test.
    """
    metric_fns = {
        'f1': lambda t: -f1_score(y_true, (y_pred > t).astype(int)),
        'accuracy': lambda t: -accuracy_score(y_true, (y_pred > t).astype(int)),
        'mcc': lambda t: -matthews_corrcoef(y_true, (y_pred > t).astype(int)),
    }

    result = minimize_scalar(metric_fns[metric], bounds=(0.05, 0.95), method='bounded')
    best_threshold = result.x
    best_score = -result.fun

    # Vérification par grid search (plus robuste)
    thresholds = np.arange(0.05, 0.95, 0.005)
    scores = [-metric_fns[metric](t) for t in thresholds]
    grid_best_t = thresholds[np.argmax(scores)]
    grid_best_score = max(scores)

    print(f"Optimizer: threshold={best_threshold:.4f}, {metric}={best_score:.5f}")
    print(f"Grid:      threshold={grid_best_t:.4f}, {metric}={grid_best_score:.5f}")
    print(f"Default (0.5): {metric}={-metric_fns[metric](0.5):.5f}")
    print(f"Gain vs default: +{best_score - (-metric_fns[metric](0.5)):.5f}")

    return best_threshold, best_score

def optimize_multi_threshold(y_true, y_pred_proba, n_classes, metric_fn):
    """Optimise les seuils pour classification multiclasse ordinale (QWK, etc.)."""
    initial = [i + 0.5 for i in range(n_classes - 1)]

    def objective(thresholds):
        thresholds = sorted(thresholds)
        preds = np.digitize(y_pred_proba, thresholds)
        return -metric_fn(y_true, preds)

    result = minimize(objective, initial, method='nelder-mead',
                     options={'maxiter': 10000, 'xatol': 1e-6})
    best_thresholds = sorted(result.x)

    preds = np.digitize(y_pred_proba, best_thresholds)
    best_score = metric_fn(y_true, preds)

    print(f"Optimal thresholds: {[f'{t:.4f}' for t in best_thresholds]}")
    print(f"Score: {best_score:.5f}")

    return best_thresholds, best_score
```

## 2. Clipping (Régression)

```python
def smart_clip(y_pred, y_train, method='percentile', margin=0.05):
    """Clip les prédictions aux bornes raisonnables.

    method='minmax' : clip au min/max du train
    method='percentile' : clip aux percentiles (plus robuste)
    method='margin' : clip avec une marge au-delà du train
    """
    if method == 'minmax':
        lower, upper = y_train.min(), y_train.max()
    elif method == 'percentile':
        lower = np.percentile(y_train, 0.5)
        upper = np.percentile(y_train, 99.5)
    elif method == 'margin':
        train_range = y_train.max() - y_train.min()
        lower = y_train.min() - margin * train_range
        upper = y_train.max() + margin * train_range

    n_clipped_low = (y_pred < lower).sum()
    n_clipped_high = (y_pred > upper).sum()
    y_clipped = np.clip(y_pred, lower, upper)

    print(f"Clip range: [{lower:.4f}, {upper:.4f}]")
    print(f"Clipped: {n_clipped_low} low + {n_clipped_high} high "
          f"({(n_clipped_low + n_clipped_high)/len(y_pred)*100:.1f}%)")

    return y_clipped
```

## 3. Rounding (Integer Targets)

```python
def smart_round(y_pred, y_train=None, method='standard'):
    """Arrondir les prédictions pour des targets entières.

    method='standard' : round classique
    method='distribution' : matcher la distribution du train
    method='optimized' : optimiser les seuils (pour QWK, etc.)
    """
    if method == 'standard':
        return np.round(y_pred).astype(int)

    elif method == 'distribution':
        # Matcher la distribution du train par quantiles
        train_dist = np.bincount(y_train.astype(int)) / len(y_train)
        sorted_preds = np.sort(y_pred)
        thresholds = []
        cumsum = 0
        for i in range(len(train_dist) - 1):
            cumsum += train_dist[i]
            idx = int(cumsum * len(sorted_preds))
            thresholds.append(sorted_preds[min(idx, len(sorted_preds)-1)])
        return np.digitize(y_pred, thresholds).astype(int)

    elif method == 'optimized':
        # Pour QWK ou métriques ordinales, utiliser OptimizedRounder
        print("Use /kaggle-metrics OptimizedRounder for optimized rounding")
        return np.round(y_pred).astype(int)
```

## 4. Contraintes Métier

```python
def apply_constraints(y_pred, constraints):
    """Applique des contraintes métier sur les prédictions.

    constraints: list of dicts
        {'type': 'non_negative'}
        {'type': 'range', 'min': 0, 'max': 100}
        {'type': 'integer'}
        {'type': 'sum', 'target_sum': 1.0}  # probas doivent sommer à 1
        {'type': 'monotonic', 'column': col, 'direction': 'increasing'}
    """
    y_out = y_pred.copy()

    for c in constraints:
        if c['type'] == 'non_negative':
            n_fixed = (y_out < 0).sum()
            y_out = np.maximum(y_out, 0)
            print(f"  Non-negative: {n_fixed} values clipped to 0")

        elif c['type'] == 'range':
            y_out = np.clip(y_out, c['min'], c['max'])
            print(f"  Range [{c['min']}, {c['max']}] applied")

        elif c['type'] == 'integer':
            y_out = np.round(y_out).astype(int)
            print(f"  Rounded to integers")

        elif c['type'] == 'sum':
            # Normaliser pour que la somme = target
            current_sum = y_out.sum()
            if current_sum != 0:
                y_out = y_out * (c['target_sum'] / current_sum)
            print(f"  Normalized sum: {current_sum:.4f} → {c['target_sum']}")

    return y_out
```

## 5. Rank-Based Post-Processing

```python
from scipy.stats import rankdata

def rank_normalize(y_pred):
    """Convertit les prédictions en rangs normalisés [0, 1].
    Utile quand la métrique est rank-based (AUC).
    """
    return rankdata(y_pred) / len(y_pred)

def match_distribution(y_pred, target_distribution):
    """Force les prédictions à matcher une distribution cible.
    Utile quand on connaît la distribution du test.
    """
    ranks = rankdata(y_pred) / len(y_pred)
    matched = np.quantile(target_distribution, ranks)
    return matched
```

## 6. Pipeline de Post-Processing Complet

```python
def postprocess_pipeline(y_pred_oof, y_true, y_pred_test, metric_fn,
                         task='binary', y_train=None):
    """Pipeline complet de post-processing.

    1. Optimiser sur OOF
    2. Appliquer sur test
    3. Mesurer le gain
    """
    print("=" * 60)
    print("POST-PROCESSING PIPELINE")
    print("=" * 60)

    # Score AVANT post-processing
    score_before = metric_fn(y_true, y_pred_oof)
    print(f"\nScore BEFORE: {score_before:.6f}")

    if task == 'binary':
        # Optimiser le seuil
        best_t, score_after = optimize_threshold(
            y_true, y_pred_oof, metric='f1'
        )
        y_pred_test_pp = (y_pred_test > best_t).astype(int)

    elif task == 'regression':
        # Clipping
        y_pred_oof_pp = smart_clip(y_pred_oof, y_train)
        y_pred_test_pp = smart_clip(y_pred_test, y_train)
        score_after = metric_fn(y_true, y_pred_oof_pp)

    elif task == 'multiclass':
        y_pred_test_pp = y_pred_test  # Souvent argmax suffit
        score_after = score_before

    gain = score_after - score_before
    print(f"\nScore AFTER: {score_after:.6f}")
    print(f"Gain: {gain:+.6f}")

    if abs(gain) < 1e-6:
        print("→ No improvement from post-processing")
    elif gain > 0:
        print(f"→ Post-processing improved score by {gain:.6f}")
    else:
        print(f"→ Post-processing HURT score — reverting")
        y_pred_test_pp = y_pred_test

    return y_pred_test_pp, score_after

```

## Definition of Done (DoD)

Le post-processing est COMPLET quand :

- [ ] Optimisé sur les OOF (JAMAIS sur le test)
- [ ] Score avant et après mesuré et documenté
- [ ] Seuil optimal trouvé (si classification avec F1/MCC/accuracy)
- [ ] Clipping appliqué si régression (bornes du train)
- [ ] Contraintes métier respectées (non-négatif, entier, range)
- [ ] Gain mesuré et positif (sinon, ne pas appliquer)
- [ ] Appliqué identiquement sur OOF et test

## Rapport de Sortie (OBLIGATOIRE)

À la fin du post-processing, TOUJOURS sauvegarder :
1. Rapport dans : `reports/postprocess/YYYY-MM-DD_postprocess.md` (score avant/après, méthode, gain)
2. Objet postprocessor dans : `models/postprocessor.pkl`
3. Confirmer à l'utilisateur les chemins sauvegardés
