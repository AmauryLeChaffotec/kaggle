---
name: kaggle-metrics
description: Implémentation et vérification de la métrique exacte de la compétition Kaggle. Utiliser quand l'utilisateur veut s'assurer qu'il optimise la bonne métrique, implémenter une métrique custom, ou vérifier que son calcul local correspond au LB.
argument-hint: <nom de la métrique ou description>
---

# Competition Metric Expert - Kaggle Gold Medal

Tu es un expert en métriques de compétition Kaggle. Optimiser la mauvaise métrique ou mal l'implémenter = perdre des centaines de positions. Ton rôle : garantir que la métrique locale est IDENTIQUE à celle du LB.

## Philosophie

- **La métrique EST la compétition** : tout le reste en découle (modèle, seuil, post-processing)
- **Implémenter SOI-MÊME** la métrique : ne jamais se fier uniquement à sklearn
- **Vérifier contre le LB** : soumettre un cas simple pour valider
- **Optimiser POUR la métrique** : pas juste minimiser la loss

## Catalogue des Métriques Kaggle

### Classification Binaire

```python
from sklearn.metrics import (
    roc_auc_score, log_loss, f1_score, accuracy_score,
    precision_score, recall_score, matthews_corrcoef
)
import numpy as np

# === AUC-ROC ===
# Invariant au seuil. Optimiser via logloss ou AUC directement.
def auc_metric(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

# === Log Loss (Cross-Entropy) ===
# ATTENTION : clipping obligatoire pour éviter log(0)
def safe_logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return log_loss(y_true, y_pred)

# === F1 Score ===
# Dépend du SEUIL. Nécessite optimisation du threshold.
def optimized_f1(y_true, y_pred):
    from scipy.optimize import minimize_scalar
    def neg_f1(t):
        return -f1_score(y_true, (y_pred > t).astype(int))
    result = minimize_scalar(neg_f1, bounds=(0.1, 0.9), method='bounded')
    best_threshold = result.x
    best_f1 = -result.fun
    return best_f1, best_threshold

# === Matthews Correlation Coefficient (MCC) ===
# Meilleur que F1 pour données déséquilibrées. Range [-1, 1].
def optimized_mcc(y_true, y_pred):
    best_mcc, best_t = -1, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        mcc = matthews_corrcoef(y_true, (y_pred > t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_mcc, best_t
```

### Classification Multiclasse

```python
# === Accuracy ===
def accuracy_metric(y_true, y_pred_proba):
    return accuracy_score(y_true, y_pred_proba.argmax(axis=1))

# === Macro F1 ===
# Moyenne non pondérée du F1 par classe. Sensible aux classes rares.
def macro_f1(y_true, y_pred_proba):
    y_pred = y_pred_proba.argmax(axis=1)
    return f1_score(y_true, y_pred, average='macro')

# === Weighted Log Loss ===
def weighted_logloss(y_true, y_pred_proba, weights=None):
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    if weights is None:
        return log_loss(y_true, y_pred_proba)
    return log_loss(y_true, y_pred_proba, sample_weight=weights)

# === Quadratic Weighted Kappa (QWK) ===
# Pour classification ordinale (ratings, grades).
from sklearn.metrics import cohen_kappa_score

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

class OptimizedRounder:
    """Optimise les seuils pour QWK."""
    def __init__(self):
        self.coef_ = None

    def fit(self, y_true, y_pred, n_classes=5):
        from scipy.optimize import minimize
        initial = [i + 0.5 for i in range(n_classes - 1)]

        def kappa_loss(coef):
            preds = pd.cut(y_pred, [-np.inf] + sorted(coef) + [np.inf],
                          labels=range(n_classes))
            return -cohen_kappa_score(y_true, preds, weights='quadratic')

        result = minimize(kappa_loss, initial, method='nelder-mead')
        self.coef_ = sorted(result.x)
        return self

    def predict(self, y_pred, n_classes=5):
        return pd.cut(y_pred, [-np.inf] + self.coef_ + [np.inf],
                      labels=range(n_classes)).astype(int)
```

### Régression

```python
# === RMSE ===
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# === RMSLE (Root Mean Squared Log Error) ===
# Pénalise plus les sous-estimations. Prédictions DOIVENT être >= 0.
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
# ASTUCE : si la métrique est RMSLE, entraîner avec target = log1p(y)
# puis prédire avec expm1(pred).

# === MAE ===
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
# ASTUCE : si MAE, optimiser avec objective='mae' ou prédire la MÉDIANE.

# === MAPE (Mean Absolute Percentage Error) ===
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# === R² ===
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
```

### Ranking / Recommandation

```python
# === MAP@K (Mean Average Precision at K) ===
def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0

def mapk(actual_list, predicted_list, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual_list, predicted_list)])

# === NDCG@K ===
def dcg_at_k(scores, k):
    scores = np.array(scores)[:k]
    return np.sum((2**scores - 1) / np.log2(np.arange(2, len(scores) + 2)))

def ndcg_at_k(y_true, y_pred, k=10):
    best = dcg_at_k(sorted(y_true, reverse=True), k)
    if best == 0:
        return 0.0
    return dcg_at_k(y_true[np.argsort(y_pred)[::-1]], k) / best
```

### Segmentation / Détection

```python
# === Dice Coefficient ===
def dice(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

# === IoU (Jaccard) ===
def iou(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).astype(int)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)
```

## Vérification Métrique vs LB

```python
def verify_metric_against_lb(metric_fn, y_true_sample, y_pred_sample,
                              expected_lb_score, tolerance=0.001):
    """Vérifie que ton implémentation locale correspond au LB.

    Étape 1 : Soumettre un fichier de prédictions connues
    Étape 2 : Noter le score LB
    Étape 3 : Calculer localement avec la même métrique
    Étape 4 : Comparer
    """
    local_score = metric_fn(y_true_sample, y_pred_sample)
    diff = abs(local_score - expected_lb_score)

    print(f"Local score: {local_score:.6f}")
    print(f"LB score: {expected_lb_score:.6f}")
    print(f"Diff: {diff:.6f}")

    if diff < tolerance:
        print(f"✓ Metric matches LB (diff < {tolerance})")
    else:
        print(f"⚠ MISMATCH! Check your metric implementation")
        print("  Possible causes:")
        print("  - Averaging method (micro vs macro vs weighted)")
        print("  - Clipping/rounding differences")
        print("  - Sample weights")
        print("  - Probability vs class labels")

    return diff < tolerance
```

## Relation Métrique → Loss → Post-processing

| Métrique | Loss à utiliser | Post-processing |
|---|---|---|
| AUC | `binary_logloss` | Aucun (probas directes) |
| Log Loss | `binary_logloss` | Clip [eps, 1-eps] |
| F1 | `binary_logloss` | Optimiser le seuil sur OOF |
| Accuracy | `binary_logloss` / `multiclass` | argmax ou seuil 0.5 |
| QWK | `regression` ou `multiclass` | OptimizedRounder |
| RMSE | `regression` (MSE) | Clip aux bornes du train |
| RMSLE | `regression` sur log1p(y) | expm1 + clip >= 0 |
| MAE | `regression_l1` (MAE) | Prédire la médiane |
| Dice/IoU | `binary_crossentropy` | Optimiser le seuil |
| MAP@K | LambdaRank ou logloss | Top-K selection |

## Definition of Done (DoD)

La vérification de métrique est COMPLÈTE quand :

- [ ] Métrique de la compétition identifiée et comprise
- [ ] Implémentation locale codée (pas juste sklearn)
- [ ] Vérifiée contre le LB avec un cas simple
- [ ] Loss du modèle alignée avec la métrique (cf. tableau)
- [ ] Post-processing identifié (seuil, rounding, clip)
- [ ] Custom eval metric créée pour LightGBM/XGBoost si nécessaire
