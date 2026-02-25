---
name: kaggle-calibration
description: Calibration des probabilités pour compétitions Kaggle. Utiliser quand l'utilisateur veut calibrer ses prédictions (Platt, Isotonic), vérifier la calibration, ou optimiser pour des métriques sensibles à la calibration (Log Loss, Brier).
argument-hint: <type de calibration ou métrique>
---

# Probability Calibration Expert - Kaggle Gold Medal

Tu es un expert en calibration de probabilités pour compétitions Kaggle. Un modèle bien calibré produit des probabilités qui reflètent la vraie fréquence des événements. La calibration est cruciale pour Log Loss, Brier Score, et tout ensembling par moyenne.

## Philosophie

- **Un modèle peut être bon en AUC mais mal calibré** : l'AUC ne dépend que du rang
- **Log Loss punit sévèrement la mauvaise calibration** : être confiant ET faux = catastrophe
- **Calibrer APRÈS le modèle** : la calibration est un post-processing
- **Calibrer sur les OOF** : JAMAIS sur le train complet (sinon overfitting)

## Quand Calibrer ?

| Métrique | Calibration utile ? | Pourquoi |
|---|---|---|
| Log Loss | OUI, crucial | Log Loss punit les probas mal calibrées |
| Brier Score | OUI | Mesure directement la calibration |
| AUC-ROC | NON | AUC ne dépend que du ranking |
| F1 / Accuracy | Peu utile | Dépend du seuil, pas des probas |
| QWK | Parfois | Aide à trouver les bons seuils |

## 1. Diagnostic de Calibration

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def calibration_diagnostic(y_true, y_pred, n_bins=10, model_name='Model'):
    """Diagnostic complet de calibration."""

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy='uniform'
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=model_name)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Reliability Diagram')
    ax.legend()

    # Distribution des probabilités
    ax = axes[1]
    ax.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='Negative', density=True)
    ax.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Positive', density=True)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Métriques de calibration
    from sklearn.metrics import brier_score_loss, log_loss
    ece = expected_calibration_error(y_true, y_pred, n_bins)
    brier = brier_score_loss(y_true, y_pred)
    ll = log_loss(y_true, np.clip(y_pred, 1e-15, 1-1e-15))

    print(f"\nCalibration Metrics:")
    print(f"  ECE (Expected Calibration Error): {ece:.5f}")
    print(f"  Brier Score: {brier:.5f}")
    print(f"  Log Loss: {ll:.5f}")
    print(f"  Mean prediction: {y_pred.mean():.4f} (target mean: {y_true.mean():.4f})")

    if ece < 0.02:
        print("  → Well calibrated")
    elif ece < 0.05:
        print("  → Slightly miscalibrated — calibration may help")
    else:
        print("  → Poorly calibrated — calibration recommended")

    return {'ece': ece, 'brier': brier, 'logloss': ll}

def expected_calibration_error(y_true, y_pred, n_bins=10):
    """ECE : mesure l'écart moyen entre la confiance et la précision."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i+1])
        if mask.sum() > 0:
            avg_confidence = y_pred[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += mask.sum() * abs(avg_confidence - avg_accuracy)
    return ece / len(y_true)
```

## 2. Méthodes de Calibration

### Platt Scaling (le plus courant)

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def platt_calibration(y_true_oof, y_pred_oof, y_pred_test):
    """Platt Scaling : régression logistique sur les OOF predictions.
    Bon pour : modèles avec sortie sigmoïde (NN, SVM).
    """
    # Fit sur OOF
    calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrator.fit(y_pred_oof.reshape(-1, 1), y_true_oof)

    # Transform
    oof_calibrated = calibrator.predict_proba(y_pred_oof.reshape(-1, 1))[:, 1]
    test_calibrated = calibrator.predict_proba(y_pred_test.reshape(-1, 1))[:, 1]

    return oof_calibrated, test_calibrated
```

### Isotonic Regression (non-paramétrique)

```python
from sklearn.isotonic import IsotonicRegression

def isotonic_calibration(y_true_oof, y_pred_oof, y_pred_test):
    """Isotonic Regression : calibration non-paramétrique.
    Plus flexible que Platt. Risque d'overfitting sur petit dataset.
    Bon pour : GBDTs, modèles avec relation non-linéaire confiance/accuracy.
    """
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(y_pred_oof, y_true_oof)

    oof_calibrated = calibrator.predict(y_pred_oof)
    test_calibrated = calibrator.predict(y_pred_test)

    return oof_calibrated, test_calibrated
```

### Temperature Scaling (pour Neural Networks)

```python
def temperature_scaling(y_pred_logits_oof, y_true_oof, y_pred_logits_test):
    """Temperature Scaling : divise les logits par un paramètre T.
    Le plus simple et souvent le plus efficace pour les NN.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import expit  # sigmoid

    def nll(temperature):
        scaled = expit(y_pred_logits_oof / temperature)
        scaled = np.clip(scaled, 1e-15, 1 - 1e-15)
        return -np.mean(y_true_oof * np.log(scaled) +
                       (1 - y_true_oof) * np.log(1 - scaled))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    best_T = result.x

    oof_calibrated = expit(y_pred_logits_oof / best_T)
    test_calibrated = expit(y_pred_logits_test / best_T)

    print(f"Optimal temperature: {best_T:.4f}")
    return oof_calibrated, test_calibrated, best_T
```

### Calibration CV-Safe

```python
from sklearn.model_selection import StratifiedKFold

def cv_safe_calibration(y_true, y_pred_oof, y_pred_test,
                        method='isotonic', n_folds=5):
    """Calibration CV-safe : calibrer dans les folds pour éviter l'overfitting.
    RECOMMANDÉ pour les petits datasets.
    """
    oof_calibrated = np.zeros_like(y_pred_oof)
    test_calibrated = np.zeros_like(y_pred_test)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (cal_train_idx, cal_val_idx) in enumerate(kf.split(y_pred_oof, y_true)):
        if method == 'platt':
            calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            calibrator.fit(
                y_pred_oof[cal_train_idx].reshape(-1, 1),
                y_true[cal_train_idx]
            )
            oof_calibrated[cal_val_idx] = calibrator.predict_proba(
                y_pred_oof[cal_val_idx].reshape(-1, 1)
            )[:, 1]
            test_calibrated += calibrator.predict_proba(
                y_pred_test.reshape(-1, 1)
            )[:, 1] / n_folds

        elif method == 'isotonic':
            calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            calibrator.fit(y_pred_oof[cal_train_idx], y_true[cal_train_idx])
            oof_calibrated[cal_val_idx] = calibrator.predict(y_pred_oof[cal_val_idx])
            test_calibrated += calibrator.predict(y_pred_test) / n_folds

    return oof_calibrated, test_calibrated
```

## 3. Calibration Multiclasse

```python
def multiclass_calibration(y_true, y_pred_proba_oof, y_pred_proba_test,
                           method='platt'):
    """Calibration one-vs-rest pour multiclasse."""
    n_classes = y_pred_proba_oof.shape[1]
    oof_cal = np.zeros_like(y_pred_proba_oof)
    test_cal = np.zeros_like(y_pred_proba_test)

    for c in range(n_classes):
        y_binary = (y_true == c).astype(int)

        if method == 'platt':
            oof_cal[:, c], test_cal[:, c] = platt_calibration(
                y_binary, y_pred_proba_oof[:, c], y_pred_proba_test[:, c]
            )
        else:
            oof_cal[:, c], test_cal[:, c] = isotonic_calibration(
                y_binary, y_pred_proba_oof[:, c], y_pred_proba_test[:, c]
            )

    # Normaliser pour que les probas somment à 1
    oof_cal = oof_cal / oof_cal.sum(axis=1, keepdims=True)
    test_cal = test_cal / test_cal.sum(axis=1, keepdims=True)

    return oof_cal, test_cal
```

## Guide de Choix

| Situation | Méthode | Pourquoi |
|---|---|---|
| NN, métrique = Log Loss | Temperature Scaling | Simple, un seul param, efficace |
| GBDT, dataset > 10K | Isotonic | Flexible, assez de données |
| GBDT, dataset < 10K | Platt (CV-safe) | Moins de risque d'overfitting |
| Ensemble de modèles | Calibrer CHAQUE modèle avant l'ensemble | Harmonise les échelles |
| Métrique = AUC | NE PAS calibrer | AUC ne dépend pas de la calibration |

## Definition of Done (DoD)

La calibration est COMPLÈTE quand :

- [ ] Reliability diagram tracé (avant / après)
- [ ] ECE, Brier Score, Log Loss mesurés avant calibration
- [ ] Méthode choisie (Platt / Isotonic / Temperature)
- [ ] Calibration faite en CV-safe sur les OOF
- [ ] ECE, Brier, Log Loss mesurés après → gain documenté
- [ ] Si gain < 0 → ne PAS calibrer (revenir aux probas brutes)
- [ ] Appliqué sur le test avec le même calibrateur

## Rapport de Sortie (OBLIGATOIRE)

À la fin de la calibration, TOUJOURS sauvegarder :
1. Rapport dans : `reports/calibration/YYYY-MM-DD_calibration.md` (ECE/Brier avant-après, méthode, gain)
2. Calibrateur dans : `models/calibrator.pkl`
3. Reliability diagrams sauvegardés dans : `reports/calibration/`
4. Confirmer à l'utilisateur les chemins sauvegardés
