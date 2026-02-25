---
name: kaggle-submit
description: Prépare et valide une soumission Kaggle. Utiliser quand l'utilisateur veut préparer un fichier submission.csv, vérifier son format, ou optimiser ses prédictions finales.
argument-hint: <chemin_submission ou stratégie>
---

# Préparation de Soumission - Expert Kaggle

Tu es un expert en préparation de soumissions Kaggle. Assure-toi que chaque soumission est optimale et correctement formatée.

## Checklist de Soumission

### 1. Vérification du Format

```python
import pandas as pd
import numpy as np

def validate_submission(submission, sample_submission):
    """Valider le format de la soumission."""
    errors = []

    # Vérifier les colonnes
    if list(submission.columns) != list(sample_submission.columns):
        errors.append(f"Colonnes incorrectes: {list(submission.columns)} vs {list(sample_submission.columns)}")

    # Vérifier le nombre de lignes
    if len(submission) != len(sample_submission):
        errors.append(f"Nombre de lignes: {len(submission)} vs {len(sample_submission)} attendu")

    # Vérifier les IDs
    if not (submission.iloc[:, 0] == sample_submission.iloc[:, 0]).all():
        errors.append("Les IDs ne correspondent pas au sample_submission")

    # Vérifier les NaN
    n_nan = submission.isnull().sum().sum()
    if n_nan > 0:
        errors.append(f"{n_nan} valeurs NaN détectées dans la soumission")

    # Vérifier les valeurs infinies
    num_cols = submission.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(submission[num_cols]).sum().sum()
    if n_inf > 0:
        errors.append(f"{n_inf} valeurs infinies détectées")

    # Résultat
    if errors:
        print("ERREURS DÉTECTÉES :")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("Soumission valide !")
        print(f"  Shape: {submission.shape}")
        print(f"  Colonnes: {list(submission.columns)}")
        print(f"  Target stats: min={submission.iloc[:,1].min():.4f}, max={submission.iloc[:,1].max():.4f}, mean={submission.iloc[:,1].mean():.4f}")
        return True
```

### 2. Post-Processing par Type de Métrique

```python
# --- Classification binaire ---
def postprocess_binary(preds, threshold=0.5):
    """Post-processing pour classification binaire."""
    return (preds > threshold).astype(int)

def optimize_threshold(y_true, y_pred_proba, metric='f1'):
    """Trouver le seuil optimal."""
    from sklearn.metrics import f1_score, accuracy_score

    best_threshold = 0.5
    best_score = 0

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_pred_proba > threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    print(f"Seuil optimal: {best_threshold:.2f} (score: {best_score:.6f})")
    return best_threshold

# --- Régression ---
def postprocess_regression(preds, clip_min=None, clip_max=None, round_digits=None):
    """Post-processing pour régression."""
    if clip_min is not None or clip_max is not None:
        preds = np.clip(preds, clip_min, clip_max)
    if round_digits is not None:
        preds = np.round(preds, round_digits)
    return preds

# --- Classification multiclasse ---
def postprocess_multiclass(pred_proba):
    """Post-processing pour classification multiclasse."""
    return np.argmax(pred_proba, axis=1)

# --- Ordinal (QWK) ---
class OptimizedRounder:
    """Optimisation des seuils pour Quadratic Weighted Kappa."""
    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, X, y):
        from sklearn.metrics import cohen_kappa_score
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf],
                     labels=range(len(coef) + 1))
        return -cohen_kappa_score(y, X_p, weights='quadratic')

    def fit(self, X, y):
        from scipy.optimize import minimize
        n_classes = len(np.unique(y))
        initial_coef = [i + 0.5 for i in range(n_classes - 1)]
        self.coef_ = minimize(self._kappa_loss, initial_coef, args=(X, y),
                             method='nelder-mead')['x']
        return self

    def predict(self, X):
        return pd.cut(X, [-np.inf] + list(np.sort(self.coef_)) + [np.inf],
                      labels=range(len(self.coef_) + 1)).astype(int)
```

### 3. Créer la Soumission Finale

```python
def create_submission(test_ids, predictions, id_col='id', target_col='target',
                     filename='submission.csv'):
    """Créer le fichier de soumission."""
    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: predictions
    })

    submission.to_csv(filename, index=False)
    print(f"Soumission sauvegardée: {filename}")
    print(f"  Shape: {submission.shape}")
    print(f"  {target_col} stats:")
    print(f"    min:  {submission[target_col].min()}")
    print(f"    max:  {submission[target_col].max()}")
    print(f"    mean: {submission[target_col].mean():.6f}")
    print(f"    std:  {submission[target_col].std():.6f}")

    # Distribution
    if submission[target_col].nunique() <= 20:
        print(f"  Distribution:")
        print(submission[target_col].value_counts(normalize=True).sort_index())

    return submission
```

### 4. Ensemble de Soumissions

```python
def ensemble_submissions(submission_files, weights=None, method='mean'):
    """Combiner plusieurs fichiers de soumission."""
    subs = [pd.read_csv(f) for f in submission_files]

    id_col = subs[0].columns[0]
    target_col = subs[0].columns[1]

    if weights is None:
        weights = [1/len(subs)] * len(subs)

    if method == 'mean':
        blended = sum(s[target_col] * w for s, w in zip(subs, weights))
    elif method == 'rank':
        from scipy.stats import rankdata
        ranked = [rankdata(s[target_col]) / len(s) for s in subs]
        blended = sum(r * w for r, w in zip(ranked, weights))
    elif method == 'median':
        all_preds = np.column_stack([s[target_col] for s in subs])
        blended = np.median(all_preds, axis=1)

    result = pd.DataFrame({id_col: subs[0][id_col], target_col: blended})
    result.to_csv('submission_ensemble.csv', index=False)

    print(f"Ensemble de {len(subs)} soumissions ({method})")
    print(f"  {target_col}: mean={blended.mean():.6f}, std={blended.std():.6f}")

    return result
```

### 5. Analyse de Soumissions

```python
def compare_submissions(sub_files, labels=None):
    """Comparer visuellement plusieurs soumissions."""
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [f'Sub {i}' for i in range(len(sub_files))]

    subs = [pd.read_csv(f) for f in sub_files]
    target_col = subs[0].columns[1]

    # Corrélation entre soumissions
    corr_data = pd.DataFrame({l: s[target_col] for l, s in zip(labels, subs)})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distributions
    for label, sub in zip(labels, subs):
        sub[target_col].hist(bins=50, alpha=0.5, label=label, ax=axes[0], density=True)
    axes[0].legend()
    axes[0].set_title('Distribution des prédictions')

    # Corrélation
    import seaborn as sns
    sns.heatmap(corr_data.corr(), annot=True, fmt='.4f', cmap='RdBu_r',
                center=1, ax=axes[1])
    axes[1].set_title('Corrélation entre soumissions')

    plt.tight_layout()
    plt.show()

    # Diversité
    print("Diversité entre soumissions:")
    for i in range(len(subs)):
        for j in range(i+1, len(subs)):
            diff = np.abs(subs[i][target_col] - subs[j][target_col]).mean()
            corr = subs[i][target_col].corr(subs[j][target_col])
            print(f"  {labels[i]} vs {labels[j]}: corr={corr:.4f}, mean_diff={diff:.6f}")
```

## Stratégie de Soumission Gold Medal

1. **Première soumission** : baseline simple pour calibrer CV vs LB
2. **Soumissions de développement** : tester des améliorations individuelles
3. **Limiter les soumissions quotidiennes** : ne pas gaspiller, max 2-3/jour
4. **Soumissions finales** : choisir 2 soumissions (conservatrice + agressive)
   - Conservatrice : meilleur score CV stable
   - Agressive : meilleur ensemble qui a bien performé sur LB
5. **Trust your CV** : si ton CV est fiable, fais confiance à ton score local

Adapte TOUJOURS le post-processing à la métrique de la compétition.

## Definition of Done (DoD)

La soumission est COMPLÈTE quand :

- [ ] Format validé (colonnes, types, nombre de lignes = sample_submission)
- [ ] Pas de NaN ni Inf dans les prédictions
- [ ] IDs correspondants entre test et submission
- [ ] Post-processing appliqué si pertinent (seuil, clip, round)
- [ ] Distribution des prédictions vérifiée (cohérente avec le train)
- [ ] CV score et LB score documentés
- [ ] Fichier de soumission versionné (baseline_v1, features_v2, ensemble_v3...)
- [ ] Soumission comparée avec les précédentes (corrélation, delta)
