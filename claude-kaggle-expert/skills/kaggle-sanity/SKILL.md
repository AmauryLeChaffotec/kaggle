---
name: kaggle-sanity
description: Tests de sanité et vérifications rapides pour compétitions Kaggle. Utiliser quand l'utilisateur veut vérifier que son pipeline est correct, que le preprocessing est dans le fold, que les features sont utiles, ou faire un check rapide avant de soumettre.
argument-hint: <type de vérification ou chemin du notebook>
---

# Sanity Checks Expert - Kaggle Gold Medal

Tu es un expert en vérifications de sanité pour compétitions Kaggle. Les sanity checks sont les garde-fous qui empêchent de perdre des heures sur un bug silencieux. 5 minutes de vérification = des jours de travail sauvés.

## Philosophie

- **Si tu ne l'as pas vérifié, c'est faux** : chaque étape du pipeline peut cacher un bug
- **Les bugs Kaggle sont SILENCIEUX** : pas d'erreur, juste un mauvais score
- **Vérifier est RAPIDE** : chaque check prend < 1 minute
- **Exécuter SOUVENT** : après chaque changement significatif

## 1. Sanity Check : Permutation Target

```python
def permutation_target_test(model_fn, X, y, n_trials=3):
    """TEST CRITIQUE : si le modèle score bien avec des labels shufflés,
    c'est du leakage.

    model_fn: function(X_train, y_train, X_val, y_val) → score
    """
    from sklearn.model_selection import train_test_split

    # Score normal
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    real_score = model_fn(X_tr, y_tr, X_val, y_val)

    # Score avec labels permutés
    shuffled_scores = []
    for i in range(n_trials):
        y_shuffled = np.random.permutation(y)
        y_tr_s, y_val_s = y_shuffled[:len(X_tr)], y_shuffled[len(X_tr):]
        score = model_fn(X_tr, y_tr_s, X_val, y_val_s)
        shuffled_scores.append(score)

    avg_shuffled = np.mean(shuffled_scores)
    print(f"Real score: {real_score:.5f}")
    print(f"Shuffled score: {avg_shuffled:.5f} ± {np.std(shuffled_scores):.5f}")
    print(f"Ratio: {real_score / avg_shuffled:.2f}x")

    if avg_shuffled > real_score * 0.8:
        print("⚠ CRITICAL: Model scores almost as well on shuffled labels!")
        print("  → LEAKAGE is almost certain")
        return False
    elif avg_shuffled > real_score * 0.5:
        print("⚠ WARNING: Shuffled score is suspiciously high")
        return False
    else:
        print("✓ Model relies on real signal, not leakage")
        return True
```

## 2. Sanity Check : Preprocessing dans le Fold

```python
def check_preprocessing_in_fold(pipeline_code_description=None):
    """Checklist : le preprocessing est-il DANS le fold ?

    Le preprocessing HORS du fold = data leakage silencieux.
    """
    checks = {
        'Scaling (StandardScaler, etc.)': {
            'danger': 'fit_transform sur tout le train avant le split',
            'correct': 'fit sur X_train du fold, transform sur X_val',
        },
        'Imputation (median, mean, KNN)': {
            'danger': 'Calculer médiane/mean sur tout le train',
            'correct': 'Calculer sur X_train du fold, appliquer sur X_val',
        },
        'Target Encoding': {
            'danger': 'Encoder sur tout le train',
            'correct': 'OOF encoding : encoder dans les folds internes',
        },
        'Feature Selection': {
            'danger': 'Sélectionner les features sur tout le train',
            'correct': 'Sélectionner dans chaque fold (ou fixer avant CV)',
        },
        'PCA / SVD': {
            'danger': 'fit sur tout le train',
            'correct': 'fit sur X_train du fold, transform sur X_val',
        },
        'Outlier Removal': {
            'danger': 'Définir les seuils sur tout le train',
            'correct': 'Seuils sur X_train du fold (ou fixer avant CV)',
        },
    }

    print("=" * 60)
    print("PREPROCESSING IN-FOLD CHECKLIST")
    print("=" * 60)
    for step, info in checks.items():
        print(f"\n  [{step}]")
        print(f"    DANGER: {info['danger']}")
        print(f"    CORRECT: {info['correct']}")

    return checks
```

## 3. Sanity Check : Features Utiles

```python
def check_features_useful(model, X_train, y_train, X_val, y_val, features,
                          metric_fn, n_random=5):
    """Vérifie que les features apportent plus qu'un bruit aléatoire."""
    # Score avec les vraies features
    model.fit(X_train[features], y_train)
    real_score = metric_fn(y_val, model.predict(X_val[features]))

    # Score avec des features aléatoires
    random_scores = []
    for i in range(n_random):
        X_random_tr = pd.DataFrame(
            np.random.randn(len(X_train), len(features)),
            columns=features
        )
        X_random_val = pd.DataFrame(
            np.random.randn(len(X_val), len(features)),
            columns=features
        )
        model.fit(X_random_tr, y_train)
        score = metric_fn(y_val, model.predict(X_random_val))
        random_scores.append(score)

    avg_random = np.mean(random_scores)
    print(f"Real features score: {real_score:.5f}")
    print(f"Random features score: {avg_random:.5f}")
    print(f"Lift: {real_score - avg_random:+.5f}")

    if real_score <= avg_random * 1.05:
        print("⚠ CRITICAL: Real features are no better than random!")
        print("  → Check your feature pipeline")
        return False
    else:
        print("✓ Features carry real signal")
        return True
```

## 4. Sanity Check : Train sur Subset

```python
def subset_sanity_check(train_fn, X, y, subsets=[0.1, 0.3, 0.5, 1.0]):
    """Vérifie que le score augmente avec plus de données.
    Si ce n'est pas le cas → problème dans le pipeline.
    """
    scores = []

    for ratio in subsets:
        n = int(len(X) * ratio)
        X_sub = X.iloc[:n]
        y_sub = y.iloc[:n]
        score = train_fn(X_sub, y_sub)
        scores.append(score)
        print(f"  {ratio*100:.0f}% ({n:,} samples): score={score:.5f}")

    # Vérifier la tendance
    is_increasing = all(scores[i] <= scores[i+1] * 1.01 for i in range(len(scores)-1))

    if is_increasing:
        print("✓ Score increases with more data (expected behavior)")
    else:
        print("⚠ WARNING: Score does NOT increase with more data")
        print("  → Possible causes: overfitting, leakage, or bad split")

    return scores
```

## 5. Sanity Check : Soumission

```python
def submission_sanity_check(submission_path, sample_submission_path, y_train=None):
    """Vérifications avant de soumettre."""
    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_submission_path)

    checks_passed = 0
    total_checks = 0

    def check(name, condition, detail=''):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if condition:
            checks_passed += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} — {detail}")

    print("=" * 60)
    print("SUBMISSION SANITY CHECK")
    print("=" * 60)

    # Shape
    check("Row count", sub.shape[0] == sample.shape[0],
          f"Expected {sample.shape[0]}, got {sub.shape[0]}")

    # Colonnes
    check("Column names", list(sub.columns) == list(sample.columns),
          f"Expected {list(sample.columns)}, got {list(sub.columns)}")

    # NaN
    n_nan = sub.isnull().sum().sum()
    check("No NaN values", n_nan == 0, f"Found {n_nan} NaN")

    # Inf
    numeric_cols = sub.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(sub[numeric_cols]).sum().sum()
    check("No Inf values", n_inf == 0, f"Found {n_inf} Inf")

    # IDs
    target_col = sample.columns[-1]
    id_col = sample.columns[0]
    if id_col in sub.columns and id_col in sample.columns:
        ids_match = set(sub[id_col]) == set(sample[id_col])
        check("IDs match sample", ids_match,
              f"Missing/extra IDs detected")

    # Distribution des prédictions
    pred_mean = sub[target_col].mean()
    pred_std = sub[target_col].std()
    print(f"\n  Prediction stats: mean={pred_mean:.4f}, std={pred_std:.4f}, "
          f"min={sub[target_col].min():.4f}, max={sub[target_col].max():.4f}")

    # Comparaison avec la distribution du train
    if y_train is not None:
        train_mean = y_train.mean()
        drift = abs(pred_mean - train_mean) / train_mean if train_mean != 0 else 0
        check("Prediction mean close to train",
              drift < 0.5,
              f"Pred mean={pred_mean:.4f}, Train mean={train_mean:.4f}, Drift={drift:.2f}")

    # Vérifier que ce ne sont pas toutes les mêmes prédictions
    check("Predictions vary", pred_std > 1e-6,
          "All predictions are identical!")

    # Résumé
    print(f"\n  Checks passed: {checks_passed}/{total_checks}")
    if checks_passed == total_checks:
        print("  ✓ SUBMISSION LOOKS GOOD — ready to upload")
    else:
        print("  ⚠ FIX THE ISSUES BEFORE SUBMITTING")

    return checks_passed == total_checks
```

## 6. Invariants Check

```python
def check_invariants(train, test, target, id_col):
    """Vérifie les invariants qui doivent TOUJOURS être vrais."""

    print("=" * 60)
    print("INVARIANTS CHECK")
    print("=" * 60)

    # 1. Pas de target dans le test
    assert target not in test.columns, f"TARGET '{target}' found in test!"
    print("  ✓ Target not in test")

    # 2. ID unique dans train et test
    assert train[id_col].nunique() == len(train), "Duplicate IDs in train!"
    assert test[id_col].nunique() == len(test), "Duplicate IDs in test!"
    print("  ✓ IDs are unique")

    # 3. Pas d'overlap d'IDs
    overlap = set(train[id_col]) & set(test[id_col])
    if overlap:
        print(f"  ⚠ {len(overlap)} IDs overlap between train and test")
    else:
        print("  ✓ No ID overlap between train and test")

    # 4. Mêmes colonnes (sauf target)
    train_cols = set(train.columns) - {target}
    test_cols = set(test.columns)
    missing_in_test = train_cols - test_cols
    extra_in_test = test_cols - train_cols
    if missing_in_test:
        print(f"  ⚠ Columns in train but not test: {missing_in_test}")
    if extra_in_test:
        print(f"  ⚠ Columns in test but not train: {extra_in_test}")
    if not missing_in_test and not extra_in_test:
        print("  ✓ Same columns in train and test (except target)")

    # 5. Pas de colonne constante
    constant_cols = [c for c in train.columns if train[c].nunique() <= 1]
    if constant_cols:
        print(f"  ⚠ Constant columns: {constant_cols}")
    else:
        print("  ✓ No constant columns")

    print("  DONE")
```

## Quick Sanity Suite

```python
def full_sanity_suite(train, test, target, id_col, features,
                      model_fn=None, metric_fn=None):
    """Suite complète de sanity checks en une commande."""

    print("=" * 60)
    print("FULL SANITY SUITE")
    print("=" * 60)

    # 1. Invariants
    print("\n--- 1. Invariants ---")
    check_invariants(train, test, target, id_col)

    # 2. Preprocessing in fold
    print("\n--- 2. Preprocessing Checklist ---")
    check_preprocessing_in_fold()

    # 3. Permutation test (si modèle fourni)
    if model_fn and metric_fn:
        print("\n--- 3. Permutation Target Test ---")
        permutation_target_test(model_fn, train[features], train[target])

    # 4. Feature usefulness (si modèle fourni)
    if model_fn:
        print("\n--- 4. Feature Usefulness ---")
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            train[features], train[target], test_size=0.2, random_state=42
        )
        check_features_useful(model_fn(), X_tr, y_tr, X_val, y_val,
                             features, metric_fn)

    print("\n" + "=" * 60)
    print("SANITY SUITE COMPLETE")
    print("=" * 60)
```

## Definition of Done (DoD)

Les sanity checks sont COMPLETS quand :

- [ ] Invariants vérifiés (IDs uniques, pas de target dans test, colonnes alignées)
- [ ] Permutation target test passé (shuffled score << real score)
- [ ] Preprocessing in-fold vérifié (scaling, imputation, encoding)
- [ ] Features utiles vs random confirmé
- [ ] Score augmente avec plus de données (subset test)
- [ ] Soumission validée (shape, NaN, Inf, IDs, distribution)
