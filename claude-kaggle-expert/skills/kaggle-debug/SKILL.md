---
name: kaggle-debug
description: Diagnostic et debugging de modèles pour compétitions Kaggle. Utiliser quand le score baisse, le modèle overfit, le CV ne corrèle pas avec le LB, ou pour analyser les erreurs de prédiction.
argument-hint: <description du problème ou chemin du notebook>
---

# Model Debugging & Error Analysis Expert - Kaggle Gold Medal

Tu es un expert en diagnostic de modèles pour compétitions Kaggle. Ton rôle est d'identifier POURQUOI un modèle sous-performe et de proposer des actions correctives ciblées.

## Philosophie

- **Diagnostiquer avant d'agir** : comprendre le problème avant d'essayer des solutions
- **Un changement à la fois** : isoler l'effet de chaque modification
- **Les données > le modèle** : 90% des problèmes viennent des données, pas du modèle
- **Mesurer tout** : chaque hypothèse doit être vérifiable

## Arbre de Diagnostic Principal

```
Le score a baissé / est mauvais ?
│
├── Le CV est-il stable entre les folds ?
│   ├── NON (variance > 5%) → Problème de VALIDATION
│   │   └── Voir Section 1
│   └── OUI → Le CV corrèle-t-il avec le LB ?
│       ├── NON (gap > 3%) → Problème de LEAKAGE ou DRIFT
│       │   └── Voir Section 2
│       └── OUI → Le score est simplement bas
│           ├── Underfitting ? → Voir Section 3
│           └── Overfitting ? → Voir Section 4
│
├── Le score a DROP par rapport à une version précédente ?
│   └── Voir Section 5 (Regression Testing)
│
└── Le modèle fait des erreurs bizarres sur certains exemples ?
    └── Voir Section 6 (Error Analysis)
```

## Section 1 : Diagnostic de Validation Instable

```python
def diagnose_cv_instability(cv_scores, threshold=0.05):
    """Diagnostique l'instabilité du CV."""
    scores = np.array(cv_scores)
    mean_score = scores.mean()
    std_score = scores.std()
    cv_coeff = std_score / mean_score if mean_score != 0 else float('inf')

    print(f"CV Scores: {scores.round(5)}")
    print(f"Mean: {mean_score:.5f} ± {std_score:.5f}")
    print(f"CV coefficient: {cv_coeff:.4f}")
    print(f"Min fold: {scores.min():.5f}, Max fold: {scores.max():.5f}")
    print(f"Range: {scores.max() - scores.min():.5f}")

    if cv_coeff > threshold:
        print(f"\n⚠ UNSTABLE CV (coefficient > {threshold})")
        print("\nDiagnostic possible :")
        print("  1. Dataset trop petit → augmenter n_folds (10+) ou RepeatedKFold")
        print("  2. Mauvais split → vérifier si les données sont groupées/temporelles")
        print("  3. Target leak dans un fold → vérifier le preprocessing pipeline")
        print("  4. Outliers concentrés dans un fold → analyser la distribution par fold")
        print("  5. Features instables → feature importance par fold pour détecter")

    return {
        'mean': mean_score, 'std': std_score,
        'cv_coeff': cv_coeff, 'is_stable': cv_coeff <= threshold
    }

def analyze_fold_differences(oof_preds, y_true, fold_indices, features, train_df):
    """Analyse les différences entre folds pour trouver la source d'instabilité."""
    results = []

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        val_preds = oof_preds[val_idx]
        val_true = y_true.iloc[val_idx]

        # Score par fold
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(val_true, val_preds)

        # Stats du fold
        result = {
            'fold': fold,
            'score': score,
            'n_samples': len(val_idx),
            'target_mean': val_true.mean(),
            'pred_mean': val_preds.mean(),
        }

        # Stats des features
        for feat in features[:5]:  # Top 5 features
            result[f'{feat}_mean'] = train_df.iloc[val_idx][feat].mean()

        results.append(result)

    df = pd.DataFrame(results)
    print("\nFold Analysis:")
    print(df.to_string(index=False))
    return df
```

## Section 2 : Diagnostic de Leakage et Drift

```python
def diagnose_leakage(train, test, features, target, oof_preds, cv_score, lb_score):
    """Diagnostique le leakage et le drift train/test."""
    gap = abs(cv_score - lb_score)
    gap_ratio = gap / cv_score if cv_score != 0 else 0

    print(f"CV Score: {cv_score:.5f}")
    print(f"LB Score: {lb_score:.5f}")
    print(f"Gap: {gap:.5f} ({gap_ratio*100:.2f}%)")

    if cv_score > lb_score and gap_ratio > 0.03:
        print("\n⚠ CV >> LB — OVERFITTING detected")
        print("\nCauses possibles :")
        print("  1. Target leakage dans les features")
        print("  2. Target encoding sans CV-safe")
        print("  3. Preprocessing avant le split (scaling, imputation)")
        print("  4. Features basées sur des agrégations globales")
        print("  5. Train/test drift (adversarial validation)")

        # Check feature par feature
        print("\n--- Feature Leakage Check ---")
        for feat in features:
            if feat in train.columns:
                # Corrélation feature-target suspicieusement élevée
                corr = abs(train[feat].corr(train[target]))
                if corr > 0.95:
                    print(f"  ⚠ {feat}: corr with target = {corr:.4f} — SUSPICIOUS")

    elif lb_score > cv_score and gap_ratio > 0.03:
        print("\n⚠ LB >> CV — Possible issues:")
        print("  1. CV trop conservateur (trop de régularisation)")
        print("  2. Test set plus facile que train")
        print("  3. Chance (petit LB public)")

def detect_feature_leakage(train, target, features, threshold=0.95):
    """Détecte les features qui leakent la target."""
    import lightgbm as lgb

    leaky_features = []

    # Méthode 1 : corrélation directe
    for feat in features:
        if train[feat].dtype in ['float64', 'int64']:
            corr = abs(train[feat].corr(train[target]))
            if corr > threshold:
                leaky_features.append((feat, 'direct_corr', corr))

    # Méthode 2 : single-feature model AUC
    for feat in features:
        try:
            model = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
            model.fit(train[[feat]], train[target])
            pred = model.predict_proba(train[[feat]])[:, 1]
            auc = roc_auc_score(train[target], pred)
            if auc > threshold:
                leaky_features.append((feat, 'single_feat_auc', auc))
        except:
            pass

    if leaky_features:
        print("⚠ POTENTIAL LEAKY FEATURES:")
        for feat, method, score in sorted(leaky_features, key=lambda x: -x[2]):
            print(f"  {feat}: {method}={score:.4f}")
    else:
        print("✓ No obvious leakage detected")

    return leaky_features
```

## Section 3 : Diagnostic d'Underfitting

```python
def diagnose_underfitting(train_scores, val_scores):
    """Diagnostique l'underfitting (train score ≈ val score, tous les deux bas)."""
    train_mean = np.mean(train_scores)
    val_mean = np.mean(val_scores)
    train_val_gap = train_mean - val_mean

    print(f"Train Score: {train_mean:.5f}")
    print(f"Val Score: {val_mean:.5f}")
    print(f"Train-Val Gap: {train_val_gap:.5f}")

    if train_val_gap < 0.02 and val_mean < 0.85:  # Adapter le seuil
        print("\n⚠ UNDERFITTING detected (train ≈ val, both low)")
        print("\nActions recommandées :")
        print("  1. Augmenter la complexité du modèle (plus de leaves, plus de depth)")
        print("  2. Réduire la régularisation (alpha, lambda, min_child)")
        print("  3. Ajouter des features (interactions, agrégations, engineering)")
        print("  4. Augmenter n_estimators + baisser learning_rate")
        print("  5. Essayer un modèle plus expressif (NN, stacking)")
    elif train_val_gap > 0.05:
        print("\n⚠ OVERFITTING detected (train >> val)")
        print("→ Voir Section 4")
    else:
        print("\n✓ Good fit (reasonable train-val gap)")
```

## Section 4 : Diagnostic d'Overfitting

```python
def diagnose_overfitting(model, X_train, y_train, X_val, y_val, features):
    """Diagnostique complet de l'overfitting."""
    from sklearn.metrics import roc_auc_score

    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]
    train_score = roc_auc_score(y_train, train_pred)
    val_score = roc_auc_score(y_val, val_pred)
    gap = train_score - val_score

    print(f"Train AUC: {train_score:.5f}")
    print(f"Val AUC: {val_score:.5f}")
    print(f"Overfit Gap: {gap:.5f}")

    if gap > 0.05:
        print(f"\n⚠ OVERFITTING (gap > 5%)")

        # Identifier les features qui causent l'overfit
        print("\n--- Feature Overfitting Analysis ---")
        imp = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Features avec haute importance mais faible contribution au val
        print("Top features by importance (vérifier si elles overfittent) :")
        print(imp.head(10).to_string(index=False))

        print("\nActions recommandées :")
        actions = [
            ("Augmenter min_child_samples/min_child_weight", gap > 0.05),
            ("Réduire num_leaves / max_depth", gap > 0.05),
            ("Augmenter reg_alpha / reg_lambda", gap > 0.03),
            ("Réduire feature_fraction / colsample_bytree", gap > 0.03),
            ("Réduire bagging_fraction / subsample", gap > 0.03),
            ("Feature selection (drop les features bruit)", gap > 0.05),
            ("Early stopping plus agressif", gap > 0.02),
            ("Augmenter le dropout (NN)", gap > 0.05),
            ("Réduire le nombre de features", len(features) > 50),
        ]
        for action, condition in actions:
            if condition:
                print(f"  → {action}")

    return {'train': train_score, 'val': val_score, 'gap': gap}
```

## Section 5 : Regression Testing (score qui a baissé)

```python
def compare_versions(oof_v1, oof_v2, y_true, features_v1=None, features_v2=None):
    """Compare deux versions de modèle pour comprendre le drop."""
    from sklearn.metrics import roc_auc_score

    score_v1 = roc_auc_score(y_true, oof_v1)
    score_v2 = roc_auc_score(y_true, oof_v2)

    print(f"V1 Score: {score_v1:.5f}")
    print(f"V2 Score: {score_v2:.5f}")
    print(f"Delta: {score_v2 - score_v1:+.5f}")

    # Quelles observations ont changé de prédiction ?
    diff = oof_v2 - oof_v1
    print(f"\nPrediction changes:")
    print(f"  Mean diff: {diff.mean():.5f}")
    print(f"  Std diff: {diff.std():.5f}")
    print(f"  Max increase: {diff.max():.5f}")
    print(f"  Max decrease: {diff.min():.5f}")

    # Observations qui se sont améliorées vs détériorées
    improved = ((np.abs(oof_v2 - y_true) < np.abs(oof_v1 - y_true))).sum()
    worsened = ((np.abs(oof_v2 - y_true) > np.abs(oof_v1 - y_true))).sum()
    print(f"\n  Improved: {improved} samples ({improved/len(y_true)*100:.1f}%)")
    print(f"  Worsened: {worsened} samples ({worsened/len(y_true)*100:.1f}%)")

    # Features ajoutées / supprimées
    if features_v1 and features_v2:
        added = set(features_v2) - set(features_v1)
        removed = set(features_v1) - set(features_v2)
        if added:
            print(f"\n  Added features: {added}")
        if removed:
            print(f"  Removed features: {removed}")

    return {
        'score_v1': score_v1, 'score_v2': score_v2,
        'improved': improved, 'worsened': worsened
    }
```

## Section 6 : Error Analysis Approfondie

```python
def error_analysis(y_true, y_pred, train_df, features, task='classification',
                   threshold=0.5, n_worst=20):
    """Analyse détaillée des erreurs du modèle."""

    if task == 'classification':
        y_class = (y_pred > threshold).astype(int)
        errors = y_class != y_true

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_class)
        print("Confusion Matrix:")
        print(cm)
        print(f"\nFalse Positives: {cm[0,1]} ({cm[0,1]/cm[0].sum()*100:.1f}%)")
        print(f"False Negatives: {cm[1,0]} ({cm[1,0]/cm[1].sum()*100:.1f}%)")

        # Classification report
        print(f"\n{classification_report(y_true, y_class)}")

        # Analyse des erreurs par confidence
        error_confidence = y_pred[errors]
        print(f"\nError confidence distribution:")
        print(f"  Mean: {error_confidence.mean():.4f}")
        print(f"  Very confident errors (>0.9 or <0.1): "
              f"{((error_confidence > 0.9) | (error_confidence < 0.1)).sum()}")

    elif task == 'regression':
        residuals = y_true - y_pred

        print(f"Residual Analysis:")
        print(f"  Mean residual: {residuals.mean():.5f}")
        print(f"  Std residual: {residuals.std():.5f}")
        print(f"  Skewness: {residuals.skew():.3f}")

    # Worst predictions
    if task == 'classification':
        # Les plus confiants mais faux
        train_df = train_df.copy()
        train_df['pred'] = y_pred
        train_df['error'] = errors.astype(int)
        train_df['confidence_error'] = np.where(
            y_true == 1, 1 - y_pred, y_pred
        ) * errors

        worst = train_df.nlargest(n_worst, 'confidence_error')
        print(f"\n--- Top {n_worst} Worst Predictions ---")
        print(worst[features[:5] + ['pred', 'confidence_error']].to_string())
    else:
        abs_errors = np.abs(y_true - y_pred)
        worst_idx = abs_errors.nlargest(n_worst).index
        worst = train_df.loc[worst_idx].copy()
        worst['pred'] = y_pred[worst_idx]
        worst['abs_error'] = abs_errors[worst_idx]
        print(f"\n--- Top {n_worst} Worst Predictions ---")
        print(worst[features[:5] + ['pred', 'abs_error']].to_string())

    # Error patterns par feature
    print(f"\n--- Error Patterns by Feature ---")
    train_df['is_error'] = errors if task == 'classification' else (abs_errors > abs_errors.quantile(0.9))
    for feat in features[:10]:
        if train_df[feat].dtype in ['float64', 'int64']:
            correct_mean = train_df[~train_df['is_error']][feat].mean()
            error_mean = train_df[train_df['is_error']][feat].mean()
            diff = abs(error_mean - correct_mean)
            if diff > 0:
                print(f"  {feat}: correct_mean={correct_mean:.3f}, "
                      f"error_mean={error_mean:.3f}, diff={diff:.3f}")

    return train_df

def learning_curve_analysis(model_class, X, y, train_sizes=None, cv=5):
    """Learning curve pour diagnostiquer bias vs variance."""
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model_class, X, y, train_sizes=train_sizes,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-', label='Train')
    ax.plot(train_sizes_abs, val_scores.mean(axis=1), 'o-', label='Validation')
    ax.fill_between(train_sizes_abs,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    ax.fill_between(train_sizes_abs,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend()
    plt.show()

    # Diagnostic
    final_train = train_scores.mean(axis=1)[-1]
    final_val = val_scores.mean(axis=1)[-1]
    gap = final_train - final_val

    if gap > 0.05 and final_val < 0.85:
        print("→ HIGH VARIANCE (overfitting): more data or regularization needed")
    elif gap < 0.02 and final_val < 0.80:
        print("→ HIGH BIAS (underfitting): more complex model or features needed")
    elif final_val > 0.85:
        print("→ GOOD FIT: model is performing well")

    return train_sizes_abs, train_scores, val_scores
```

## Checklist de Debugging Kaggle

1. **Score instable entre folds ?** → Vérifier la stratégie de split
2. **CV >> LB ?** → Chercher du leakage (target encoding, features temporelles, preprocessing)
3. **LB >> CV ?** → CV trop conservateur ou test plus facile
4. **Train >> Val ?** → Overfitting → régulariser, réduire features
5. **Train ≈ Val, tous bas ?** → Underfitting → complexifier, ajouter features
6. **Score a baissé vs version précédente ?** → compare_versions() pour isoler le changement
7. **Erreurs concentrées sur un sous-groupe ?** → error_analysis() par segment
8. **Feature importance incohérente ?** → Possible multicolinéarité ou leakage
9. **Learning curve plate ?** → Plus de données n'aidera pas, changer l'approche
10. **Tout semble OK mais score bas ?** → Revoir l'EDA, il manque probablement un signal

## Rapport de Sortie (OBLIGATOIRE)

À la fin du diagnostic, TOUJOURS sauvegarder :
1. Rapport dans : `reports/debug/YYYY-MM-DD_<probleme>.md` (diagnostic complet, patch plan, vérifications)
2. Confirmer à l'utilisateur le chemin du rapport sauvegardé
