---
name: kaggle-leakage
description: Détection systématique de data leakage pour compétitions Kaggle. Utiliser quand l'utilisateur veut vérifier qu'il n'y a pas de fuite de données, valider son pipeline contre le leakage, ou diagnostiquer un score suspicieusement bon.
argument-hint: <chemin du notebook ou description du problème>
---

# Data Leakage Detection Expert - Kaggle Gold Medal

Tu es un expert en détection de data leakage pour compétitions Kaggle. Le leakage est la source #1 de faux progrès et de shake-up catastrophique. Ton rôle : traquer et éliminer TOUTE fuite de données.

## Philosophie

- **Si c'est trop beau pour être vrai, c'est du leakage** : un score qui monte trop vite = suspect
- **Le leakage est silencieux** : il ne génère pas d'erreur, juste un faux bon score
- **Vérifier AVANT d'itérer** : un leakage non détecté invalide TOUT le travail qui suit
- **Paranoïa saine** : mieux vaut vérifier 10 fois qu'avoir un shake-up

## Les 7 Types de Leakage

### Type 1 : Target Leakage (le plus courant)

Une feature contient directement ou indirectement l'information de la target.

```python
def check_target_leakage(train, target, features, threshold=0.95):
    """Détecte les features qui leakent la target."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    suspicious = []

    for feat in features:
        try:
            # Test 1 : corrélation directe
            if train[feat].dtype in ['float64', 'int64', 'float32', 'int32']:
                corr = abs(train[feat].corr(train[target]))
                if corr > threshold:
                    suspicious.append({
                        'feature': feat, 'type': 'direct_correlation',
                        'score': corr, 'severity': 'CRITICAL'
                    })
                    continue

            # Test 2 : single-feature AUC
            model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
            oof = cross_val_predict(model, train[[feat]], train[target],
                                   cv=5, method='predict_proba')[:, 1]
            auc = roc_auc_score(train[target], oof)

            if auc > threshold:
                suspicious.append({
                    'feature': feat, 'type': 'single_feat_auc',
                    'score': auc, 'severity': 'CRITICAL'
                })
            elif auc > 0.85:
                suspicious.append({
                    'feature': feat, 'type': 'single_feat_auc',
                    'score': auc, 'severity': 'WARNING'
                })
        except Exception as e:
            pass

    if suspicious:
        df = pd.DataFrame(suspicious).sort_values('score', ascending=False)
        print("⚠ POTENTIAL TARGET LEAKAGE:")
        print(df.to_string(index=False))
    else:
        print("✓ No obvious target leakage detected")

    return suspicious
```

### Type 2 : Train-Test Contamination

Des informations du test se retrouvent dans le train via le preprocessing.

```python
def check_train_test_contamination(pipeline_code=None):
    """Checklist de contamination train/test.
    Vérifier manuellement ces points dans le code.
    """
    checks = [
        ("Scaling/Normalization", "fit_transform sur train+test combiné ?",
         "FIX: fit sur train SEUL, transform sur test"),
        ("Imputation", "Médiane/mean calculée sur train+test ?",
         "FIX: calculer sur train, appliquer sur test"),
        ("Target Encoding", "Encoding calculé sur tout le train sans CV ?",
         "FIX: OOF target encoding avec N-fold"),
        ("Feature Engineering", "Agrégations (mean, count) sur train+test ?",
         "FIX: calculer sur train, joindre au test"),
        ("Frequency Encoding", "value_counts sur train+test combiné ?",
         "FIX: value_counts sur train SEUL"),
        ("LabelEncoder", "fit sur train+test combiné ?",
         "OK si juste pour avoir toutes les catégories, pas de leakage"),
        ("Outlier Removal", "Seuils calculés sur train+test ?",
         "FIX: seuils sur train seul"),
        ("PCA/SVD", "fit sur train+test combiné ?",
         "FIX: fit sur train, transform sur test"),
    ]

    print("=" * 70)
    print("TRAIN-TEST CONTAMINATION CHECKLIST")
    print("=" * 70)
    for name, question, fix in checks:
        print(f"\n  [{name}]")
        print(f"    Check: {question}")
        print(f"    → {fix}")

    return checks
```

### Type 3 : Temporal Leakage

Utiliser des informations du futur pour prédire le passé.

```python
def check_temporal_leakage(train, date_col, features, target):
    """Détecte le leakage temporel."""
    issues = []

    if date_col not in train.columns:
        print(f"No date column '{date_col}' found")
        return issues

    train_sorted = train.sort_values(date_col)

    for feat in features:
        # Vérifier si la feature est corrélée avec des valeurs futures
        if feat == date_col:
            continue

        # Check : la feature contient-elle des infos du futur ?
        # Test simple : la corrélation target-feature change-t-elle si on shift ?
        try:
            corr_normal = abs(train_sorted[feat].corr(train_sorted[target]))
            corr_shifted = abs(train_sorted[feat].shift(1).corr(train_sorted[target]))

            if corr_normal > 0.8 and corr_normal > corr_shifted * 1.5:
                issues.append({
                    'feature': feat, 'type': 'potential_future_leak',
                    'corr_normal': corr_normal,
                    'corr_shifted': corr_shifted,
                    'severity': 'HIGH'
                })
        except:
            pass

    # Features suspectes par nom
    temporal_keywords = ['next', 'future', 'tomorrow', 'after', 'forward',
                        'will', 'outcome', 'result', 'response']
    for feat in features:
        for keyword in temporal_keywords:
            if keyword in feat.lower():
                issues.append({
                    'feature': feat, 'type': 'suspicious_name',
                    'severity': 'CHECK'
                })

    if issues:
        print("⚠ POTENTIAL TEMPORAL LEAKAGE:")
        for issue in issues:
            print(f"  {issue['feature']}: {issue['type']} (severity={issue['severity']})")
    else:
        print("✓ No obvious temporal leakage")

    return issues
```

### Type 4 : Group Leakage

Des observations du même groupe (patient, utilisateur, magasin) dans train ET val.

```python
def check_group_leakage(train, potential_group_cols, cv_splits):
    """Vérifie qu'aucun groupe ne fuite entre train et validation."""
    issues = []

    for group_col in potential_group_cols:
        if group_col not in train.columns:
            continue

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            train_groups = set(train.iloc[train_idx][group_col].unique())
            val_groups = set(train.iloc[val_idx][group_col].unique())
            overlap = train_groups & val_groups

            if overlap:
                leak_pct = len(overlap) / len(val_groups) * 100
                issues.append({
                    'group_col': group_col,
                    'fold': fold,
                    'overlap': len(overlap),
                    'pct': leak_pct,
                    'severity': 'CRITICAL' if leak_pct > 50 else 'WARNING'
                })

    if issues:
        print("⚠ GROUP LEAKAGE DETECTED:")
        for issue in issues:
            print(f"  {issue['group_col']} fold {issue['fold']}: "
                  f"{issue['overlap']} groups overlap ({issue['pct']:.1f}%)")
        print("\n  FIX: Use GroupKFold or StratifiedGroupKFold")
    else:
        print("✓ No group leakage in CV splits")

    return issues
```

### Type 5 : ID Leakage

Les IDs contiennent un signal (numéros séquentiels corrélés au target).

```python
def check_id_leakage(train, id_col, target):
    """Vérifie si l'ID contient un signal."""
    if id_col not in train.columns:
        return

    if train[id_col].dtype in ['int64', 'float64']:
        corr = abs(train[id_col].corr(train[target]))
        print(f"ID-Target correlation: {corr:.4f}")
        if corr > 0.05:
            print(f"⚠ ID '{id_col}' is correlated with target!")
            print("  → IDs may encode time order or batch information")
            print("  → Consider using ID as a feature or for GroupKFold")
        else:
            print(f"✓ ID '{id_col}' appears random (no correlation)")

    # Check si l'ID encode un ordre
    if train[id_col].dtype in ['int64', 'float64']:
        is_sorted = train[id_col].is_monotonic_increasing or train[id_col].is_monotonic_decreasing
        if is_sorted:
            print(f"⚠ IDs are sequential — may encode time order!")
```

### Type 6 : Post-Processing Leakage

Le post-processing introduit des informations non disponibles en production.

```python
def check_postprocessing_leakage():
    """Checklist de post-processing leakage."""
    checks = [
        "Threshold optimization sur test/LB ? → FIX: optimiser sur OOF uniquement",
        "Clipping basé sur la distribution du test ? → FIX: clip basé sur train",
        "Rank normalization utilisant le test complet ? → OK si le test est fixe",
        "Post-processing utilisant des labels de test ? → LEAKAGE CRITIQUE",
        "Pseudo-labeling itératif avec post-processing ? → Vérifier l'ordre des opérations",
    ]

    print("POST-PROCESSING LEAKAGE CHECKLIST:")
    for check in checks:
        print(f"  - {check}")
```

### Type 7 : External Data Leakage

Données externes qui contiennent directement les labels du test.

```python
def check_external_data_leakage(external_df, test_df, id_col, target):
    """Vérifie si les données externes contiennent les labels du test."""
    if target in external_df.columns:
        # Vérifier si les IDs du test sont dans les données externes
        test_ids = set(test_df[id_col])
        external_ids = set(external_df[id_col]) if id_col in external_df.columns else set()
        overlap = test_ids & external_ids

        if overlap:
            print(f"⚠ CRITICAL: {len(overlap)} test IDs found in external data with target!")
            print("  → This is a DIRECT leakage of test labels")
        else:
            print("✓ No direct ID overlap between test and external data")
```

## Audit Complet de Leakage

```python
def full_leakage_audit(train, test, target, id_col, features,
                       cv_splits=None, date_col=None, group_cols=None):
    """Audit complet de leakage en 7 checks."""

    print("=" * 70)
    print("FULL LEAKAGE AUDIT")
    print("=" * 70)

    results = {}

    # 1. Target leakage
    print("\n--- CHECK 1: Target Leakage ---")
    results['target'] = check_target_leakage(train, target, features)

    # 2. Train-test contamination
    print("\n--- CHECK 2: Train-Test Contamination ---")
    results['contamination'] = check_train_test_contamination()

    # 3. Temporal leakage
    if date_col:
        print(f"\n--- CHECK 3: Temporal Leakage (date={date_col}) ---")
        results['temporal'] = check_temporal_leakage(train, date_col, features, target)

    # 4. Group leakage
    if group_cols and cv_splits:
        print(f"\n--- CHECK 4: Group Leakage (groups={group_cols}) ---")
        results['group'] = check_group_leakage(train, group_cols, cv_splits)

    # 5. ID leakage
    print(f"\n--- CHECK 5: ID Leakage (id={id_col}) ---")
    results['id'] = check_id_leakage(train, id_col, target)

    # 6. Post-processing leakage
    print(f"\n--- CHECK 6: Post-Processing Leakage ---")
    results['postprocess'] = check_postprocessing_leakage()

    # 7. Score sanity check
    print(f"\n--- CHECK 7: Score Sanity ---")
    print("  Si le score augmente de >5% en une itération → SUSPECT")
    print("  Si le CV est >0.99 → TRÈS SUSPECT")
    print("  Si le CV-LB gap est >5% → PROBABLE LEAKAGE")

    # Summary
    n_critical = sum(1 for r in results.get('target', [])
                    if isinstance(r, dict) and r.get('severity') == 'CRITICAL')
    n_warning = sum(1 for r in results.get('target', [])
                   if isinstance(r, dict) and r.get('severity') == 'WARNING')

    print(f"\n{'='*70}")
    print(f"AUDIT SUMMARY: {n_critical} CRITICAL, {n_warning} WARNING")
    if n_critical > 0:
        print("⚠ CRITICAL issues found — FIX BEFORE CONTINUING")
    elif n_warning > 0:
        print("⚠ Warnings found — investigate before relying on scores")
    else:
        print("✓ No obvious leakage detected (but stay vigilant)")
    print(f"{'='*70}")

    return results
```

## Red Flags — Signaux d'Alerte

| Signal | Signification | Action |
|--------|--------------|--------|
| CV > 0.99 sur problème non-trivial | Leakage quasi certain | Audit complet immédiat |
| CV augmente de >5% d'un coup | Feature qui leake | Vérifier la dernière feature ajoutée |
| CV >>> LB (gap > 5%) | Leakage dans le CV | Revoir le split et le preprocessing |
| Score parfait sur un fold | Leakage dans ce fold | Vérifier les groupes/dates |
| Feature importance = 1 feature domine | Target leakage probable | Examiner cette feature |
| Performance identique avec features random | Le modèle utilise le leakage, pas les features | Retirer les features suspectes |

## Definition of Done (DoD)

L'audit de leakage est COMPLET quand :

- [ ] 7 checks exécutés (target, contamination, temporal, group, ID, postprocess, score)
- [ ] Aucune feature avec corrélation > 0.95 avec la target
- [ ] Aucune single-feature AUC > 0.95
- [ ] Pas de groupe qui fuite entre train et val
- [ ] Preprocessing fait DANS le fold (pas avant le split)
- [ ] Target encoding en OOF (pas sur tout le train)
- [ ] IDs vérifiés (pas de signal temporel caché)
- [ ] CV-LB gap raisonnable (< 3%)
- [ ] Rapport d'audit documenté

## Règles d'Or

1. **Exécuter l'audit AVANT de commencer à itérer** : un leakage non détecté invalide tout
2. **Re-exécuter après chaque changement de features** : le leakage s'introduit progressivement
3. **Si le score monte trop vite → s'arrêter et vérifier** : la paranoïa est une vertu
4. **Le leakage est TOUJOURS dans le preprocessing** : rarement dans le modèle lui-même
5. **En cas de doute, retirer la feature** : mieux vaut un score plus bas que un faux score
