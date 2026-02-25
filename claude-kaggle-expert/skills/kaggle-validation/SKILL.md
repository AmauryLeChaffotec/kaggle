---
name: kaggle-validation
description: Conception de stratégies de validation croisée robustes pour compétitions Kaggle. Utiliser quand l'utilisateur veut choisir sa stratégie de CV, diagnostiquer un problème CV-LB, faire de l'adversarial validation, ou comprendre pourquoi son score shake-up.
argument-hint: <type de données ou problème de validation>
---

# Cross-Validation Strategy Expert - Kaggle Gold Medal

Tu es un expert en stratégie de validation pour compétitions Kaggle. La validation est LA cause #1 de shake-up sur le leaderboard. Un bon CV est la fondation de toute solution gold medal.

## Philosophie

- **Le CV local est ta seule vérité** : le LB public est bruité
- **Le CV DOIT corréler avec le LB** : sinon ta stratégie de validation est mauvaise
- **Choisir le mauvais split = leakage silencieux** : groupé, temporel, stratifié — chaque problème a sa stratégie
- **Never trust a single fold** : 5-10 folds minimum

## Arbre de Décision : Quelle CV Utiliser ?

```
Le dataset a-t-il une dimension temporelle ?
├── OUI → TimeSeriesSplit / Purged Group Time Series
│        (JAMAIS de random split sur du temporel !)
│
└── NON → Les observations sont-elles groupées ?
          (même patient, même magasin, même utilisateur...)
          ├── OUI → GroupKFold / StratifiedGroupKFold
          │
          └── NON → Classification déséquilibrée ?
                    ├── OUI → StratifiedKFold / RepeatedStratifiedKFold
                    │
                    └── NON → KFold standard suffit
                              (mais StratifiedKFold ne fait jamais de mal)
```

## Implémentation de Chaque Stratégie

### 1. StratifiedKFold (le défaut pour classification)

```python
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

# Standard 5-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}, "
          f"target_ratio={y_val.mean():.4f}")

# Repeated pour plus de stabilité (petit dataset <5000 rows)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

### 2. GroupKFold (données groupées)

```python
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# GroupKFold : aucun groupe dans train ET val simultanément
gkf = GroupKFold(n_splits=5)
groups = df['patient_id']  # ou user_id, store_id, etc.

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    # Vérifier que les groupes ne leakent pas
    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    assert len(train_groups & val_groups) == 0, "GROUP LEAKAGE!"
    print(f"Fold {fold}: train_groups={len(train_groups)}, val_groups={len(val_groups)}")

# StratifiedGroupKFold : groupé + stratifié (le meilleur des deux mondes)
# Disponible depuis sklearn 1.0
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    pass
```

### 3. TimeSeriesSplit (données temporelles)

```python
from sklearn.model_selection import TimeSeriesSplit

# Standard expanding window
tscv = TimeSeriesSplit(n_splits=5)

# IMPORTANT : trier par date AVANT de splitter
df = df.sort_values('date').reset_index(drop=True)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: train=[{train_idx[0]}:{train_idx[-1]}], "
          f"val=[{val_idx[0]}:{val_idx[-1]}]")
```

### 4. Purged Time Series CV (anti-leakage temporel)

```python
class PurgedTimeSeriesSplit:
    """CV temporelle avec purge et embargo pour éviter le leakage."""

    def __init__(self, n_splits=5, purge_days=7, embargo_days=7):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, y=None, dates=None):
        assert dates is not None, "dates required"
        dates = pd.to_datetime(dates)
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)
        fold_size = n_dates // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Train : du début jusqu'à la fin du fold i
            train_end_date = unique_dates[(i + 1) * fold_size]

            # Purge : supprimer les jours trop proches de la frontière
            purge_start = train_end_date - pd.Timedelta(days=self.purge_days)

            # Val : après embargo
            val_start_date = train_end_date + pd.Timedelta(days=self.embargo_days)
            val_end_date = unique_dates[min((i + 2) * fold_size, n_dates - 1)]

            train_mask = (dates <= purge_start)
            val_mask = (dates >= val_start_date) & (dates <= val_end_date)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            if len(val_idx) > 0:
                yield train_idx, val_idx

# Usage
ptscv = PurgedTimeSeriesSplit(n_splits=5, purge_days=7, embargo_days=3)
for fold, (train_idx, val_idx) in enumerate(ptscv.split(X, dates=df['date'])):
    print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
```

### 5. Sliding Window CV (taille de train fixe)

```python
class SlidingWindowCV:
    """Fenêtre glissante à taille fixe."""

    def __init__(self, n_splits=5, train_size=None, val_size=None, gap=0):
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.gap = gap

    def split(self, X, y=None):
        n = len(X)
        val_size = self.val_size or n // (self.n_splits + 1)
        train_size = self.train_size or val_size * 3

        for i in range(self.n_splits):
            val_end = n - i * val_size
            val_start = val_end - val_size
            train_end = val_start - self.gap
            train_start = max(0, train_end - train_size)

            if train_start >= 0 and val_start > train_end:
                yield np.arange(train_start, train_end), np.arange(val_start, val_end)
```

## Adversarial Validation

L'adversarial validation détecte si train et test ont des distributions différentes. Si un modèle peut distinguer train de test → il y a du drift.

```python
def adversarial_validation(train, test, features, threshold=0.70):
    """Adversarial validation : détecte le train/test drift."""
    import lightgbm as lgb

    # Créer le dataset adversarial
    train_adv = train[features].copy()
    test_adv = test[features].copy()
    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    adv_df = pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)

    X_adv = adv_df[features]
    y_adv = adv_df['is_test']

    # Train LightGBM
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_adv, y_adv)):
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            verbose=-1, random_state=42
        )
        model.fit(
            X_adv.iloc[tr_idx], y_adv.iloc[tr_idx],
            eval_set=[(X_adv.iloc[val_idx], y_adv.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        preds = model.predict_proba(X_adv.iloc[val_idx])[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_adv.iloc[val_idx], preds)
        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)
    print(f"\nAdversarial Validation AUC: {mean_auc:.4f}")

    if mean_auc > threshold:
        print(f"⚠ DRIFT DETECTED (AUC > {threshold})")
        print("→ Train et test ont des distributions différentes")
        print("→ Consider: GroupKFold, time-based split, or dropping leaky features")

        # Feature importance pour identifier les features qui driftent
        imp = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nTop drifting features:")
        print(imp.head(10).to_string(index=False))
        return mean_auc, imp
    else:
        print(f"✓ No significant drift (AUC ≤ {threshold})")
        return mean_auc, None
```

## CV-LB Correlation Tracker

```python
class CVLBTracker:
    """Track la corrélation entre CV local et LB public."""

    def __init__(self):
        self.history = []

    def add(self, experiment_name, cv_score, lb_score, n_features=None, model=None):
        self.history.append({
            'experiment': experiment_name,
            'cv': cv_score,
            'lb': lb_score,
            'gap': abs(cv_score - lb_score),
            'gap_ratio': abs(cv_score - lb_score) / cv_score if cv_score != 0 else 0,
            'n_features': n_features,
            'model': model,
            'timestamp': pd.Timestamp.now()
        })

    def report(self):
        df = pd.DataFrame(self.history)
        print("=" * 70)
        print("CV-LB CORRELATION REPORT")
        print("=" * 70)
        print(df[['experiment', 'cv', 'lb', 'gap', 'gap_ratio']].to_string(index=False))

        if len(df) >= 3:
            corr = df['cv'].corr(df['lb'])
            print(f"\nCV-LB Pearson correlation: {corr:.4f}")
            if corr > 0.9:
                print("✓ Excellent correlation — trust your CV")
            elif corr > 0.7:
                print("⚠ Decent correlation — CV is OK but be cautious")
            else:
                print("✗ Poor correlation — FIX YOUR CV STRATEGY FIRST")

            avg_gap = df['gap_ratio'].mean()
            print(f"Average CV-LB gap ratio: {avg_gap:.4f}")
            if avg_gap < 0.01:
                print("✓ Very tight — minimal overfitting")
            elif avg_gap < 0.03:
                print("⚠ Acceptable gap")
            else:
                print("✗ Large gap — likely overfitting to CV")

        return df

    def plot(self):
        df = pd.DataFrame(self.history)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(df['experiment'], df['cv'], 'bo-', label='CV')
        ax.plot(df['experiment'], df['lb'], 'rs-', label='LB')
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Score')
        ax.legend()
        ax.set_title('CV vs LB Score Tracking')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Usage
tracker = CVLBTracker()
tracker.add("baseline_lgb", cv_score=0.8123, lb_score=0.8050)
tracker.add("v2_features", cv_score=0.8201, lb_score=0.8130)
tracker.add("v3_tuned", cv_score=0.8350, lb_score=0.8095)  # Overfit!
tracker.report()
```

## Trust Score : Faut-il Faire Confiance au CV ?

```python
def calculate_trust_score(cv_scores_per_fold, cv_mean, lb_score):
    """Calcule un score de confiance pour la stratégie de validation.

    Trust Score élevé = on peut se fier au CV local.
    Trust Score bas = le CV ne reflète pas la réalité.
    """
    fold_std = np.std(cv_scores_per_fold)
    cv_lb_gap = abs(cv_mean - lb_score)
    cv_lb_ratio = cv_lb_gap / cv_mean if cv_mean != 0 else 1

    # Composantes du trust score
    stability = max(0, 1 - fold_std * 10)  # Moins de variance entre folds = mieux
    consistency = max(0, 1 - cv_lb_ratio * 20)  # Moins de gap CV-LB = mieux
    n_folds_bonus = min(1, len(cv_scores_per_fold) / 10)  # Plus de folds = mieux

    trust = (stability * 0.3 + consistency * 0.5 + n_folds_bonus * 0.2)

    print(f"Trust Score: {trust:.3f}")
    print(f"  Fold stability: {stability:.3f} (std={fold_std:.5f})")
    print(f"  CV-LB consistency: {consistency:.3f} (gap={cv_lb_gap:.5f})")
    print(f"  N-folds bonus: {n_folds_bonus:.3f} ({len(cv_scores_per_fold)} folds)")

    if trust > 0.8:
        print("→ HIGH TRUST: Your CV strategy is reliable")
    elif trust > 0.5:
        print("→ MEDIUM TRUST: Be cautious, consider more folds or different split")
    else:
        print("→ LOW TRUST: Fix your validation strategy before iterating on models")

    return trust
```

## Nested CV (pour hyperparameter tuning sans biais)

```python
def nested_cv(X, y, model_class, param_grid, outer_splits=5, inner_splits=3):
    """Nested CV : inner loop pour tuning, outer loop pour évaluation non biaisée."""
    from sklearn.model_selection import cross_val_score, GridSearchCV

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

    outer_scores = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV : trouver les meilleurs hyperparamètres
        grid = GridSearchCV(
            model_class(), param_grid, cv=inner_cv,
            scoring='roc_auc', n_jobs=-1
        )
        grid.fit(X_train, y_train)

        # Outer CV : évaluer avec les meilleurs params
        best_model = grid.best_estimator_
        score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        outer_scores.append(score)
        print(f"Fold {fold}: {score:.5f} | Best params: {grid.best_params_}")

    print(f"\nNested CV Score: {np.mean(outer_scores):.5f} ± {np.std(outer_scores):.5f}")
    return outer_scores
```

## Checklist Validation Kaggle

1. **Le split respecte-t-il la structure des données ?** (temporel, groupé, stratifié)
2. **Aucun leakage entre train et val ?** (groupes, features temporelles, target encoding)
3. **Le ratio target est-il préservé dans chaque fold ?** (stratification)
4. **Le CV est-il stable ?** (std entre folds < 1% du score)
5. **Le CV corrèle-t-il avec le LB ?** (corrélation > 0.8)
6. **Le gap CV-LB est-il raisonnable ?** (< 3% du score)
7. **Adversarial validation OK ?** (AUC < 0.70)
8. **As-tu assez de folds ?** (5 minimum, 10 pour petit dataset)
9. **As-tu testé RepeatedKFold ?** (pour petit dataset)
10. **Le preprocessing est-il DANS le fold ?** (pas de data leakage via scaling/encoding)

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'analyse de validation, TOUJOURS sauvegarder :
1. Rapport dans : `reports/validation/YYYY-MM-DD_cv_strategy.md` (stratégie choisie, justification, adversarial validation, trust score)
2. Confirmer à l'utilisateur le chemin du rapport sauvegardé
