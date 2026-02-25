---
name: kaggle-experiments
description: Tracking d'expériences et gestion de runs pour compétitions Kaggle. Utiliser quand l'utilisateur veut tracker ses expériences, comparer des runs, faire des ablations, ou gérer un historique CV/LB structuré.
argument-hint: <action (init/log/compare/ablation) ou description>
---

# Experiment Tracking Expert - Kaggle Gold Medal

Tu es un expert en tracking d'expériences pour compétitions Kaggle. Sans tracking rigoureux, un agent "oublie" et tourne en rond. Ton rôle : maintenir un historique structuré de chaque run pour que chaque décision soit data-driven.

## Philosophie

- **Ce qui n'est pas tracké n'existe pas** : chaque run doit être documenté
- **Comparaison > score absolu** : c'est le delta qui compte, pas le chiffre
- **Ablation systématique** : retirer une feature/technique pour mesurer sa vraie contribution
- **Reproductibilité** : seed, params, features, preprocessing — tout doit être reconstituable

## Experiment Tracker

```python
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class ExperimentTracker:
    """Tracker léger d'expériences pour compétitions Kaggle.
    Pas besoin de MLflow/W&B — un CSV + JSON suffit.
    """

    def __init__(self, competition_name, save_dir='experiments'):
        self.competition = competition_name
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, 'experiment_log.csv')
        self.detail_dir = os.path.join(save_dir, 'details')
        os.makedirs(self.detail_dir, exist_ok=True)

        # Charger l'historique existant
        if os.path.exists(self.log_file):
            self.log = pd.read_csv(self.log_file)
        else:
            self.log = pd.DataFrame(columns=[
                'run_id', 'name', 'timestamp', 'cv_mean', 'cv_std',
                'cv_folds', 'lb_score', 'cv_lb_gap', 'n_features',
                'model', 'notes', 'status'
            ])

    def log_run(self, name, cv_scores, lb_score=None, features=None,
                params=None, model_type='lgb', notes='', oof_preds=None,
                test_preds=None):
        """Log un run complet."""
        run_id = f"run_{len(self.log):03d}"
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_lb_gap = abs(cv_mean - lb_score) if lb_score else None

        # Log principal
        entry = {
            'run_id': run_id,
            'name': name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'cv_mean': round(cv_mean, 6),
            'cv_std': round(cv_std, 6),
            'cv_folds': str([round(s, 5) for s in cv_scores]),
            'lb_score': lb_score,
            'cv_lb_gap': round(cv_lb_gap, 6) if cv_lb_gap else None,
            'n_features': len(features) if features else None,
            'model': model_type,
            'notes': notes,
            'status': 'active'
        }
        self.log = pd.concat([self.log, pd.DataFrame([entry])], ignore_index=True)
        self.log.to_csv(self.log_file, index=False)

        # Détails du run (params, features, etc.)
        detail = {
            'run_id': run_id,
            'name': name,
            'cv_scores': [float(s) for s in cv_scores],
            'lb_score': lb_score,
            'features': features,
            'params': params,
            'model_type': model_type,
            'notes': notes,
        }
        detail_file = os.path.join(self.detail_dir, f'{run_id}.json')
        with open(detail_file, 'w') as f:
            json.dump(detail, f, indent=2, default=str)

        # Sauver les prédictions OOF et test
        if oof_preds is not None:
            np.save(os.path.join(self.detail_dir, f'{run_id}_oof.npy'), oof_preds)
        if test_preds is not None:
            np.save(os.path.join(self.detail_dir, f'{run_id}_test.npy'), test_preds)

        print(f"✓ Logged: {run_id} | {name} | CV={cv_mean:.5f}±{cv_std:.5f} | LB={lb_score}")

        # Comparer avec le run précédent
        if len(self.log) > 1:
            prev = self.log.iloc[-2]
            delta_cv = cv_mean - prev['cv_mean']
            print(f"  Delta vs {prev['name']}: {delta_cv:+.5f}")

        return run_id

    def report(self, top_n=None):
        """Affiche le tableau récapitulatif de toutes les expériences."""
        df = self.log[self.log['status'] == 'active'].copy()
        if top_n:
            df = df.nlargest(top_n, 'cv_mean')

        print(f"\n{'='*80}")
        print(f"EXPERIMENT LOG — {self.competition}")
        print(f"{'='*80}")
        display_cols = ['run_id', 'name', 'cv_mean', 'cv_std', 'lb_score',
                       'cv_lb_gap', 'n_features', 'model']
        print(df[display_cols].to_string(index=False))

        if len(df) >= 2:
            best_cv = df.loc[df['cv_mean'].idxmax()]
            best_lb = df.loc[df['lb_score'].idxmax()] if df['lb_score'].notna().any() else None

            print(f"\nBest CV: {best_cv['name']} ({best_cv['cv_mean']:.5f})")
            if best_lb is not None:
                print(f"Best LB: {best_lb['name']} ({best_lb['lb_score']:.5f})")

            if df['lb_score'].notna().sum() >= 3:
                corr = df.dropna(subset=['lb_score'])[['cv_mean', 'lb_score']].corr().iloc[0, 1]
                print(f"CV-LB Correlation: {corr:.4f}")

        return df

    def compare(self, run_id_1, run_id_2):
        """Compare deux runs en détail."""
        d1_file = os.path.join(self.detail_dir, f'{run_id_1}.json')
        d2_file = os.path.join(self.detail_dir, f'{run_id_2}.json')

        with open(d1_file) as f:
            d1 = json.load(f)
        with open(d2_file) as f:
            d2 = json.load(f)

        print(f"\n{'='*60}")
        print(f"COMPARISON: {run_id_1} vs {run_id_2}")
        print(f"{'='*60}")

        print(f"\n  {d1['name']:>30} | {d2['name']}")
        print(f"  {'CV Mean':>30}: {np.mean(d1['cv_scores']):.5f} | {np.mean(d2['cv_scores']):.5f}")
        print(f"  {'CV Std':>30}: {np.std(d1['cv_scores']):.5f} | {np.std(d2['cv_scores']):.5f}")
        if d1['lb_score'] and d2['lb_score']:
            print(f"  {'LB Score':>30}: {d1['lb_score']:.5f} | {d2['lb_score']:.5f}")

        # Feature diff
        if d1.get('features') and d2.get('features'):
            added = set(d2['features']) - set(d1['features'])
            removed = set(d1['features']) - set(d2['features'])
            if added:
                print(f"\n  Features ADDED: {added}")
            if removed:
                print(f"  Features REMOVED: {removed}")
            print(f"  Features: {len(d1['features'])} → {len(d2['features'])}")

        # Params diff
        if d1.get('params') and d2.get('params'):
            changed = {}
            for key in set(list(d1['params'].keys()) + list(d2['params'].keys())):
                v1 = d1['params'].get(key)
                v2 = d2['params'].get(key)
                if v1 != v2:
                    changed[key] = (v1, v2)
            if changed:
                print(f"\n  Params CHANGED:")
                for k, (v1, v2) in changed.items():
                    print(f"    {k}: {v1} → {v2}")

    def get_oof(self, run_id):
        """Charge les OOF predictions d'un run."""
        return np.load(os.path.join(self.detail_dir, f'{run_id}_oof.npy'))

    def get_test(self, run_id):
        """Charge les test predictions d'un run."""
        return np.load(os.path.join(self.detail_dir, f'{run_id}_test.npy'))
```

## Ablation Study

```python
def ablation_study(tracker, train, test, target, features, base_features,
                   train_fn, feature_groups):
    """Ablation study : mesurer la contribution de chaque groupe de features.

    Args:
        feature_groups: dict {'group_name': [list of features]}
        train_fn: function(features) → (cv_scores, oof_preds, test_preds)
    """
    print(f"{'='*60}")
    print(f"ABLATION STUDY")
    print(f"{'='*60}")

    # Baseline avec toutes les features
    all_features = list(set(feat for group in feature_groups.values() for feat in group))
    all_features = [f for f in all_features if f in base_features] + \
                   [f for f in all_features if f not in base_features]

    cv_all, _, _ = train_fn(base_features + all_features)
    print(f"\nAll features ({len(base_features + all_features)}): "
          f"CV={np.mean(cv_all):.5f}")

    results = [{'group': 'ALL', 'n_features': len(base_features + all_features),
                'cv_mean': np.mean(cv_all), 'delta': 0}]

    # Retirer chaque groupe un par un
    for group_name, group_features in feature_groups.items():
        remaining = [f for f in base_features + all_features if f not in group_features]
        cv_without, _, _ = train_fn(remaining)
        delta = np.mean(cv_all) - np.mean(cv_without)

        results.append({
            'group': f'- {group_name}',
            'n_features': len(remaining),
            'cv_mean': np.mean(cv_without),
            'delta': delta
        })
        print(f"Without {group_name} ({len(group_features)} feats): "
              f"CV={np.mean(cv_without):.5f} | delta={delta:+.5f}")

    # Ajouter chaque groupe un par un (à partir du base)
    print(f"\n--- Additive (from base) ---")
    cv_base, _, _ = train_fn(base_features)
    print(f"Base only ({len(base_features)}): CV={np.mean(cv_base):.5f}")

    for group_name, group_features in feature_groups.items():
        with_group = base_features + [f for f in group_features if f not in base_features]
        cv_with, _, _ = train_fn(with_group)
        delta = np.mean(cv_with) - np.mean(cv_base)
        print(f"+ {group_name} ({len(group_features)} feats): "
              f"CV={np.mean(cv_with):.5f} | delta={delta:+.5f}")

    # Summary
    df = pd.DataFrame(results).sort_values('delta', ascending=False)
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY (negative delta = group helps)")
    print(df.to_string(index=False))

    return df
```

## Seed Study

```python
def seed_stability_study(train_fn, seeds=[42, 123, 456, 789, 2024, 2025, 3407, 9999]):
    """Mesure la stabilité du score à travers différents seeds."""
    scores = []

    for seed in seeds:
        cv, _, _ = train_fn(seed=seed)
        mean_cv = np.mean(cv)
        scores.append({'seed': seed, 'cv_mean': mean_cv, 'cv_std': np.std(cv)})
        print(f"Seed {seed}: CV={mean_cv:.5f}")

    df = pd.DataFrame(scores)
    print(f"\nSeed Stability:")
    print(f"  Mean of means: {df['cv_mean'].mean():.5f}")
    print(f"  Std of means: {df['cv_mean'].std():.5f}")
    print(f"  Range: {df['cv_mean'].max() - df['cv_mean'].min():.5f}")

    if df['cv_mean'].std() > 0.005:
        print("⚠ HIGH seed variance — consider multi-seed averaging")
    else:
        print("✓ Stable across seeds")

    return df
```

## Config Management

```python
import yaml

def save_config(config, path='config.yaml'):
    """Sauvegarde la config d'un run."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(path='config.yaml'):
    """Charge une config."""
    with open(path) as f:
        return yaml.safe_load(f)

# Exemple de config standard
TEMPLATE_CONFIG = {
    'competition': 'competition-name',
    'seed': 42,
    'n_folds': 5,
    'target': 'target',
    'id_col': 'id',
    'metric': 'auc',

    'features': {
        'numeric': [],
        'categorical': [],
        'engineered': [],
        'dropped': [],
    },

    'preprocessing': {
        'missing_strategy': 'median',
        'scaling': None,
        'encoding': 'label',
    },

    'model': {
        'type': 'lightgbm',
        'params': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 10000,
            'early_stopping_rounds': 100,
        }
    },

    'ensemble': {
        'models': [],
        'weights': [],
        'method': 'rank_average',
    }
}
```

## Usage Intégré au Workflow

```python
# === INIT ===
tracker = ExperimentTracker('my-competition')

# === BASELINE ===
cv_scores = [0.812, 0.815, 0.809, 0.818, 0.811]
tracker.log_run(
    name='baseline_lgb',
    cv_scores=cv_scores,
    lb_score=0.808,
    features=features,
    params=lgb_params,
    model_type='lgb',
    notes='Default params, minimal preprocessing',
    oof_preds=oof_preds,
    test_preds=test_preds
)

# === ITERATION ===
tracker.log_run(
    name='v2_features',
    cv_scores=[0.821, 0.825, 0.819, 0.828, 0.822],
    lb_score=0.815,
    features=features_v2,
    params=lgb_params,
    model_type='lgb',
    notes='Added interaction features + freq encoding',
    oof_preds=oof_v2,
    test_preds=test_v2
)

# === COMPARE ===
tracker.compare('run_000', 'run_001')

# === REPORT ===
tracker.report()

# === ENSEMBLE ===
oof_dict = {
    'lgb_v2': tracker.get_oof('run_001'),
    'xgb_v1': tracker.get_oof('run_002'),
}
```

## Definition of Done (DoD)

Chaque run est COMPLET quand :

- [ ] Run loggé avec nom descriptif, CV scores par fold, et notes
- [ ] Params et features sauvegardés (pour reproduction)
- [ ] OOF predictions sauvegardées (pour l'ensembling futur)
- [ ] Test predictions sauvegardées (pour la soumission)
- [ ] Delta vs run précédent calculé et documenté
- [ ] LB score noté après soumission
- [ ] Config reproductible (seed, preprocessing, model)

## Règles d'Or

1. **Logger CHAQUE run** : même les échecs (ils informent)
2. **Nommer clairement** : `v3_freq_encoding` pas `test_v3`
3. **Sauver les OOF** : ils sont nécessaires pour l'ensembling
4. **Ablation avant d'ajouter** : vérifier que chaque feature apporte vraiment
5. **Seeds multiples** : tester la stabilité avant de conclure
6. **Comparer avec le bon baseline** : toujours vs le run précédent ET vs le baseline original

## Rapport de Sortie (OBLIGATOIRE)

À la fin de chaque run, TOUJOURS sauvegarder :
1. Ligne ajoutée dans : `runs.csv` (run_id, date, description, cv_score, cv_std, lb_score, n_features, model_type, params_hash, notes)
2. OOF predictions dans : `artifacts/oof_<model>_v<N>.npy`
3. Test predictions dans : `artifacts/test_<model>_v<N>.npy`
4. Config dans : `configs/<experiment>.yaml`
5. Si rapport demandé : `reports/experiments/YYYY-MM-DD_summary.md` (tableau récap de tous les runs)
6. Confirmer à l'utilisateur les chemins sauvegardés
