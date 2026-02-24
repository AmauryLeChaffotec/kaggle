# Historique des codes et scores (March Madness 2026)

## Objectif

Ce fichier sert de registre des versions de scripts, des changements testes, des scores Kaggle obtenus (Stage 1), et des fichiers de soumission associes.

## Organisation actuelle

- Script principal (stable) :
  - `march_madness_2026_run.py`
- Scripts d'experiences :
  - `experiments/`
- Snapshots "meilleurs runs" (versions figees) :
  - `snapshots/`
- Outil de blend de soumissions :
  - `blend_submissions.py`

## Registre des scripts / scores

| ID | Script | Base | Changements principaux | Score Kaggle Stage 1 | Statut | Fichiers lies |
|---|---|---|---|---:|---|---|
| `stable_v1_platt` | `march_madness_2026_run.py` | pipeline initial | LR + XGBoost, CV temporelle, calibration Platt, clipping | `0.14042` | Stable / reference | `submission_stage1.csv`, `submission_stage2.csv` |
| `exp_hgb_imputation` | `experiments/march_madness_2026_run_exp_hgb_imputation_20260223.py` | `stable_v1_platt` | imputation + indicateurs de manquants + `HistGradientBoosting` + blend 3 modeles | `0.15144` | Echec (degrade) | (sorties ecrasees a l'epoque du test) |
| `exp_elo_mov_recent` | `experiments/march_madness_2026_run_exp_elo_mov_recent.py` | `stable_v1_platt` | Elo avec marge de victoire (MoV) + features de forme recente (12 derniers matchs) | `0.14451` | Mieux que `exp_hgb_imputation`, mais < stable | `submission_stage1_exp_elo_mov_recent.csv`, `submission_stage2_exp_elo_mov_recent.csv` |
| `exp_elo_mov_only` | `experiments/march_madness_2026_run_exp_elo_mov_only.py` | `stable_v1_platt` | Elo avec marge de victoire (MoV) uniquement (ablation propre) | **`0.13469`** | **Meilleur score actuel** | `submission_stage1_exp_elo_mov_only.csv`, `submission_stage2_exp_elo_mov_only.csv` |
| `best_0_13469_snapshot` | `snapshots/march_madness_2026_run_best_0_13469.py` | copie de `exp_elo_mov_only` | snapshot fige du meilleur script | **`0.13469`** | Snapshot / rollback facile | `submission_stage1_best_0_13469.csv`, `submission_stage2_best_from_0_13469_stage1.csv` |
| `exp_elo_mov_tuned` | `experiments/march_madness_2026_run_exp_elo_mov_tuned.py` | `exp_elo_mov_only` | tuning Optuna des parametres Elo MoV (`k`, `home_adv`, `mean_revert`, `mov_cap`, `mov_scale`) | `0.15082` | Echec (degrade) | `submission_stage1_exp_elo_mov_tuned.csv`, `submission_stage2_exp_elo_mov_tuned.csv` |
| `blend_tool` | `blend_submissions.py` | outil standalone | blend de 2 soumissions + recherche d'`alpha` locale sur Stage 1 | n/a | Outil | `submission_stage1_blend_*.csv`, `submission_stage2_blend_*.csv`, `blend_alpha_search_*.csv` |

## Timeline rapide des scores Stage 1 (Kaggle)

| Ordre | Experience | Score |
|---|---|---:|
| 1 | `stable_v1_platt` | `0.14042` |
| 2 | `exp_hgb_imputation` | `0.15144` |
| 3 | `exp_elo_mov_recent` | `0.14451` |
| 4 | `exp_elo_mov_only` | **`0.13469`** |
| 5 | `exp_elo_mov_tuned` | `0.15082` |

## Interprétation (ce qui a aidé / ce qui a degradé)

- **A aidé**
  - Elo avec marge de victoire (MoV) **sans** ajouter trop de complexite (`exp_elo_mov_only`)
- **A degrade**
  - Ajouter un modele `HistGradientBoosting` + imputation globale (`exp_hgb_imputation`)
  - Tuning Elo optimise sur un objectif Elo seul (pas aligne avec le pipeline final) (`exp_elo_mov_tuned`)
- **Gain partiel mais insuffisant**
  - Elo MoV + forme recente (`exp_elo_mov_recent`) : mieux que certaines experiences, mais moins bon que `exp_elo_mov_only`

## Convention pour les prochains tests

### Scripts

- Stable (production) : `march_madness_2026_run.py`
- Nouvelle experience : `experiments/march_madness_2026_run_exp_<idee>.py`
- Snapshot d'un meilleur run : `snapshots/march_madness_2026_run_best_<score>.py`

### Soumissions

- Stage 1 (meilleur connu) : `submission_stage1_best_<score>.csv`
- Stage 2 associe au meme run : `submission_stage2_best_from_<score>_stage1.csv`
- Experiments : `submission_stage1_exp_<idee>.csv`, `submission_stage2_exp_<idee>.csv`

## Commandes utiles

### Lancer le script stable

```powershell
cd "C:\Users\Amaury\Documents\kaggle\La folie du Machine Learning"
python .\march_madness_2026_run.py
```

### Lancer une experience

```powershell
python .\experiments\march_madness_2026_run_exp_elo_mov_only.py
```

### Blender 2 soumissions

```powershell
python .\blend_submissions.py `
  --stage1-a .\submission_stage1_best_0_13469.csv `
  --stage1-b .\submission_stage1.csv `
  --stage2-a .\submission_stage2_best_from_0_13469_stage1.csv `
  --stage2-b .\submission_stage2.csv `
  --grid-step 0.01 `
  --write-top-k 3 `
  --out-tag best_vs_stable
```

## Template de nouvelle entree (copier-coller)

```md
| `exp_<nom>` | `experiments/march_madness_2026_run_exp_<nom>.py` | `<base>` | `<changements>` | `<score>` | `<statut>` | `<fichiers>` |
```
