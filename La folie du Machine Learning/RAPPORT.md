# March Machine Learning Mania 2026 -- Rapport de Projet

## Table des matieres

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Description des donnees](#2-description-des-donnees)
3. [Feature engineering](#3-feature-engineering)
4. [Construction du dataset d'entrainement](#4-construction-du-dataset-dentrainement)
5. [Modelisation](#5-modelisation)
6. [Generation des soumissions](#6-generation-des-soumissions)
7. [Architecture du pipeline](#7-architecture-du-pipeline)

---

## 1. Contexte et objectif

### La competition Kaggle

**March Machine Learning Mania 2026** est une competition annuelle sur Kaggle ou les participants doivent predire les resultats du tournoi NCAA de basketball (March Madness) pour les equipes masculines **et** feminines.

### Objectif

Pour chaque paire d'equipes susceptible de se rencontrer en tournoi, on doit predire la **probabilite que l'equipe A batte l'equipe B** (ou A est l'equipe avec le plus petit `TeamID`). La metrique d'evaluation est le **Brier Score** (erreur quadratique moyenne entre la probabilite predite et le resultat binaire).

### Format de soumission

Le fichier de soumission contient des paires au format `Season_TeamA_TeamB` avec une colonne `Pred` :

| ID | Pred |
|---|---|
| `2026_1101_1102` | 0.5 |
| `2026_1101_1103` | 0.5 |

- **Stage 1** : Paires couvrant les saisons 2022-2025 (validation historique)
- **Stage 2** : Paires pour le tournoi 2026 (predictions reelles)

### Strategie retenue

Ensemble de **Regression Logistique + XGBoost** avec optimisation des poids, **calibration Platt** sur les predictions de l'ensemble (OOF), modeles **separes Hommes/Femmes**, validation croisee temporelle.

---

## 2. Description des donnees

### Vue d'ensemble

Les donnees sont reparties entre basketball masculin (prefixe `M`) et feminin (prefixe `W`), couvrant les saisons de **1985 a 2026** (hommes) et **1998 a 2026** (femmes).

### Fichiers principaux utilises

| Fichier | Taille | Description |
|---|---|---|
| `MTeams.csv` / `WTeams.csv` | 10 Ko / 6 Ko | Repertoire des equipes (`TeamID`, `TeamName`, periodes D1) |
| `MRegularSeasonCompactResults.csv` | 5.6 Mo | Resultats compacts saison reguliere hommes |
| `MRegularSeasonDetailedResults.csv` | 11.6 Mo | Resultats detailles saison reguliere hommes (box scores) |
| `WRegularSeasonCompactResults.csv` | 4.0 Mo | Resultats compacts saison reguliere femmes |
| `WRegularSeasonDetailedResults.csv` | 8.1 Mo | Resultats detailles saison reguliere femmes |
| `MNCAATourneyCompactResults.csv` | 76 Ko | Resultats du tournoi NCAA hommes |
| `WNCAATourneyCompactResults.csv` | 50 Ko | Resultats du tournoi NCAA femmes |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 39 Ko / 26 Ko | Seeds du tournoi par equipe et saison |
| `MMasseyOrdinals.csv` | 121 Mo | Rankings de multiples systemes de classement (hommes uniquement) |
| `MTeamCoaches.csv` | 405 Ko | Historique des coachs par equipe/saison |
| `MTeamConferences.csv` / `WTeamConferences.csv` | 232 Ko / 166 Ko | Conference de chaque equipe par saison |
| `Conferences.csv` | 1.6 Ko | Nom complet de chaque conference |
| `SampleSubmissionStage1.csv` | 9.9 Mo | Template de soumission Stage 1 |
| `SampleSubmissionStage2.csv` | 2.5 Mo | Template de soumission Stage 2 |

### Schemas des tables cles

#### Resultats compacts (`MRegularSeasonCompactResults.csv`)

| Colonne | Type | Description |
|---|---|---|
| `Season` | int | Annee de la saison |
| `DayNum` | int | Jour dans la saison (0 = debut) |
| `WTeamID` | int | ID de l'equipe gagnante |
| `WScore` | int | Score du gagnant |
| `LTeamID` | int | ID de l'equipe perdante |
| `LScore` | int | Score du perdant |
| `WLoc` | str | Lieu du match pour le gagnant (`H` = domicile, `A` = exterieur, `N` = neutre) |
| `NumOT` | int | Nombre de prolongations |

#### Resultats detailles (`MRegularSeasonDetailedResults.csv`)

En plus des colonnes compactes, chaque match inclut les **box scores complets** pour les deux equipes (prefixe `W` pour le gagnant, `L` pour le perdant) :

| Suffixe | Description |
|---|---|
| `FGM` / `FGA` | Tirs reussis / tentes |
| `FGM3` / `FGA3` | Tirs a 3 points reussis / tentes |
| `FTM` / `FTA` | Lancers francs reussis / tentes |
| `OR` / `DR` | Rebonds offensifs / defensifs |
| `Ast` | Passes decisives |
| `TO` | Balles perdues (turnovers) |
| `Stl` | Interceptions |
| `Blk` | Contres |
| `PF` | Fautes personnelles |

#### Seeds (`MNCAATourneySeeds.csv`)

| Colonne | Type | Description |
|---|---|---|
| `Season` | int | Annee |
| `Seed` | str | Code seed (ex: `W01`, `X16a`) -- lettre = region, nombre = classement 1-16 |
| `TeamID` | int | ID de l'equipe |

#### Massey Ordinals (`MMasseyOrdinals.csv`)

| Colonne | Type | Description |
|---|---|---|
| `Season` | int | Annee |
| `RankingDayNum` | int | Jour du classement |
| `SystemName` | str | Nom du systeme de classement (ex: `POM`, `SAG`, `AP`) |
| `TeamID` | int | ID de l'equipe |
| `OrdinalRank` | int | Rang dans ce systeme (1 = meilleur) |

---

## 3. Feature engineering

Le pipeline construit **4 familles de features** a partir des donnees brutes.

### 3.1 Systeme Elo

Un rating Elo est calcule pour chaque equipe en parcourant chronologiquement tous les matchs de saison reguliere.

**Parametres :**
- `K = 20` (facteur de mise a jour)
- `home_adv = 100` (bonus Elo pour le domicile)
- `mean_revert = 0.75` (regression vers 1500 entre les saisons)

**Formule :**
```
P(victoire) = 1 / (1 + 10^((Elo_adversaire - Elo_equipe) / 400))
Mise a jour : Elo += K * (resultat - P(victoire))
```

A chaque nouvelle saison, les ratings regressent vers la moyenne :
```
Elo_new = 1500 + 0.75 * (Elo_old - 1500)
```

**Sortie :** Un rating Elo de fin de saison par `(Season, TeamID)`.

### 3.2 Massey Composite (hommes uniquement)

Le fichier `MMasseyOrdinals.csv` contient les classements de dizaines de systemes independants. Le pipeline :

1. **Selectionne 10 systemes reconnus** : POM, SAG, MOR, WLK, DOL, COL, RPI, AP, USA, NET
2. **Filtre sur la fin de saison** : `RankingDayNum >= 128`
3. **Prend le classement le plus recent** par `(Season, SystemName, TeamID)`
4. **Calcule la moyenne** des rangs pour obtenir un **rang composite**

**Sortie :** Un rang moyen composite par `(Season, TeamID)` -- plus le rang est bas, meilleure est l'equipe.

### 3.3 Les 4 Facteurs de Dean Oliver

Ce sont les statistiques avancees les plus reconnues en basketball analytics, calculees a partir des box scores detailles de la saison reguliere.

#### Facteurs offensifs

| Facteur | Formule | Interpretation |
|---|---|---|
| **eFG%** (effective FG%) | `(FGM + 0.5 * FGM3) / FGA` | Efficacite au tir (bonus 3 pts) |
| **TOV%** (turnover rate) | `TO / Possessions` | Taux de balles perdues (bas = bon) |
| **ORB%** (offensive rebound rate) | `OR / (OR + Opp_DR)` | Taux de rebonds offensifs |
| **FTr** (free throw rate) | `FTM / FGA` | Capacite a aller aux lancers francs |

Ou les **possessions** sont estimees par :
```
Poss = FGA - OR + TO + 0.475 * FTA
```

#### Facteurs defensifs

Les memes 4 facteurs calcules **du point de vue de l'adversaire** (prefixe `Opp_`).

#### Facteurs nets

Difference offensif - defensif, mesurant l'avantage global de l'equipe :
- `Net_eFG = eFG% - Opp_eFG%`
- `Net_TOV = Opp_TOV% - TOV%` (inverse : forcer les turnovers adverses est positif)
- `Net_ORB = ORB% - Opp_ORB%`
- `Net_FTr = FTr - Opp_FTr`

**Sortie :** Stats agreges par `(Season, TeamID)` incluant `WinPct`, `AvgMargin`, les 4 facteurs offensifs/defensifs/nets.

### 3.4 Seeds du tournoi

Le numero de seed (1-16) est extrait du code brut (ex: `W01` -> 1, `X16a` -> 16). Un seed 1 est le favori, un seed 16 est le moins bien classe.

### Resume des features finales

Toutes les features sont exprimees en **delta (TeamA - TeamB)** pour capter la difference entre les deux equipes :

| Feature | Description | Hommes | Femmes |
|---|---|---|---|
| `DeltaSeed` | Difference de seed | x | x |
| `DeltaElo` | Difference de rating Elo | x | x |
| `DeltaMassey` | Difference de rang composite Massey | x | -- |
| `Delta_WinPct` | Difference de % de victoires | x | x |
| `Delta_AvgMargin` | Difference de marge moyenne | x | x |
| `Delta_eFGpct` | Difference de eFG% | x | x |
| `Delta_TOVpct` | Difference de taux de turnovers | x | x |
| `Delta_ORBpct` | Difference de taux de rebonds offensifs | x | x |
| `Delta_FTr` | Difference de free throw rate | x | x |
| `Delta_Net_eFG` | Difference de eFG% net | x | x |
| `Delta_Net_TOV` | Difference de TOV% net | x | x |
| `Delta_Net_ORB` | Difference de ORB% net | x | x |
| `Delta_Net_FTr` | Difference de FTr net | x | x |

**Total : 13 features (hommes) / 12 features (femmes)**

---

## 4. Construction du dataset d'entrainement

### Orientation des paires

Chaque match de tournoi historique est converti en une **paire orientee** :
- `TeamA` = equipe avec le plus petit `TeamID`
- `TeamB` = equipe avec le plus grand `TeamID`
- `Target` = 1 si TeamA a gagne, 0 sinon

Ce schema garantit la coherence avec le format de soumission Kaggle.

### Donnees utilisees

| Genre | Matchs de tournoi | Apres nettoyage NaN |
|---|---|---|
| Hommes | ~2 400 matchs | ~1 500 (retrait des saisons sans Massey) |
| Femmes | ~1 600 matchs | ~1 600 |

La cible `Target` est centree autour de 0.50, confirmant que l'orientation par `TeamID` n'introduit pas de biais.

---

## 5. Modelisation

### 5.1 Baseline : Regression Logistique

- **Preprocessing** : `StandardScaler` sur les features
- **Regularisation** : `C = 1.0` (defaut)
- **Validation** : CV temporelle (entrainer sur les N premieres saisons, valider sur la N+1e)

### 5.2 Modele principal : XGBoost

- **Objectif** : `binary:logistic`
- **Metrique** : RMSE (proxy du Brier Score)
- **Early stopping** : 50 rounds sans amelioration, max 500 rounds
- **Optimisation** : Optuna (20 trials hommes, 50 trials femmes)

**Espace de recherche des hyperparametres :**

| Parametre | Intervalle |
|---|---|
| `max_depth` | 2 - 6 (hommes) / 2 - 5 (femmes) |
| `learning_rate` | 0.01 - 0.3 (log) |
| `subsample` | 0.6 - 1.0 |
| `colsample_bytree` | 0.5 - 1.0 |
| `reg_alpha` | 1e-4 - 10 (log) |
| `reg_lambda` | 1e-4 - 10 (log) |
| `min_child_weight` | 1 - 10 |

### 5.3 Ensemble LR + XGBoost

Les predictions des deux modeles sont combinees par moyenne ponderee :

```
Pred_ensemble = alpha * Pred_LR + (1 - alpha) * Pred_XGBoost
```

Le poids `alpha` est optimise par recherche bornee (`scipy.optimize.minimize_scalar`) pour minimiser le Brier Score sur les predictions out-of-fold de la CV temporelle.

### 5.4 Calibration Platt (sigmoide)

Une **calibration Platt** est appliquee sur la prediction de l'ensemble pour mieux calibrer les probabilites :

1. generation des predictions **out-of-fold** de l'ensemble (CV temporelle)
2. transformation en **logit** des probabilites brutes
3. apprentissage d'une **regression logistique 1D** (sigmoide de Platt)
4. application du calibrateur sur les predictions finales de soumission

Cette etape est directement pertinente pour la metrique **Brier Score**, qui penalise les probabilites mal calibrees.

### Validation croisee temporelle

La validation est **strictement temporelle** pour eviter le data leakage :

```
Saison 1 ... Saison N   ->  Entrainement
Saison N+1               ->  Validation
```

On commence a valider apres un minimum de 10 saisons d'entrainement.

---

## 6. Generation des soumissions

### Entrainement final

Les modeles finaux (LR + XGBoost) sont entraines sur **l'integralite des donnees historiques** disponibles, avec les meilleurs hyperparametres identifies par Optuna. Le calibrateur de Platt est ajuste sur les predictions out-of-fold historiques de l'ensemble puis reutilise pour calibrer les probabilites finales.

### Prediction vectorisee

Les predictions de soumission sont generees par **merges pandas** (et non en boucle ligne par ligne) pour des performances ~100x superieures :

1. Parsing des IDs (`Season_TeamA_TeamB`)
2. Merge des seeds, Elo, stats, et Massey composite
3. Calcul des deltas
4. Prediction LR + XGBoost -> ensemble
5. **Calibration Platt** (sigmoide) de l'ensemble
6. **Clipping** des probabilites dans `[0.02, 0.98]` pour eviter les penalites extremes du Brier Score

### Separation hommes / femmes

Les equipes masculines ont un `TeamID < 3000`, les feminines `>= 3000`. Les deux genres sont traites avec des modeles distincts puis recombines dans le fichier de soumission final.

### Fichiers de sortie

| Fichier | Contenu |
|---|---|
| `submission_stage1.csv` | Predictions sur les saisons 2022-2025 (validation) |
| `submission_stage2.csv` | Predictions pour le tournoi 2026 (soumission finale) |

---

## 7. Architecture du pipeline

```
Donnees brutes (CSV)
    |
    v
+-------------------------------------------+
| 1. Chargement & EDA                       |
|    - Distribution des seeds                |
|    - Taux d'upsets                         |
|    - Matrice de matchups par seed          |
+-------------------------------------------+
    |
    v
+-------------------------------------------+
| 2. Feature Engineering                     |
|    - Elo ratings (saison reguliere)        |
|    - Massey composite (hommes)             |
|    - 4 Facteurs de Dean Oliver             |
|    - Seeds du tournoi                      |
+-------------------------------------------+
    |
    v
+-------------------------------------------+
| 3. Construction des paires                 |
|    - Matchs de tournoi -> paires orientees |
|    - Calcul des deltas (A - B)             |
+-------------------------------------------+
    |
    v
+-------------------------------------------+
| 4. Modelisation                            |
|    - Baseline : Regression Logistique      |
|    - XGBoost + Optuna                      |
|    - Ensemble LR + XGB (alpha optimal)     |
|    - Calibration Platt (OOF)               |
|    - Validation : CV temporelle            |
+-------------------------------------------+
    |
    v
+-------------------------------------------+
| 5. Soumission                              |
|    - Entrainement final (toutes saisons)   |
|    - Predictions vectorisees               |
|    - Calibration Platt                     |
|    - Clipping [0.02, 0.98]                 |
|    - submission_stage1.csv                 |
|    - submission_stage2.csv                 |
+-------------------------------------------+
```

### Dependances

| Librairie | Usage |
|---|---|
| `pandas`, `numpy` | Manipulation de donnees |
| `scikit-learn` | Regression Logistique, StandardScaler, metriques |
| `xgboost` | Modele de gradient boosting |
| `optuna` | Optimisation bayesienne des hyperparametres |
| `scipy` | Optimisation du poids d'ensemble |
| `matplotlib`, `seaborn` | Visualisations EDA |
