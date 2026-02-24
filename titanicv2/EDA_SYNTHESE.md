# Titanic V2 - Synthese EDA

**Competition:** Titanic - Machine Learning from Disaster
**Objectif:** Classification binaire (Survived: 0/1)
**Metrique:** Accuracy
**Donnees:** 891 train / 418 test / 12 colonnes

## Donnees brutes

- **Train:** 891 lignes, 12 colonnes
- **Test:** 418 lignes, 11 colonnes (sans Survived)
- **Target:** 38.4% survivants / 61.6% morts (desequilibre modere)

## Valeurs manquantes

| Feature  | Train          | Test           | Strategie d'imputation         |
|----------|----------------|----------------|--------------------------------|
| Cabin    | 687 (77.1%)    | 327 (78.2%)    | HasCabin flag + CabinDeck      |
| Age      | 177 (19.9%)    | 86 (20.6%)     | Mediane par Title              |
| Embarked | 2 (0.2%)       | 0              | Mode ('S')                     |
| Fare     | 0              | 1 (0.2%)       | Mediane par Pclass             |

## Analyse par feature

### Sex (feature #1)

| Sex    | Survie | Effectif |
|--------|--------|----------|
| Female | 74.2%  | 314      |
| Male   | 18.9%  | 577      |

- Ratio 4x entre femmes et hommes
- Un modele base uniquement sur Sex donne ~78% accuracy
- Principe "women and children first" clairement visible

### Pclass (feature #2)

| Classe | Survie | Effectif |
|--------|--------|----------|
| 1st    | 63.0%  | 216      |
| 2nd    | 47.3%  | 184      |
| 3rd    | 24.2%  | 491      |

- Gradient lineaire de survie
- La 3eme classe represente 55% du dataset mais seulement 24% de survie

### Sex x Pclass (interaction critique)

|         | 1st    | 2nd    | 3rd    |
|---------|--------|--------|--------|
| Female  | 96.8%  | 92.1%  | 50.0%  |
| Male    | 36.9%  | 15.7%  | 13.5%  |

- Femmes 1ere/2eme classe: quasi-certitude de survie (~95%)
- Hommes 2eme/3eme classe: quasi-certitude de mort (~14%)
- **Cas difficiles:** femmes 3eme classe (50%) et hommes 1ere classe (37%)

### Title (extrait du Name)

| Title  | Survie | Effectif | Description           |
|--------|--------|----------|-----------------------|
| Mr     | 15.7%  | 517      | Hommes adultes        |
| Miss   | 69.8%  | 182      | Femmes non mariees    |
| Mrs    | 79.2%  | 125      | Femmes mariees        |
| Master | 57.5%  | 40       | Garcons               |
| Rev    | 0.0%   | 6        | Reverends (honneur)   |

- Master (garcons) = 57.5%, confirme "children first"
- Regroupement optimal: Mr, Miss, Mrs, Master, Rare

### Family Size (SibSp + Parch + 1)

| Taille   | Survie   | Effectif | Categorie |
|----------|----------|----------|-----------|
| 1 (seul) | 30.4%    | 537      | Alone     |
| 2        | 55.3%    | 161      | Small     |
| 3        | 57.8%    | 102      | Small     |
| 4        | 72.4%    | 29       | Small     |
| 5+       | 0-20%    | 62       | Large     |

- Pattern en U inverse: taille 2-4 = sweet spot
- Seul ou grande famille = danger
- FamilyCategory (Alone/Small/Large) capture ce pattern non-lineaire

### Fare

- Distribution tres skewed: mean=32.20, median=14.45, max=512.33
- 15 passagers avec Fare=0
- Log transform necessaire (LogFare)
- Proxy de classe sociale (Pclass vs Fare: r=-0.55)

### Cabin

- 77% manquant, donc HasCabin est un proxy de richesse
- HasCabin=1: 66.7% survie vs HasCabin=0: 30.0%
- Decks B, D, E: ~75% survie (proximite des canots)
- Pclass vs HasCabin: r=-0.73 (forte redondance)

### Age

- Moyenne: 29.7 ans, mediane: 28.0 ans
- Enfants (<10): 61.3% survie
- Adultes (10-60): 39.1%
- Seniors (>60): 26.9%
- Faible predicteur seul, mais fort en interaction (Age x Sex, Age x Pclass)

### Embarked

| Port         | Survie | Effectif |
|--------------|--------|----------|
| C (Cherbourg)    | 55.4%  | 168      |
| Q (Queenstown)   | 39.0%  | 77       |
| S (Southampton)  | 33.7%  | 644      |

- Cherbourg = plus de passagers 1ere classe
- Southampton = 72% du volume, taux le plus bas
- Proxy indirect de la classe sociale

## Correlations avec Survived (numeriques)

| Feature    | Correlation | Direction                      |
|------------|-------------|--------------------------------|
| Sex*       | ~0.54       | Femme = survie                 |
| Pclass     | -0.34       | Classe basse = mort            |
| HasCabin   | +0.32       | Cabine = richesse = survie     |
| Fare       | +0.26       | Prix eleve = meilleure classe  |
| Parch      | +0.08       | Leger positif                  |
| Age        | -0.08       | Leger negatif                  |

*Sex est categoriel, correlation point-biseriale estimee.

## Features a creer (notebook 02)

1. **Title** (extrait du Name) - 5 groupes
2. **FamilySize** + **IsAlone** + **FamilyCategory**
3. **HasCabin** + **CabinDeck** + **CabinCount**
4. **AgeBin** + **IsChild** + **Age_Pclass**
5. **LogFare** + **FarePerPerson** + **FareBin**
6. **TicketFreq** + **TicketPrefix** + **TicketIsNum**
7. **SurnameFreq** (group survival trick)

**Total: 24 features a partir de 11 colonnes originales**

## Strategie de modelisation

- **Modeles:** LightGBM, XGBoost, CatBoost, Random Forest, Logistic Regression, SVM
- **CV:** 5-Fold Stratified
- **Tuning:** Optuna (100 trials par modele GBDT)
- **Ensemble:** Weighted Average, Rank Average, Majority Voting, Stacking
