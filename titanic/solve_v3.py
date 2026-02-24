import pandas as pd
import numpy as np
import warnings
import optuna
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
np.random.seed(SEED)

# =================================================================
# 1. LOAD
# =================================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_len = len(train)
passenger_ids = test["PassengerId"]
df = pd.concat([train, test], sort=False).reset_index(drop=True)

# =================================================================
# 2. FEATURE ENGINEERING
# =================================================================

# --- Title ---
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map_raw = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Mrs", "Countess": "Mrs", "Dona": "Mrs",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Capt": "Rare", "Sir": "Rare", "Don": "Rare", "Jonkheer": "Rare",
}
df["Title"] = df["Title"].map(lambda x: title_map_raw.get(x, "Rare"))

# --- Family ---
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
# Survival-optimal family size bins: 1=alone, 2-4=small, 5+=large
df["FamilySizeBin"] = df["FamilySize"].map(
    lambda x: "alone" if x == 1 else ("small" if x <= 4 else "large")
)

# --- Ticket group ---
ticket_counts = df["Ticket"].value_counts()
df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)

# --- Cabin / Deck ---
df["HasCabin"] = df["Cabin"].notna().astype(int)
df["Deck"] = df["Cabin"].str[0].fillna("U")

# --- Fill Age by (Title, Pclass) median ---
df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
df["Age"] = df["Age"].fillna(df["Age"].median())

# --- Fill Embarked & Fare ---
df["Embarked"] = df["Embarked"].fillna("S")
df["Fare"] = df.groupby(["Pclass", "Embarked"])["Fare"].transform(
    lambda x: x.fillna(x.median())
)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# --- Fare per person ---
df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

# --- Age features ---
df["IsChild"] = (df["Age"] <= 12).astype(int)
df["IsYoungFemale"] = ((df["Sex"] == "female") & (df["Age"] <= 30)).astype(int)
df["AgeBin"] = pd.cut(df["Age"], bins=[0, 5, 12, 18, 30, 45, 60, 120],
                      labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)

# --- Fare bins ---
df["FareBin"] = pd.qcut(df["FarePerPerson"], 5, labels=False, duplicates="drop")

# --- Ticket prefix ---
df["TicketPrefix"] = df["Ticket"].str.extract(r"^([A-Za-z./]+)", expand=False)
df["TicketPrefix"] = df["TicketPrefix"].fillna("NONE")
# Group rare prefixes
tp_counts = df["TicketPrefix"].value_counts()
df["TicketPrefix"] = df["TicketPrefix"].map(
    lambda x: x if tp_counts.get(x, 0) >= 10 else "RARE"
)

# --- Sex * Pclass ---
df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)

# --- Name length ---
df["NameLen"] = df["Name"].str.len()

# --- Mother ---
df["IsMother"] = (
    (df["Sex"] == "female") & (df["Parch"] > 0) &
    (df["Age"] > 18) & (df["Title"] == "Mrs")
).astype(int)

# =================================================================
# 3. GROUP SURVIVAL FEATURES (the key trick for Titanic)
# =================================================================
# The idea: families and ticket groups survived/died together.
# We compute survival rates for surname-groups and ticket-groups
# using ONLY training data, then propagate to test.

df["Surname"] = df["Name"].str.split(",").str[0]

# ---- Surname + Fare group ----
df["FamilyKey"] = df["Surname"] + "_" + df["FamilySize"].astype(str)

train_part = df.iloc[:train_len]

# Family group survival
fam_stats = train_part.groupby("FamilyKey").agg(
    FamSurvMean=("Survived", "mean"),
    FamSurvCount=("Survived", "count"),
).reset_index()
# Only trust groups with >1 member
fam_stats.loc[fam_stats["FamSurvCount"] <= 1, "FamSurvMean"] = 0.5
df = df.merge(fam_stats[["FamilyKey", "FamSurvMean"]], on="FamilyKey", how="left")
df["FamSurvMean"] = df["FamSurvMean"].fillna(0.5)

# Ticket group survival
tick_stats = train_part.groupby("Ticket").agg(
    TickSurvMean=("Survived", "mean"),
    TickSurvCount=("Survived", "count"),
).reset_index()
tick_stats.loc[tick_stats["TickSurvCount"] <= 1, "TickSurvMean"] = 0.5
df = df.merge(tick_stats[["Ticket", "TickSurvMean"]], on="Ticket", how="left")
df["TickSurvMean"] = df["TickSurvMean"].fillna(0.5)

# Combined group survival (max signal)
df["GroupSurvival"] = df[["FamSurvMean", "TickSurvMean"]].max(axis=1)

# --- KEY TRICK: override predictions for specific group patterns ---
# Women/children in groups where EVERYONE died -> likely died too
# Men in groups where EVERYONE survived -> likely survived too
# We encode this as a strong feature
df["GroupSurvHigh"] = (df["GroupSurvival"] >= 0.75).astype(int)
df["GroupSurvLow"] = (df["GroupSurvival"] <= 0.25).astype(int)

# Female in low-survival group
df["FemaleInDyingGroup"] = (
    (df["Sex"] == "female") & (df["GroupSurvLow"] == 1)
).astype(int)

# Male in high-survival group
df["MaleInSurvGroup"] = (
    (df["Sex"] == "male") & (df["GroupSurvHigh"] == 1)
).astype(int)

# =================================================================
# 4. ENCODE CATEGORICALS
# =================================================================
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
embarked_map = {"S": 0, "C": 1, "Q": 2}
df["Embarked"] = df["Embarked"].map(embarked_map)
title_enc = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
df["Title"] = df["Title"].map(title_enc)
deck_enc = {d: i for i, d in enumerate(sorted(df["Deck"].unique()))}
df["Deck"] = df["Deck"].map(deck_enc)
fsb_enc = {"alone": 0, "small": 1, "large": 2}
df["FamilySizeBin"] = df["FamilySizeBin"].map(fsb_enc)
sp_enc = {v: i for i, v in enumerate(df["Sex_Pclass"].unique())}
df["Sex_Pclass"] = df["Sex_Pclass"].map(sp_enc)
tp_enc = {v: i for i, v in enumerate(df["TicketPrefix"].unique())}
df["TicketPrefix"] = df["TicketPrefix"].map(tp_enc)

# =================================================================
# 5. FEATURE SELECTION
# =================================================================
features = [
    "Pclass", "Sex", "AgeBin", "FareBin", "FarePerPerson", "Title",
    "FamilySize", "IsAlone", "FamilySizeBin",
    "SibSp", "Parch", "Embarked",
    "HasCabin", "Deck",
    "TicketGroupSize", "TicketPrefix",
    "IsChild", "IsYoungFemale", "Sex_Pclass",
    "NameLen", "IsMother",
    "FamSurvMean", "TickSurvMean", "GroupSurvival",
    "GroupSurvHigh", "GroupSurvLow",
    "FemaleInDyingGroup", "MaleInSurvGroup",
]

X_train = df[features].iloc[:train_len]
y_train = df["Survived"].iloc[:train_len].astype(int)
X_test = df[features].iloc[train_len:]

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=features)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# =================================================================
# 6. OPTUNA HYPERPARAMETER TUNING
# =================================================================
print("=" * 55)
print("OPTUNA HYPERPARAMETER TUNING (50 trials each)")
print("=" * 55)


def objective_xgb(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": SEED,
    }
    m = XGBClassifier(**p)
    return cross_val_score(m, X_train, y_train, cv=skf, scoring="accuracy").mean()


def objective_lgbm(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "random_state": SEED,
        "verbose": -1,
    }
    m = LGBMClassifier(**p)
    return cross_val_score(m, X_train, y_train, cv=skf, scoring="accuracy").mean()


def objective_gbc(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": SEED,
    }
    m = GradientBoostingClassifier(**p)
    return cross_val_score(m, X_train, y_train, cv=skf, scoring="accuracy").mean()


def objective_rf(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": SEED,
    }
    m = RandomForestClassifier(**p)
    return cross_val_score(m, X_train, y_train, cv=skf, scoring="accuracy").mean()


def objective_svc(trial):
    p = {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": "rbf",
        "probability": True,
        "random_state": SEED,
    }
    m = SVC(**p)
    return cross_val_score(m, X_train_sc, y_train, cv=skf, scoring="accuracy").mean()


def objective_lr(trial):
    p = {
        "C": trial.suggest_float("C", 0.001, 100.0, log=True),
        "penalty": "l2",
        "max_iter": 2000,
        "random_state": SEED,
    }
    m = LogisticRegression(**p)
    return cross_val_score(m, X_train_sc, y_train, cv=skf, scoring="accuracy").mean()


N_TRIALS = 50
tuners = {
    "XGB": objective_xgb,
    "LGBM": objective_lgbm,
    "GBC": objective_gbc,
    "RF": objective_rf,
    "SVC": objective_svc,
    "LR": objective_lr,
}

best_params = {}
for name, obj in tuners.items():
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    best_params[name] = study.best_params
    print(f"  {name:6s}  best CV = {study.best_value:.4f}")

# =================================================================
# 7. BUILD TUNED MODELS
# =================================================================
print("\n" + "=" * 55)
print("BUILDING TUNED ENSEMBLE")
print("=" * 55)

bp = best_params
models = {
    "XGB": XGBClassifier(
        **bp["XGB"], eval_metric="logloss", use_label_encoder=False, random_state=SEED
    ),
    "LGBM": LGBMClassifier(**bp["LGBM"], random_state=SEED, verbose=-1),
    "GBC": GradientBoostingClassifier(**bp["GBC"], random_state=SEED),
    "RF": RandomForestClassifier(**bp["RF"], random_state=SEED),
    "ET": ExtraTreesClassifier(
        n_estimators=500, max_depth=bp["RF"].get("max_depth", 6),
        min_samples_split=bp["RF"].get("min_samples_split", 8),
        min_samples_leaf=bp["RF"].get("min_samples_leaf", 4),
        random_state=SEED,
    ),
    "SVC": SVC(**bp["SVC"], kernel="rbf", probability=True, random_state=SEED),
    "LR": LogisticRegression(**bp["LR"], penalty="l2", max_iter=2000, random_state=SEED),
}

scale_models = {"SVC", "LR"}

for name, model in models.items():
    if name in scale_models:
        s = cross_val_score(model, X_train_sc, y_train, cv=skf, scoring="accuracy")
    else:
        s = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    print(f"  {name:6s}  CV = {s.mean():.4f} (+/- {s.std():.4f})")

# =================================================================
# 8. STACKING
# =================================================================
print("\n" + "=" * 55)
print("STACKING ENSEMBLE")
print("=" * 55)

n_models = len(models)
S_train = np.zeros((train_len, n_models))
S_test = np.zeros((len(X_test), n_models))

for i, (name, model) in enumerate(models.items()):
    S_test_folds = np.zeros((len(X_test), 10))
    for j, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        if name in scale_models:
            xtr, xval = X_train_sc.iloc[tr_idx], X_train_sc.iloc[val_idx]
        else:
            xtr, xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr = y_train.iloc[tr_idx]

        model.fit(xtr, ytr)
        S_train[val_idx, i] = model.predict_proba(xval)[:, 1]
        if name in scale_models:
            S_test_folds[:, j] = model.predict_proba(X_test_sc)[:, 1]
        else:
            S_test_folds[:, j] = model.predict_proba(X_test)[:, 1]

    S_test[:, i] = S_test_folds.mean(axis=1)
    print(f"  {name} done")

# Meta-learner
meta = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
meta_cv = cross_val_score(meta, S_train, y_train, cv=skf, scoring="accuracy")
print(f"\n  Stacking meta CV: {meta_cv.mean():.4f} (+/- {meta_cv.std():.4f})")
meta.fit(S_train, y_train)
stack_preds = meta.predict(S_test)
stack_proba = meta.predict_proba(S_test)[:, 1]

# =================================================================
# 9. WEIGHTED SOFT VOTING
# =================================================================
print("\n" + "=" * 55)
print("WEIGHTED SOFT VOTING")
print("=" * 55)

# Use Optuna to find best weights
def objective_weights(trial):
    w = {}
    for name in models:
        w[name] = trial.suggest_float(f"w_{name}", 0.0, 3.0)
    total = sum(w.values())
    if total < 0.1:
        return 0.0

    proba = np.zeros(train_len)
    for idx, (name, model) in enumerate(models.items()):
        proba += w[name] * S_train[:, idx]
    proba /= total
    preds = (proba >= 0.5).astype(int)
    return (preds == y_train).mean()


study_w = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=SEED))
study_w.optimize(objective_weights, n_trials=200, show_progress_bar=False)

best_w = {name: study_w.best_params[f"w_{name}"] for name in models}
total_w = sum(best_w.values())
print("  Best weights:")
for name, w in best_w.items():
    print(f"    {name:6s}: {w:.3f} ({w/total_w*100:.1f}%)")

# Compute voting probabilities using stacking OOF predictions
vote_proba_train = np.zeros(train_len)
vote_proba_test = np.zeros(len(X_test))
for idx, (name, _) in enumerate(models.items()):
    vote_proba_train += best_w[name] * S_train[:, idx]
    vote_proba_test += best_w[name] * S_test[:, idx]
vote_proba_train /= total_w
vote_proba_test /= total_w

vote_preds_train = (vote_proba_train >= 0.5).astype(int)
vote_acc = (vote_preds_train == y_train).mean()
print(f"\n  Voting OOF accuracy: {vote_acc:.4f}")

vote_preds = (vote_proba_test >= 0.5).astype(int)

# =================================================================
# 10. BLEND STACKING + VOTING
# =================================================================
blend_proba = 0.5 * stack_proba + 0.5 * vote_proba_test
blend_preds = (blend_proba >= 0.5).astype(int)

# OOF blend
blend_train_proba = 0.5 * meta.predict_proba(S_train)[:, 1] + 0.5 * vote_proba_train
blend_train_preds = (blend_train_proba >= 0.5).astype(int)
blend_acc = (blend_train_preds == y_train).mean()
print(f"  Blend OOF accuracy:  {blend_acc:.4f}")

# =================================================================
# 11. CHOOSE BEST & SAVE
# =================================================================
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Stacking CV:   {meta_cv.mean():.4f}")
print(f"  Voting OOF:    {vote_acc:.4f}")
print(f"  Blend OOF:     {blend_acc:.4f}")

results = {
    "stack": (meta_cv.mean(), stack_preds),
    "vote": (vote_acc, vote_preds),
    "blend": (blend_acc, blend_preds),
}
best_name = max(results, key=lambda k: results[k][0])
final_preds = results[best_name][1]
print(f"  -> Using {best_name.upper()} (score={results[best_name][0]:.4f})")

submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": final_preds.astype(int),
})
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(submission["Survived"].value_counts())
