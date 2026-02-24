"""
Titanic v4 - Focus on GENERALIZATION, not CV score.
Key principles:
  - No leaky group-survival features (inflate CV, hurt LB)
  - Fewer, proven features that generalize
  - Strong regularization
  - Simple ensemble of well-tuned models
"""
import pandas as pd
import numpy as np
import warnings
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
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
# 2. CLEAN FEATURE ENGINEERING (no leaky features)
# =================================================================

# --- Title (proven #1 feature) ---
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace(["Mme"], "Mrs")
df["Title"] = df["Title"].replace(
    ["Lady", "Countess", "Dona"], "Mrs"
)
df["Title"] = df["Title"].replace(
    ["Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer"], "Rare"
)
title_enc = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
df["Title_enc"] = df["Title"].map(title_enc)

# --- Sex ---
df["Sex_enc"] = df["Sex"].map({"male": 0, "female": 1})

# --- Family size ---
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
# Optimal grouping: alone=bad, small(2-4)=good, large(5+)=bad
df["FamilyGroup"] = df["FamilySize"].map(
    lambda x: 0 if x == 1 else (1 if x <= 4 else 2)
)

# --- Ticket group size (how many share same ticket - NOT survival rate) ---
ticket_counts = df["Ticket"].value_counts()
df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)

# --- Cabin ---
df["HasCabin"] = df["Cabin"].notna().astype(int)
df["Deck"] = df["Cabin"].str[0].fillna("U")
deck_surv = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "U": 0}
df["Deck_enc"] = df["Deck"].map(deck_surv)

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

# --- Fare per person (divide by ticket group) ---
df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

# --- Age bins ---
df["AgeBin"] = pd.cut(
    df["Age"], bins=[0, 5, 12, 18, 30, 50, 120], labels=[0, 1, 2, 3, 4, 5]
).astype(int)
df["IsChild"] = (df["Age"] <= 12).astype(int)

# --- Fare bins ---
df["FareBin"] = pd.qcut(df["FarePerPerson"], 4, labels=False, duplicates="drop")

# --- Embarked encoding ---
embarked_map = {"S": 0, "C": 1, "Q": 2}
df["Embarked_enc"] = df["Embarked"].map(embarked_map)

# --- Interactions (proven to help) ---
df["Sex_Pclass"] = df["Sex_enc"] * 10 + df["Pclass"]
df["Age_Pclass"] = df["AgeBin"] * 10 + df["Pclass"]
df["Child_Sex"] = df["IsChild"] * 10 + df["Sex_enc"]

# =================================================================
# 3. FEATURE SELECTION (lean set)
# =================================================================
features = [
    "Pclass", "Sex_enc", "Title_enc", "AgeBin", "FareBin",
    "FarePerPerson", "FamilySize", "FamilyGroup", "IsAlone",
    "SibSp", "Parch",
    "Embarked_enc", "HasCabin", "Deck_enc",
    "TicketGroupSize",
    "IsChild",
    "Sex_Pclass", "Age_Pclass", "Child_Sex",
]

X_train = df[features].iloc[:train_len]
y_train = df["Survived"].iloc[:train_len].astype(int)
X_test = df[features].iloc[train_len:]

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=features)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# =================================================================
# 4. OPTUNA TUNING (focus on regularization to prevent overfit)
# =================================================================
print("=" * 55)
print("OPTUNA TUNING (60 trials, regularized)")
print("=" * 55)

N_TRIALS = 60


def obj_xgb(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
        "gamma": trial.suggest_float("gamma", 0.1, 5.0),
        "eval_metric": "logloss",
        "random_state": SEED,
    }
    return cross_val_score(
        XGBClassifier(**p), X_train, y_train, cv=skf, scoring="accuracy"
    ).mean()


def obj_lgbm(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 8, 31),
        "verbose": -1,
        "random_state": SEED,
    }
    return cross_val_score(
        LGBMClassifier(**p), X_train, y_train, cv=skf, scoring="accuracy"
    ).mean()


def obj_gbc(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 15),
        "random_state": SEED,
    }
    return cross_val_score(
        GradientBoostingClassifier(**p), X_train, y_train, cv=skf, scoring="accuracy"
    ).mean()


def obj_svc(trial):
    p = {
        "C": trial.suggest_float("C", 0.01, 50.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": "rbf",
        "probability": True,
        "random_state": SEED,
    }
    return cross_val_score(
        SVC(**p), X_train_sc, y_train, cv=skf, scoring="accuracy"
    ).mean()


tuners = {"XGB": obj_xgb, "LGBM": obj_lgbm, "GBC": obj_gbc, "SVC": obj_svc}
best_params = {}
for name, obj in tuners.items():
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(obj, n_trials=N_TRIALS)
    best_params[name] = study.best_params
    print(f"  {name:6s}  best CV = {study.best_value:.4f}")

# =================================================================
# 5. BUILD TUNED MODELS
# =================================================================
bp = best_params
models = {
    "XGB": XGBClassifier(**bp["XGB"], eval_metric="logloss", random_state=SEED),
    "LGBM": LGBMClassifier(**bp["LGBM"], verbose=-1, random_state=SEED),
    "GBC": GradientBoostingClassifier(**bp["GBC"], random_state=SEED),
    "SVC": SVC(**bp["SVC"], kernel="rbf", probability=True, random_state=SEED),
    "LR": LogisticRegression(C=0.5, max_iter=2000, random_state=SEED),
}
scale_models = {"SVC", "LR"}

print("\n" + "=" * 55)
print("TUNED MODEL CV SCORES")
print("=" * 55)
for name, model in models.items():
    X = X_train_sc if name in scale_models else X_train
    s = cross_val_score(model, X, y_train, cv=skf, scoring="accuracy")
    print(f"  {name:6s}  CV = {s.mean():.4f} (+/- {s.std():.4f})")

# =================================================================
# 6. STACKING (clean, no leaky features)
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
            xtest_use = X_test_sc
        else:
            xtr, xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            xtest_use = X_test
        ytr = y_train.iloc[tr_idx]
        model.fit(xtr, ytr)
        S_train[val_idx, i] = model.predict_proba(xval)[:, 1]
        S_test_folds[:, j] = model.predict_proba(xtest_use)[:, 1]

    S_test[:, i] = S_test_folds.mean(axis=1)
    print(f"  {name} done")

meta = LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)
meta_cv = cross_val_score(meta, S_train, y_train, cv=skf, scoring="accuracy")
print(f"\n  Stacking CV: {meta_cv.mean():.4f} (+/- {meta_cv.std():.4f})")
meta.fit(S_train, y_train)
stack_proba = meta.predict_proba(S_test)[:, 1]
stack_preds = (stack_proba >= 0.5).astype(int)

# =================================================================
# 7. SOFT VOTING (equal weights, simple = robust)
# =================================================================
vote_proba = S_test.mean(axis=1)
vote_preds = (vote_proba >= 0.5).astype(int)

vote_train_proba = S_train.mean(axis=1)
vote_cv_preds = (vote_train_proba >= 0.5).astype(int)
vote_oof_acc = (vote_cv_preds == y_train).mean()
print(f"  Voting OOF:  {vote_oof_acc:.4f}")

# =================================================================
# 8. AGREEMENT-BASED FINAL PREDICTIONS
# =================================================================
# Where stacking and voting agree -> use that
# Where they disagree -> use voting (simpler, more robust)
agree = stack_preds == vote_preds
disagree_count = (~agree).sum()
print(f"\n  Stack vs Vote agreement: {agree.sum()}/{len(agree)} ({disagree_count} differ)")

# Use voting as final (simpler = generalizes better on small test set)
final_preds = vote_preds

# =================================================================
# 9. THRESHOLD TUNING
# =================================================================
# Find best threshold on OOF predictions
best_thr = 0.5
best_acc = vote_oof_acc
for thr in np.arange(0.40, 0.60, 0.01):
    acc = ((vote_train_proba >= thr).astype(int) == y_train).mean()
    if acc > best_acc:
        best_acc = acc
        best_thr = thr

print(f"  Best threshold: {best_thr:.2f} (OOF acc: {best_acc:.4f})")
final_preds = (vote_proba >= best_thr).astype(int)

# =================================================================
# 10. SAVE
# =================================================================
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"  Stacking CV:  {meta_cv.mean():.4f}")
print(f"  Voting OOF:   {vote_oof_acc:.4f}")
print(f"  Threshold:    {best_thr:.2f}")

submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": final_preds.astype(int),
})
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(submission["Survived"].value_counts())
