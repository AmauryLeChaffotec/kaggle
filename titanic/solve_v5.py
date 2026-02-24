"""
Titanic v5 - Sweet spot: good features, NO leaky survival rates,
moderate regularization. Target: honest 85-86% CV -> 0.80+ LB
"""
import pandas as pd
import numpy as np
import warnings
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace(["Mme"], "Mrs")
df["Title"] = df["Title"].replace(["Lady", "Countess", "Dona"], "Mrs")
df["Title"] = df["Title"].replace(
    ["Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer"], "Rare"
)
title_enc = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
df["Title"] = df["Title"].map(title_enc)

# --- Sex ---
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# --- Family ---
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FamilyBin"] = df["FamilySize"].map(
    lambda x: 0 if x == 1 else (1 if x <= 4 else 2)
)

# --- Ticket group ---
ticket_counts = df["Ticket"].value_counts()
df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)

# --- Cabin / Deck ---
df["HasCabin"] = df["Cabin"].notna().astype(int)
df["Deck"] = df["Cabin"].str[0].fillna("U")
deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "U": 0}
df["Deck"] = df["Deck"].map(deck_map)

# --- Age ---
df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
df["Age"] = df["Age"].fillna(df["Age"].median())

# --- Embarked & Fare ---
df["Embarked"] = df["Embarked"].fillna("S")
df["Fare"] = df.groupby(["Pclass", "Embarked"])["Fare"].transform(
    lambda x: x.fillna(x.median())
)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# --- Fare per person ---
df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

# --- Age bins ---
df["AgeBin"] = pd.cut(
    df["Age"], bins=[0, 5, 12, 18, 30, 50, 120], labels=[0, 1, 2, 3, 4, 5]
).astype(int)
df["IsChild"] = (df["Age"] <= 12).astype(int)

# --- Fare bins ---
df["FareBin"] = pd.qcut(df["FarePerPerson"], 5, labels=False, duplicates="drop")

# --- Interactions ---
df["Sex_Pclass"] = df["Sex"] * 10 + df["Pclass"]

# --- Ticket prefix ---
df["TicketPrefix"] = df["Ticket"].str.extract(r"^([A-Za-z./]+)", expand=False)
df["TicketPrefix"] = df["TicketPrefix"].fillna("NONE")
tp_counts = df["TicketPrefix"].value_counts()
df["TicketPrefix"] = df["TicketPrefix"].map(
    lambda x: x if tp_counts.get(x, 0) >= 10 else "RARE"
)
tp_enc = {v: i for i, v in enumerate(df["TicketPrefix"].unique())}
df["TicketPrefix"] = df["TicketPrefix"].map(tp_enc)

# --- Name length ---
df["NameLen"] = df["Name"].apply(lambda x: len(x) if isinstance(x, str) else 0)

# =================================================================
# 3. FEATURES
# =================================================================
features = [
    "Pclass", "Sex", "Title", "Age", "AgeBin", "FareBin",
    "Fare", "FarePerPerson",
    "FamilySize", "FamilyBin", "IsAlone",
    "SibSp", "Parch",
    "Embarked", "HasCabin", "Deck",
    "TicketGroupSize", "TicketPrefix",
    "IsChild", "Sex_Pclass", "NameLen",
]

X_train = df[features].iloc[:train_len]
y_train = df["Survived"].iloc[:train_len].astype(int)
X_test = df[features].iloc[train_len:]

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=features)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# =================================================================
# 4. OPTUNA TUNING
# =================================================================
print("=" * 55)
print("OPTUNA TUNING (80 trials)")
print("=" * 55)

N_TRIALS = 80


def obj_xgb(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample", 0.5, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "min_child_weight": trial.suggest_int("mcw", 1, 12),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "eval_metric": "logloss",
        "random_state": SEED,
    }
    return cross_val_score(
        XGBClassifier(**p), X_train, y_train, cv=skf, scoring="accuracy"
    ).mean()


def obj_lgbm(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample", 0.5, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "min_child_weight": trial.suggest_int("mcw", 1, 12),
        "num_leaves": trial.suggest_int("num_leaves", 8, 50),
        "verbose": -1,
        "random_state": SEED,
    }
    return cross_val_score(
        LGBMClassifier(**p), X_train, y_train, cv=skf, scoring="accuracy"
    ).mean()


def obj_gbc(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "min_samples_split": trial.suggest_int("mss", 2, 25),
        "min_samples_leaf": trial.suggest_int("msl", 1, 12),
        "random_state": SEED,
    }
    return cross_val_score(
        GradientBoostingClassifier(**p), X_train, y_train, cv=skf, scoring="accuracy"
    ).mean()


def obj_rf(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("mss", 2, 20),
        "min_samples_leaf": trial.suggest_int("msl", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": SEED,
    }
    return cross_val_score(
        RandomForestClassifier(**p), X_train, y_train, cv=skf, scoring="accuracy"
    ).mean()


def obj_svc(trial):
    p = {
        "C": trial.suggest_float("C", 0.1, 50.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": "rbf",
        "probability": True,
        "random_state": SEED,
    }
    return cross_val_score(
        SVC(**p), X_train_sc, y_train, cv=skf, scoring="accuracy"
    ).mean()


tuners = {
    "XGB": obj_xgb, "LGBM": obj_lgbm, "GBC": obj_gbc,
    "RF": obj_rf, "SVC": obj_svc,
}
best_params = {}
for name, obj in tuners.items():
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(obj, n_trials=N_TRIALS)
    best_params[name] = study.best_params
    print(f"  {name:6s}  best CV = {study.best_value:.4f}")

# =================================================================
# 5. BUILD MODELS
# =================================================================
bp = best_params

# Rename Optuna short param names to sklearn expected names
def fix_params(d, renames):
    out = dict(d)
    for old, new in renames.items():
        if old in out:
            out[new] = out.pop(old)
    return out

bp["XGB"] = fix_params(bp["XGB"], {"lr": "learning_rate", "mcw": "min_child_weight", "colsample": "colsample_bytree"})
bp["LGBM"] = fix_params(bp["LGBM"], {"lr": "learning_rate", "mcw": "min_child_weight", "colsample": "colsample_bytree"})
bp["GBC"] = fix_params(bp["GBC"], {"lr": "learning_rate", "mss": "min_samples_split", "msl": "min_samples_leaf"})
bp["RF"] = fix_params(bp["RF"], {"mss": "min_samples_split", "msl": "min_samples_leaf"})

models = {
    "XGB": XGBClassifier(**bp["XGB"], eval_metric="logloss", random_state=SEED),
    "LGBM": LGBMClassifier(**bp["LGBM"], verbose=-1, random_state=SEED),
    "GBC": GradientBoostingClassifier(**bp["GBC"], random_state=SEED),
    "RF": RandomForestClassifier(**bp["RF"], random_state=SEED),
    "SVC": SVC(**bp["SVC"], kernel="rbf", probability=True, random_state=SEED),
    "LR": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
}
scale_models = {"SVC", "LR"}

print("\n" + "=" * 55)
print("TUNED MODEL CV SCORES")
print("=" * 55)
cv_scores = {}
for name, model in models.items():
    X = X_train_sc if name in scale_models else X_train
    s = cross_val_score(model, X, y_train, cv=skf, scoring="accuracy")
    cv_scores[name] = s.mean()
    print(f"  {name:6s}  CV = {s.mean():.4f} (+/- {s.std():.4f})")

# =================================================================
# 6. STACKING
# =================================================================
print("\n" + "=" * 55)
print("STACKING")
print("=" * 55)

n_models = len(models)
S_train = np.zeros((train_len, n_models))
S_test = np.zeros((len(X_test), n_models))

for i, (name, model) in enumerate(models.items()):
    S_test_folds = np.zeros((len(X_test), 10))
    for j, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        if name in scale_models:
            xtr, xval, xte = X_train_sc.iloc[tr_idx], X_train_sc.iloc[val_idx], X_test_sc
        else:
            xtr, xval, xte = X_train.iloc[tr_idx], X_train.iloc[val_idx], X_test
        ytr = y_train.iloc[tr_idx]
        model.fit(xtr, ytr)
        S_train[val_idx, i] = model.predict_proba(xval)[:, 1]
        S_test_folds[:, j] = model.predict_proba(xte)[:, 1]
    S_test[:, i] = S_test_folds.mean(axis=1)
    print(f"  {name} done")

# Meta-learner
meta = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
meta_cv = cross_val_score(meta, S_train, y_train, cv=skf, scoring="accuracy")
print(f"\n  Stacking CV: {meta_cv.mean():.4f} (+/- {meta_cv.std():.4f})")
meta.fit(S_train, y_train)

# =================================================================
# 7. GENERATE MULTIPLE SUBMISSIONS
# =================================================================
print("\n" + "=" * 55)
print("GENERATING SUBMISSIONS")
print("=" * 55)

# Stacking predictions
stack_proba = meta.predict_proba(S_test)[:, 1]
stack_preds = (stack_proba >= 0.5).astype(int)

# Simple average voting
vote_proba = S_test.mean(axis=1)
vote_preds = (vote_proba >= 0.5).astype(int)

# Top-3 weighted average (by CV score)
top3 = sorted(cv_scores, key=cv_scores.get, reverse=True)[:3]
top3_indices = [list(models.keys()).index(n) for n in top3]
top3_proba = S_test[:, top3_indices].mean(axis=1)
top3_preds = (top3_proba >= 0.5).astype(int)

# OOF accuracy for each
for label, train_proba, test_preds in [
    ("Stacking", meta.predict_proba(S_train)[:, 1], stack_preds),
    ("Voting", S_train.mean(axis=1), vote_preds),
    (f"Top3({','.join(top3)})", S_train[:, top3_indices].mean(axis=1), top3_preds),
]:
    oof_acc = ((train_proba >= 0.5).astype(int) == y_train).mean()
    print(f"  {label:30s}  OOF={oof_acc:.4f}  survived={test_preds.sum()}")

# Agreement analysis
print(f"\n  Stack vs Vote: {(stack_preds == vote_preds).sum()}/418 agree")
print(f"  Stack vs Top3: {(stack_preds == top3_preds).sum()}/418 agree")
print(f"  Vote  vs Top3: {(vote_preds == top3_preds).sum()}/418 agree")

# Majority vote of the 3 strategies
majority = ((stack_preds + vote_preds + top3_preds) >= 2).astype(int)
print(f"  Majority survived: {majority.sum()}")

# Save all variants
for label, preds in [
    ("submission_stack", stack_preds),
    ("submission_vote", vote_preds),
    ("submission_top3", top3_preds),
    ("submission_majority", majority),
]:
    sub = pd.DataFrame({"PassengerId": passenger_ids, "Survived": preds})
    sub.to_csv(f"{label}.csv", index=False)

# Main submission = stacking (usually best on LB)
submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": stack_preds})
submission.to_csv("submission.csv", index=False)

print(f"\nSaved 5 submission files. Try them all on Kaggle!")
print("  submission.csv       (stacking - default)")
print("  submission_stack.csv")
print("  submission_vote.csv")
print("  submission_top3.csv")
print("  submission_majority.csv")
