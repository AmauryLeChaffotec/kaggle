import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_len = len(train)
passenger_ids = test["PassengerId"]
df = pd.concat([train, test], sort=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

# --- Title ---
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(
    ["Lady", "Countess", "Dona", "Mme"], "Mrs"
)
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace(
    ["Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer"], "Rare"
)

# --- Family features ---
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FamilySizeBin"] = df["FamilySize"].map(
    lambda x: 0 if x == 1 else (1 if x <= 4 else 2)
)

# --- Ticket group size (people sharing same ticket) ---
ticket_counts = df["Ticket"].value_counts()
df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)
df["TicketGroupBin"] = df["TicketGroupSize"].map(
    lambda x: 0 if x == 1 else (1 if x <= 4 else 2)
)

# --- Cabin deck ---
df["Deck"] = df["Cabin"].str[0]
df["HasCabin"] = df["Cabin"].notna().astype(int)
deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
df["Deck"] = df["Deck"].map(deck_map).fillna(0).astype(int)

# --- Fill missing Age using median per (Title, Pclass) ---
df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
# Fallback for any remaining NaN
df["Age"] = df["Age"].fillna(df["Age"].median())

# --- Fill Embarked & Fare ---
df["Embarked"] = df["Embarked"].fillna("S")
df["Fare"] = df.groupby(["Pclass", "Embarked"])["Fare"].transform(
    lambda x: x.fillna(x.median())
)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# --- Fare per person ---
df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

# --- Age-based features ---
df["AgeBin"] = pd.cut(df["Age"], bins=[0, 5, 12, 18, 35, 60, 120],
                      labels=[0, 1, 2, 3, 4, 5]).astype(int)
df["IsChild"] = (df["Age"] <= 12).astype(int)

# --- Fare bins ---
df["FareBin"] = pd.qcut(df["Fare"], 5, labels=False, duplicates="drop")

# --- Sex * Pclass interaction ---
df["Sex_Pclass"] = df["Sex"].map({"male": 0, "female": 1}) * 3 + df["Pclass"]

# --- Name length ---
df["NameLen"] = df["Name"].str.len()

# --- Mother: female with Parch > 0, Mrs title, age > 18 ---
df["IsMother"] = (
    (df["Sex"] == "female") & (df["Parch"] > 0) &
    (df["Age"] > 18) & (df["Title"] == "Mrs")
).astype(int)

# ═══════════════════════════════════════════════════════════
# 3. FAMILY SURVIVAL RATE (leak-free for train, global for test)
# ═══════════════════════════════════════════════════════════
# Extract surname
df["Surname"] = df["Name"].str.split(",").str[0]

# Build family group key: Surname + FamilySize + Fare (approximate)
df["FamilyKey"] = df["Surname"] + "_" + df["FamilySize"].astype(str)

# Compute family survival rate from TRAIN only
train_df = df.iloc[:train_len].copy()
family_surv = train_df.groupby("FamilyKey")["Survived"].agg(["mean", "count"])
family_surv.columns = ["FamilySurvRate", "FamilyCount"]
# Only use families with >1 member
family_surv.loc[family_surv["FamilyCount"] <= 1, "FamilySurvRate"] = 0.5

df = df.merge(family_surv[["FamilySurvRate"]], left_on="FamilyKey",
              right_index=True, how="left")
df["FamilySurvRate"] = df["FamilySurvRate"].fillna(0.5)

# Ticket group survival rate
ticket_surv = train_df.groupby("Ticket")["Survived"].agg(["mean", "count"])
ticket_surv.columns = ["TicketSurvRate", "TicketCount"]
ticket_surv.loc[ticket_surv["TicketCount"] <= 1, "TicketSurvRate"] = 0.5

df = df.merge(ticket_surv[["TicketSurvRate"]], left_on="Ticket",
              right_index=True, how="left")
df["TicketSurvRate"] = df["TicketSurvRate"].fillna(0.5)

# ═══════════════════════════════════════════════════════════
# 4. ENCODE CATEGORICALS
# ═══════════════════════════════════════════════════════════
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
embarked_map = {"S": 0, "C": 1, "Q": 2}
df["Embarked"] = df["Embarked"].map(embarked_map)
title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
df["Title"] = df["Title"].map(title_map)

# ═══════════════════════════════════════════════════════════
# 5. SELECT FEATURES
# ═══════════════════════════════════════════════════════════
features = [
    "Pclass", "Sex", "AgeBin", "FareBin", "Title",
    "FamilySize", "IsAlone", "FamilySizeBin",
    "SibSp", "Parch", "Embarked",
    "HasCabin", "Deck",
    "TicketGroupSize", "TicketGroupBin",
    "FarePerPerson", "IsChild", "Sex_Pclass",
    "NameLen", "IsMother",
    "FamilySurvRate", "TicketSurvRate",
]

X_train = df[features].iloc[:train_len]
y_train = df["Survived"].iloc[:train_len].astype(int)
X_test = df[features].iloc[train_len:]

# Scale for SVC / LR / KNN
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════
# 6. DEFINE MODELS
# ═══════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    "XGB": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        min_child_weight=3, eval_metric="logloss",
        use_label_encoder=False, random_state=42,
    ),
    "LGBM": LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        min_child_weight=3, random_state=42, verbose=-1,
    ),
    "GBC": GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_split=10, min_samples_leaf=5,
        random_state=42,
    ),
    "RF": RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_split=8,
        min_samples_leaf=4, random_state=42,
    ),
    "ET": ExtraTreesClassifier(
        n_estimators=500, max_depth=6, min_samples_split=8,
        min_samples_leaf=4, random_state=42,
    ),
    "SVC": SVC(
        C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42,
    ),
    "LR": LogisticRegression(
        C=1.0, max_iter=1000, random_state=42,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=11),
}

# ═══════════════════════════════════════════════════════════
# 7. EVALUATE EACH MODEL
# ═══════════════════════════════════════════════════════════
print("=" * 50)
print("Individual model CV scores (10-fold):")
print("=" * 50)

for name, model in models.items():
    if name in ("SVC", "LR", "KNN"):
        scores = cross_val_score(model, X_train_sc, y_train, cv=skf, scoring="accuracy")
    else:
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    print(f"  {name:6s}  {scores.mean():.4f} (+/- {scores.std():.4f})")

# ═══════════════════════════════════════════════════════════
# 8. STACKING: Level-1 predictions via CV
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("Building stacking ensemble...")
print("=" * 50)

n_models = len(models)
S_train = np.zeros((train_len, n_models))
S_test = np.zeros((len(X_test), n_models))

for i, (name, model) in enumerate(models.items()):
    S_test_fold = np.zeros((len(X_test), 10))

    for j, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        if name in ("SVC", "LR", "KNN"):
            X_tr, X_val = X_train_sc[tr_idx], X_train_sc[val_idx]
        else:
            X_tr = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]

        y_tr = y_train.iloc[tr_idx]
        model.fit(X_tr, y_tr)

        S_train[val_idx, i] = model.predict_proba(X_val)[:, 1]

        if name in ("SVC", "LR", "KNN"):
            S_test_fold[:, j] = model.predict_proba(X_test_sc)[:, 1]
        else:
            S_test_fold[:, j] = model.predict_proba(X_test)[:, 1]

    S_test[:, i] = S_test_fold.mean(axis=1)
    print(f"  {name} done")

# Level-2 meta-learner
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_scores = cross_val_score(meta, S_train, y_train, cv=skf, scoring="accuracy")
print(f"\n  Stacking meta CV: {meta_scores.mean():.4f} (+/- {meta_scores.std():.4f})")

meta.fit(S_train, y_train)
stack_preds = meta.predict(S_test)

# ═══════════════════════════════════════════════════════════
# 9. SOFT VOTING ENSEMBLE (top models)
# ═══════════════════════════════════════════════════════════
# Also try a simpler soft-voting approach with the best models
voting = VotingClassifier(
    estimators=[
        ("xgb", models["XGB"]),
        ("lgbm", models["LGBM"]),
        ("gbc", models["GBC"]),
        ("svc", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)),
    ],
    voting="soft",
    weights=[2, 2, 2, 1],
)
# For voting, we need scaled data for SVC inside pipeline - use unscaled and let it handle
# Actually, since SVC needs scaled, let's just average probabilities manually

# Manual soft voting with proper scaling
print("\n" + "=" * 50)
print("Manual soft voting...")
print("=" * 50)

proba_test = np.zeros(len(X_test))
proba_train = np.zeros(train_len)
weights = {"XGB": 2, "LGBM": 2, "GBC": 2, "RF": 1, "ET": 1, "SVC": 1.5, "LR": 1}
total_w = sum(weights.values())

for name, w in weights.items():
    model = models[name]
    if name in ("SVC", "LR", "KNN"):
        model.fit(X_train_sc, y_train)
        proba_test += w * model.predict_proba(X_test_sc)[:, 1]
        proba_train += w * model.predict_proba(X_train_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba_test += w * model.predict_proba(X_test)[:, 1]
        proba_train += w * model.predict_proba(X_train)[:, 1]

proba_test /= total_w
proba_train /= total_w

vote_preds = (proba_test >= 0.5).astype(int)
vote_train_acc = ((proba_train >= 0.5).astype(int) == y_train).mean()
print(f"  Voting train accuracy: {vote_train_acc:.4f}")

# Cross-val for voting estimate
from sklearn.base import BaseEstimator, ClassifierMixin

class ManualVoter(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, weights, scale_models):
        self.base_models = base_models
        self.weights = weights
        self.scale_models = scale_models
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.X_sc = self.scaler.fit_transform(X)
        for name, m in self.base_models.items():
            if name in self.scale_models:
                m.fit(self.X_sc, y)
            else:
                m.fit(X, y)
        return self

    def predict(self, X):
        X_sc = self.scaler.transform(X)
        proba = np.zeros(len(X))
        total = sum(self.weights.values())
        for name, w in self.weights.items():
            m = self.base_models[name]
            if name in self.scale_models:
                proba += w * m.predict_proba(X_sc)[:, 1]
            else:
                proba += w * m.predict_proba(X)[:, 1]
        proba /= total
        return (proba >= 0.5).astype(int)

# Fresh models for CV
def fresh_models():
    return {
        "XGB": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
            min_child_weight=3, eval_metric="logloss", use_label_encoder=False, random_state=42),
        "LGBM": LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
            min_child_weight=3, random_state=42, verbose=-1),
        "GBC": GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_split=10, min_samples_leaf=5, random_state=42),
        "RF": RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_split=8,
            min_samples_leaf=4, random_state=42),
        "ET": ExtraTreesClassifier(n_estimators=500, max_depth=6, min_samples_split=8,
            min_samples_leaf=4, random_state=42),
        "SVC": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42),
        "LR": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    }

voter = ManualVoter(fresh_models(), weights, {"SVC", "LR", "KNN"})
voter_cv = cross_val_score(voter, X_train.values, y_train, cv=skf, scoring="accuracy")
print(f"  Voting CV accuracy: {voter_cv.mean():.4f} (+/- {voter_cv.std():.4f})")

# ═══════════════════════════════════════════════════════════
# 10. CHOOSE BEST & SAVE
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Stacking CV: {meta_scores.mean():.4f}")
print(f"  Voting  CV:  {voter_cv.mean():.4f}")

# Use the one with higher CV
if meta_scores.mean() >= voter_cv.mean():
    final_preds = stack_preds
    print("  -> Using STACKING predictions")
else:
    final_preds = vote_preds
    print("  -> Using VOTING predictions")

submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": final_preds.astype(int),
})
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(submission["Survived"].value_counts())
