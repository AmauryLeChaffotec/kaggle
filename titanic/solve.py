import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# ── Load data ──────────────────────────────────────────────
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

passenger_ids = test["PassengerId"]
combine = pd.concat([train, test], sort=False).reset_index(drop=True)

# ── Feature engineering ────────────────────────────────────

# Title from Name
combine["Title"] = combine["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
combine["Title"] = combine["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
     "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
)
combine["Title"] = combine["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Family size & alone
combine["FamilySize"] = combine["SibSp"] + combine["Parch"] + 1
combine["IsAlone"] = (combine["FamilySize"] == 1).astype(int)

# Fill missing Age with median per Title
combine["Age"] = combine.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Fill missing Embarked & Fare
combine["Embarked"] = combine["Embarked"].fillna(combine["Embarked"].mode()[0])
combine["Fare"] = combine["Fare"].fillna(combine["Fare"].median())

# Age bins
combine["AgeBin"] = pd.cut(combine["Age"], bins=[0, 12, 20, 40, 60, 120],
                           labels=[0, 1, 2, 3, 4]).astype(int)

# Fare bins
combine["FareBin"] = pd.qcut(combine["Fare"], 4, labels=[0, 1, 2, 3]).astype(int)

# Encode categoricals
combine["Sex"] = combine["Sex"].map({"male": 0, "female": 1})
combine["Embarked"] = combine["Embarked"].map({"S": 0, "C": 1, "Q": 2})
title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
combine["Title"] = combine["Title"].map(title_map)

# ── Select features ────────────────────────────────────────
features = ["Pclass", "Sex", "AgeBin", "FareBin", "Title",
            "FamilySize", "IsAlone", "SibSp", "Parch", "Embarked"]

train_len = len(train)
X_train = combine[features].iloc[:train_len]
y_train = combine["Survived"].iloc[:train_len].astype(int)
X_test = combine[features].iloc[train_len:]

# ── Train model ────────────────────────────────────────────
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42,
)

# Cross-validation score
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Fit on full training set
model.fit(X_train, y_train)

# Feature importances
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:15s} {imp:.4f}")

# ── Predict & save submission ──────────────────────────────
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions.astype(int),
})
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(submission.head(10))
