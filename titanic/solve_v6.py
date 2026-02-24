"""
Titanic v6 - Leak-safe group features + OOF ensemble.

Goals:
  - Keep useful Titanic group signals (family/ticket survival patterns)
  - Avoid blatant target leakage in validation by computing group stats per fold
  - Produce multiple submissions (stack / vote / top3 / blend) with honest OOF metrics

Usage:
  python solve_v6.py
  python solve_v6.py --n-splits 5 --seed 123 --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Faster settings (5 folds, fewer trees)")
    parser.add_argument(
        "--seeds-bagging",
        type=int,
        default=1,
        help="Average XGB/LGBM predictions across N random seeds per fold (>=1)",
    )
    return parser.parse_args()


def safe_qcut(values: pd.Series, q: int) -> pd.Series:
    out = pd.qcut(values, q=q, labels=False, duplicates="drop")
    if out.isna().any():
        fill_value = int(out.dropna().median()) if out.notna().any() else 0
        out = out.fillna(fill_value)
    return out.astype(int)


def build_base_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_len = len(train_df)
    df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

    # Raw keys for fold-safe target/group features later.
    keys = pd.DataFrame(index=df.index)
    keys["Ticket"] = df["Ticket"].fillna("MISSING").astype(str)
    keys["Surname"] = df["Name"].str.split(",").str[0].fillna("UNKNOWN").str.strip()
    keys["Sex"] = df["Sex"].fillna("unknown").astype(str)

    # Core engineered features (target-free)
    feat = pd.DataFrame(index=df.index)
    feat["Pclass"] = df["Pclass"].astype(int)
    feat["SibSp"] = df["SibSp"].fillna(0).astype(int)
    feat["Parch"] = df["Parch"].fillna(0).astype(int)

    title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Rare")
    title = title.replace(["Mlle", "Ms"], "Miss")
    title = title.replace(["Mme"], "Mrs")
    title = title.replace(["Lady", "Countess", "Dona"], "Mrs")
    title = title.replace(["Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer"], "Rare")

    feat["FamilySize"] = feat["SibSp"] + feat["Parch"] + 1
    feat["IsAlone"] = (feat["FamilySize"] == 1).astype(int)
    feat["FamilyBin"] = feat["FamilySize"].map(lambda x: 0 if x == 1 else (1 if x <= 4 else 2)).astype(int)

    ticket_counts = keys["Ticket"].value_counts()
    feat["TicketGroupSize"] = keys["Ticket"].map(ticket_counts).astype(int)

    feat["HasCabin"] = df["Cabin"].notna().astype(int)
    feat["Deck"] = df["Cabin"].str[0].fillna("U")

    embarked = df["Embarked"].fillna("S")
    fare = df["Fare"].copy()
    fare = fare.fillna(pd.DataFrame({"Pclass": df["Pclass"], "Embarked": embarked, "Fare": fare})
                       .groupby(["Pclass", "Embarked"])["Fare"]
                       .transform("median"))
    fare = fare.fillna(fare.median())

    age = df["Age"].copy()
    age = age.fillna(pd.DataFrame({"Title": title, "Pclass": df["Pclass"], "Age": age})
                     .groupby(["Title", "Pclass"])["Age"]
                     .transform("median"))
    age = age.fillna(age.median())

    feat["Age"] = age
    feat["Fare"] = fare
    feat["FarePerPerson"] = feat["Fare"] / feat["TicketGroupSize"].clip(lower=1)

    feat["AgeBin"] = pd.cut(feat["Age"], bins=[0, 5, 12, 18, 30, 50, 120], labels=False, include_lowest=True)
    feat["AgeBin"] = feat["AgeBin"].fillna(0).astype(int)
    feat["IsChild"] = (feat["Age"] <= 12).astype(int)

    feat["FareBin"] = safe_qcut(feat["FarePerPerson"], q=5)

    feat["NameLen"] = df["Name"].fillna("").str.len().astype(int)
    feat["NameWords"] = df["Name"].fillna("").str.split().str.len().astype(int)

    feat["TicketPrefix"] = keys["Ticket"].str.extract(r"^([A-Za-z./]+)", expand=False).fillna("NONE")
    tp_counts = feat["TicketPrefix"].value_counts()
    feat["TicketPrefix"] = feat["TicketPrefix"].map(lambda x: x if tp_counts.get(x, 0) >= 10 else "RARE")

    feat["TicketNumber"] = keys["Ticket"].str.extract(r"(\d+)$", expand=False).fillna("NONE")
    tn_counts = feat["TicketNumber"].value_counts()
    feat["TicketNumber"] = feat["TicketNumber"].map(lambda x: x if tn_counts.get(x, 0) >= 5 else "RARE")

    surname_counts = keys["Surname"].value_counts()
    feat["SurnameFreq"] = keys["Surname"].map(surname_counts).astype(int)

    feat["Sex"] = keys["Sex"].map({"male": 0, "female": 1}).fillna(0).astype(int)
    feat["Embarked"] = embarked.map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(int)
    feat["Sex_Pclass"] = feat["Sex"] * 10 + feat["Pclass"]

    keys["FamilySize"] = feat["FamilySize"]
    keys["FamilyKey"] = keys["Surname"] + "_" + keys["FamilySize"].astype(str)
    keys["SexCode"] = feat["Sex"]

    # Encode categoricals with shared train+test vocabulary (target-free).
    cat_cols = ["Deck", "TicketPrefix", "TicketNumber"]
    for col in cat_cols:
        feat[col] = pd.Categorical(feat[col]).codes.astype(int)

    # Title encode after mapping (shared vocabulary)
    feat["Title"] = pd.Categorical(title).codes.astype(int)

    base_features = [
        "Pclass", "Sex", "Title", "Age", "AgeBin", "Fare", "FarePerPerson", "FareBin",
        "FamilySize", "FamilyBin", "IsAlone", "SibSp", "Parch",
        "Embarked", "HasCabin", "Deck",
        "TicketGroupSize", "TicketPrefix", "TicketNumber",
        "IsChild", "Sex_Pclass", "NameLen", "NameWords", "SurnameFreq",
    ]

    feat = feat[base_features].copy()
    assert len(feat.iloc[:train_len]) == train_len
    return feat, keys, base_features


def compute_group_maps(train_keys: pd.DataFrame, y_fold: pd.Series) -> tuple[dict, dict, dict, dict]:
    tmp = train_keys.copy()
    tmp["Survived"] = y_fold.values

    fam = tmp.groupby("FamilyKey")["Survived"].agg(["mean", "count"])
    fam_mean = fam["mean"].copy()
    fam_mean.loc[fam["count"] <= 1] = 0.5

    tick = tmp.groupby("Ticket")["Survived"].agg(["mean", "count"])
    tick_mean = tick["mean"].copy()
    tick_mean.loc[tick["count"] <= 1] = 0.5

    return (
        fam_mean.to_dict(),
        fam["count"].to_dict(),
        tick_mean.to_dict(),
        tick["count"].to_dict(),
    )


def map_group_features(keys_slice: pd.DataFrame, group_maps: tuple[dict, dict, dict, dict]) -> pd.DataFrame:
    fam_mean_map, fam_count_map, tick_mean_map, tick_count_map = group_maps
    out = pd.DataFrame(index=keys_slice.index)

    out["FamSurvMean"] = keys_slice["FamilyKey"].map(fam_mean_map).fillna(0.5).astype(float)
    out["FamKnown"] = (keys_slice["FamilyKey"].map(fam_count_map).fillna(0).astype(int) > 1).astype(int)

    out["TickSurvMean"] = keys_slice["Ticket"].map(tick_mean_map).fillna(0.5).astype(float)
    out["TickKnown"] = (keys_slice["Ticket"].map(tick_count_map).fillna(0).astype(int) > 1).astype(int)

    out["GroupSurvival"] = out[["FamSurvMean", "TickSurvMean"]].max(axis=1)
    out["GroupSurvHigh"] = (out["GroupSurvival"] >= 0.75).astype(int)
    out["GroupSurvLow"] = (out["GroupSurvival"] <= 0.25).astype(int)
    out["FemaleInDyingGroup"] = ((keys_slice["SexCode"] == 1) & (out["GroupSurvLow"] == 1)).astype(int)
    out["MaleInSurvGroup"] = ((keys_slice["SexCode"] == 0) & (out["GroupSurvHigh"] == 1)).astype(int)

    return out


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_thr, best_acc = 0.5, accuracy_score(y_true, (proba >= 0.5).astype(int))
    for thr in np.arange(0.35, 0.66, 0.01):
        acc = accuracy_score(y_true, (proba >= thr).astype(int))
        if acc > best_acc:
            best_thr, best_acc = float(round(thr, 2)), float(acc)
    return best_thr, best_acc


def make_models(seed: int, quick: bool) -> tuple[dict[str, object], set[str]]:
    n_estimators = 220 if quick else 420
    xgb_trees = 180 if quick else 360
    lgbm_trees = 220 if quick else 420
    rf_trees = 300 if quick else 700

    models = {
        "XGB": XGBClassifier(
            n_estimators=xgb_trees,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=0.5,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        ),
        "LGBM": LGBMClassifier(
            n_estimators=lgbm_trees,
            max_depth=4,
            num_leaves=24,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        ),
        "GBC": GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=seed,
        ),
        "RF": RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "LR": LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    }
    scale_models = {"LR"}
    return models, scale_models


def save_submission(path: Path, passenger_ids: pd.Series, preds: np.ndarray) -> None:
    sub = pd.DataFrame({"PassengerId": passenger_ids, "Survived": preds.astype(int)})
    sub.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.quick and args.n_splits == 10:
        args.n_splits = 5
    args.seeds_bagging = max(1, int(args.seeds_bagging))

    data_dir = args.data_dir
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test CSV in {data_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    passenger_ids = test_df["PassengerId"].copy()
    y_train = train_df["Survived"].astype(int).to_numpy()
    train_len = len(train_df)
    test_len = len(test_df)

    base_feat_all, keys_all, base_features = build_base_features(train_df, test_df)
    X_base_train = base_feat_all.iloc[:train_len].reset_index(drop=True)
    X_base_test = base_feat_all.iloc[train_len:].reset_index(drop=True)
    keys_train = keys_all.iloc[:train_len].reset_index(drop=True)
    keys_test = keys_all.iloc[train_len:].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(skf.split(X_base_train, y_train))

    # Fixed fold-safe group feature schema.
    sample_maps = compute_group_maps(keys_train.iloc[splits[0][0]], pd.Series(y_train[splits[0][0]]))
    group_feature_names = list(map_group_features(keys_train.iloc[splits[0][1]], sample_maps).columns)
    feature_names = base_features + group_feature_names

    models, scale_models = make_models(args.seed, args.quick)
    model_names = list(models.keys())
    n_models = len(model_names)

    oof = np.zeros((train_len, n_models), dtype=float)
    test_fold_preds = np.zeros((test_len, n_models, args.n_splits), dtype=float)

    print("=" * 60)
    print("V6 OOF ENSEMBLE (LEAK-SAFE GROUP FEATURES)")
    print("=" * 60)
    print(f"Train rows: {train_len} | Test rows: {test_len} | Folds: {args.n_splits}")
    print(f"Base features: {len(base_features)} | Group features: {len(group_feature_names)}")
    print(f"Seed bagging for XGB/LGBM: {args.seeds_bagging}")

    for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
        y_tr = y_train[tr_idx]

        fold_maps = compute_group_maps(keys_train.iloc[tr_idx], pd.Series(y_tr))
        g_tr = map_group_features(keys_train.iloc[tr_idx], fold_maps).reset_index(drop=True)
        g_val = map_group_features(keys_train.iloc[val_idx], fold_maps).reset_index(drop=True)
        g_test = map_group_features(keys_test, fold_maps).reset_index(drop=True)

        xtr = pd.concat([X_base_train.iloc[tr_idx].reset_index(drop=True), g_tr], axis=1)[feature_names]
        xval = pd.concat([X_base_train.iloc[val_idx].reset_index(drop=True), g_val], axis=1)[feature_names]
        xtest = pd.concat([X_base_test.reset_index(drop=True), g_test], axis=1)[feature_names]

        print(f"\nFold {fold_idx}/{args.n_splits} - train={len(tr_idx)} val={len(val_idx)}")
        for m_idx, name in enumerate(model_names):
            model = clone(models[name])

            if name in scale_models:
                scaler = StandardScaler()
                xtr_use = scaler.fit_transform(xtr)
                xval_use = scaler.transform(xval)
                xtest_use = scaler.transform(xtest)
            else:
                xtr_use, xval_use, xtest_use = xtr, xval, xtest

            # Optional seed bagging for high-variance boosted trees.
            if args.seeds_bagging > 1 and name in {"XGB", "LGBM"}:
                val_bag = np.zeros(len(val_idx), dtype=float)
                test_bag = np.zeros(test_len, dtype=float)
                for bag_seed_offset in range(args.seeds_bagging):
                    bag_model = clone(model)
                    bag_model.set_params(random_state=args.seed + 1000 * fold_idx + bag_seed_offset)
                    bag_model.fit(xtr_use, y_tr)
                    val_bag += bag_model.predict_proba(xval_use)[:, 1]
                    test_bag += bag_model.predict_proba(xtest_use)[:, 1]
                val_proba = val_bag / args.seeds_bagging
                test_proba = test_bag / args.seeds_bagging
            else:
                model.fit(xtr_use, y_tr)
                val_proba = model.predict_proba(xval_use)[:, 1]
                test_proba = model.predict_proba(xtest_use)[:, 1]

            oof[val_idx, m_idx] = val_proba
            test_fold_preds[:, m_idx, fold_idx - 1] = test_proba

            fold_acc = accuracy_score(y_train[val_idx], (val_proba >= 0.5).astype(int))
            print(f"  {name:4s} fold acc={fold_acc:.4f}")

    S_test = test_fold_preds.mean(axis=2)

    print("\n" + "=" * 60)
    print("BASE MODEL OOF SCORES")
    print("=" * 60)
    cv_scores: dict[str, float] = {}
    for idx, name in enumerate(model_names):
        acc = accuracy_score(y_train, (oof[:, idx] >= 0.5).astype(int))
        cv_scores[name] = float(acc)
        print(f"  {name:4s} OOF acc={acc:.4f}")

    # Strategy 1: soft vote (all models)
    vote_proba_train = oof.mean(axis=1)
    vote_proba_test = S_test.mean(axis=1)
    vote_thr, vote_oof = best_threshold(y_train, vote_proba_train)
    vote_preds = (vote_proba_test >= vote_thr).astype(int)

    # Strategy 2: top-3 weighted by OOF accuracy
    top3 = sorted(model_names, key=lambda n: cv_scores[n], reverse=True)[:3]
    top3_idx = [model_names.index(n) for n in top3]
    raw_w = np.array([cv_scores[n] for n in top3], dtype=float)
    top3_weights = raw_w / raw_w.sum()
    top3_proba_train = (oof[:, top3_idx] * top3_weights).sum(axis=1)
    top3_proba_test = (S_test[:, top3_idx] * top3_weights).sum(axis=1)
    top3_thr, top3_oof = best_threshold(y_train, top3_proba_train)
    top3_preds = (top3_proba_test >= top3_thr).astype(int)

    # Strategy 3: stacking with proper meta OOF
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    stack_oof = np.zeros(train_len, dtype=float)
    for tr_idx, val_idx in splits:
        meta_fold = clone(meta)
        meta_fold.fit(oof[tr_idx], y_train[tr_idx])
        stack_oof[val_idx] = meta_fold.predict_proba(oof[val_idx])[:, 1]
    meta.fit(oof, y_train)
    stack_proba_test = meta.predict_proba(S_test)[:, 1]
    stack_thr, stack_oof_acc = best_threshold(y_train, stack_oof)
    stack_preds = (stack_proba_test >= stack_thr).astype(int)

    # Strategy 4: blend stack + vote (weighted by OOF)
    blend_w_stack = max(stack_oof_acc, 1e-6)
    blend_w_vote = max(vote_oof, 1e-6)
    blend_sum = blend_w_stack + blend_w_vote
    blend_proba_train = (blend_w_stack * stack_oof + blend_w_vote * vote_proba_train) / blend_sum
    blend_proba_test = (blend_w_stack * stack_proba_test + blend_w_vote * vote_proba_test) / blend_sum
    blend_thr, blend_oof = best_threshold(y_train, blend_proba_train)
    blend_preds = (blend_proba_test >= blend_thr).astype(int)

    # Strategy 5: majority vote across best 3 strategies
    majority_preds = ((stack_preds + vote_preds + top3_preds) >= 2).astype(int)
    majority_oof_majority = (
        ((stack_oof >= stack_thr).astype(int) + (vote_proba_train >= vote_thr).astype(int) +
         (top3_proba_train >= top3_thr).astype(int)) >= 2
    ).astype(int)
    majority_oof = float(accuracy_score(y_train, majority_oof_majority))

    print("\n" + "=" * 60)
    print("ENSEMBLE STRATEGIES (HONEST OOF)")
    print("=" * 60)
    print(f"  Stacking        OOF={stack_oof_acc:.4f}  thr={stack_thr:.2f}  survived={int(stack_preds.sum())}")
    print(f"  Voting(all)     OOF={vote_oof:.4f}  thr={vote_thr:.2f}  survived={int(vote_preds.sum())}")
    print(f"  Top3({','.join(top3)}) OOF={top3_oof:.4f}  thr={top3_thr:.2f}  survived={int(top3_preds.sum())}")
    print(f"  Blend(stack+vote) OOF={blend_oof:.4f}  thr={blend_thr:.2f}  survived={int(blend_preds.sum())}")
    print(f"  Majority(3x)    OOF={majority_oof:.4f}  thr=n/a   survived={int(majority_preds.sum())}")

    print("\nAgreement on test set:")
    print(f"  stack vs vote : {(stack_preds == vote_preds).sum()}/{test_len}")
    print(f"  stack vs top3 : {(stack_preds == top3_preds).sum()}/{test_len}")
    print(f"  vote  vs top3 : {(vote_preds == top3_preds).sum()}/{test_len}")

    outputs = {
        "submission_stack.csv": (stack_oof_acc, stack_preds),
        "submission_vote.csv": (vote_oof, vote_preds),
        "submission_top3.csv": (top3_oof, top3_preds),
        "submission_blend.csv": (blend_oof, blend_preds),
        "submission_majority.csv": (majority_oof, majority_preds),
    }

    for name, (_, preds) in outputs.items():
        save_submission(data_dir / name, passenger_ids, preds)

    best_name = max(outputs, key=lambda k: outputs[k][0])
    save_submission(data_dir / "submission.csv", passenger_ids, outputs[best_name][1])

    print("\nSaved:")
    for name, (score, _) in sorted(outputs.items(), key=lambda kv: kv[1][0], reverse=True):
        print(f"  {name:22s} OOF={score:.4f}")
    print(f"  submission.csv -> {best_name}")


if __name__ == "__main__":
    main()
