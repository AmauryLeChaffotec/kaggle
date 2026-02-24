"""
Titanic local CatBoost variant (Windows-friendly, no TF-DF required).

Key ideas:
  - Native categorical handling (CatBoost) on rich Titanic features
  - Fold-safe group survival features (family/ticket) to avoid CV leakage
  - Multi-seed bagging + multiple CatBoost parameter variants
  - Honest OOF ensemble metrics and multiple submissions

Examples:
  python solve_catboost_local.py --quick
  python solve_catboost_local.py --seeds-bagging 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Faster run: fewer folds / iterations")
    parser.add_argument("--seeds-bagging", type=int, default=3, help="Average each CatBoost config over N seeds")
    parser.add_argument("--threads", type=int, default=-1, help="CatBoost thread_count")
    return parser.parse_args()


def normalize_name(x: str) -> str:
    return " ".join([v.strip(",()[].\"'") for v in str(x).split(" ") if v])


def safe_qcut(values: pd.Series, q: int) -> pd.Series:
    out = pd.qcut(values, q=q, labels=False, duplicates="drop")
    if out.isna().any():
        fill_value = int(out.dropna().median()) if out.notna().any() else 0
        out = out.fillna(fill_value)
    return out.astype(int)


def build_base_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_len = len(train_df)
    df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

    keys = pd.DataFrame(index=df.index)
    keys["Ticket"] = df["Ticket"].fillna("MISSING").astype(str)
    keys["Surname"] = df["Name"].str.split(",").str[0].fillna("UNKNOWN").str.strip()
    keys["SexRaw"] = df["Sex"].fillna("unknown").astype(str)

    feat = pd.DataFrame(index=df.index)
    feat["PassengerId"] = df["PassengerId"].astype(int)
    feat["Pclass"] = df["Pclass"].astype(int)
    feat["Sex"] = df["Sex"].fillna("unknown").astype(str)
    feat["Embarked"] = df["Embarked"].fillna("S").astype(str)
    feat["SibSp"] = df["SibSp"].fillna(0).astype(int)
    feat["Parch"] = df["Parch"].fillna(0).astype(int)

    # Name-based features
    feat["NameNorm"] = df["Name"].fillna("").map(normalize_name)
    feat["Surname"] = keys["Surname"]
    feat["NameLen"] = df["Name"].fillna("").str.len().astype(int)
    feat["NameWords"] = df["Name"].fillna("").str.split().str.len().astype(int)
    feat["HasParenInName"] = df["Name"].fillna("").str.contains(r"\(").astype(int)

    title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Rare")
    title = title.replace(["Mlle", "Ms"], "Miss")
    title = title.replace(["Mme"], "Mrs")
    title = title.replace(["Lady", "Countess", "Dona"], "Mrs")
    title = title.replace(["Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer"], "Rare")
    feat["Title"] = title.astype(str)

    # Ticket / cabin structure
    feat["Ticket"] = keys["Ticket"]
    feat["TicketPrefix"] = keys["Ticket"].str.extract(r"^([A-Za-z./]+)", expand=False).fillna("NONE")
    feat["TicketItem"] = (
        keys["Ticket"].str.split().map(lambda parts: "NONE" if len(parts) <= 1 else "_".join(parts[:-1]))
    )
    feat["TicketNumberRaw"] = keys["Ticket"].str.extract(r"(\d+)$", expand=False).fillna("NONE")

    feat["Cabin"] = df["Cabin"].fillna("NONE").astype(str)
    feat["Deck"] = feat["Cabin"].str[0].replace("N", "U")
    feat["CabinCount"] = feat["Cabin"].map(lambda x: 0 if x == "NONE" else len(str(x).split()))
    feat["HasCabin"] = (feat["Cabin"] != "NONE").astype(int)

    # Family / ticket group
    feat["FamilySize"] = feat["SibSp"] + feat["Parch"] + 1
    feat["IsAlone"] = (feat["FamilySize"] == 1).astype(int)
    feat["FamilyBin"] = feat["FamilySize"].map(lambda x: "alone" if x == 1 else ("small" if x <= 4 else "large"))
    keys["FamilyKey"] = keys["Surname"] + "_" + feat["FamilySize"].astype(str)
    keys["SexCode"] = feat["Sex"].map({"male": 0, "female": 1}).fillna(0).astype(int)

    ticket_counts = keys["Ticket"].value_counts()
    feat["TicketGroupSize"] = keys["Ticket"].map(ticket_counts).astype(int)

    surname_counts = keys["Surname"].value_counts()
    feat["SurnameFreq"] = keys["Surname"].map(surname_counts).astype(int)

    # Imputation (target-free)
    age = df["Age"].copy()
    age = age.fillna(pd.DataFrame({"Title": feat["Title"], "Pclass": feat["Pclass"], "Age": age})
                     .groupby(["Title", "Pclass"])["Age"]
                     .transform("median"))
    age = age.fillna(age.median())
    feat["Age"] = age.astype(float)
    feat["IsChild"] = (feat["Age"] <= 12).astype(int)

    fare = df["Fare"].copy()
    fare = fare.fillna(pd.DataFrame({"Pclass": feat["Pclass"], "Embarked": feat["Embarked"], "Fare": fare})
                       .groupby(["Pclass", "Embarked"])["Fare"]
                       .transform("median"))
    fare = fare.fillna(fare.median())
    feat["Fare"] = fare.astype(float)
    feat["FarePerPerson"] = feat["Fare"] / feat["TicketGroupSize"].clip(lower=1)

    feat["AgeBin"] = pd.cut(feat["Age"], bins=[0, 5, 12, 18, 30, 45, 60, 120], include_lowest=True).astype(str)
    feat["FareBin"] = safe_qcut(feat["FarePerPerson"], q=5).astype(str)

    feat["Sex_Pclass"] = feat["Sex"] + "_" + feat["Pclass"].astype(str)
    feat["Title_Pclass"] = feat["Title"] + "_" + feat["Pclass"].astype(str)
    feat["Embarked_Pclass"] = feat["Embarked"] + "_" + feat["Pclass"].astype(str)
    feat["IsMother"] = (
        (feat["Sex"] == "female") & (feat["Parch"] > 0) & (feat["Age"] > 18) & (feat["Title"] == "Mrs")
    ).astype(int)

    # Reduce high-cardinality noise a bit.
    for col, min_count in [("TicketPrefix", 10), ("TicketItem", 10), ("TicketNumberRaw", 5), ("Surname", 2)]:
        counts = feat[col].value_counts()
        feat[col] = feat[col].map(lambda x, c=counts, m=min_count: x if c.get(x, 0) >= m else "RARE").astype(str)

    feature_cols = [
        "Pclass", "Sex", "Embarked", "SibSp", "Parch",
        "Title", "Surname",
        "Ticket", "TicketPrefix", "TicketItem", "TicketNumberRaw",
        "Cabin", "Deck",
        "FamilySize", "FamilyBin", "IsAlone", "TicketGroupSize",
        "Age", "AgeBin", "Fare", "FarePerPerson", "FareBin",
        "IsChild", "NameLen", "NameWords", "HasParenInName",
        "CabinCount", "HasCabin", "SurnameFreq",
        "Sex_Pclass", "Title_Pclass", "Embarked_Pclass", "IsMother",
    ]

    categorical_cols = [
        "Sex", "Embarked", "Title", "Surname",
        "Ticket", "TicketPrefix", "TicketItem", "TicketNumberRaw",
        "Cabin", "Deck", "FamilyBin", "AgeBin", "FareBin",
        "Sex_Pclass", "Title_Pclass", "Embarked_Pclass",
    ]

    feat = feat[feature_cols].copy()
    for col in categorical_cols:
        feat[col] = feat[col].fillna("MISSING").astype(str)

    assert len(feat.iloc[:train_len]) == train_len
    return feat, keys, feature_cols, categorical_cols


def compute_group_maps(train_keys: pd.DataFrame, y_fold: np.ndarray) -> tuple[dict, dict, dict, dict]:
    tmp = train_keys.copy()
    tmp["Survived"] = y_fold

    fam = tmp.groupby("FamilyKey")["Survived"].agg(["mean", "count"])
    fam_mean = fam["mean"].copy()
    fam_mean.loc[fam["count"] <= 1] = 0.5

    tick = tmp.groupby("Ticket")["Survived"].agg(["mean", "count"])
    tick_mean = tick["mean"].copy()
    tick_mean.loc[tick["count"] <= 1] = 0.5

    return fam_mean.to_dict(), fam["count"].to_dict(), tick_mean.to_dict(), tick["count"].to_dict()


def map_group_features(keys_slice: pd.DataFrame, maps: tuple[dict, dict, dict, dict]) -> pd.DataFrame:
    fam_mean_map, fam_count_map, tick_mean_map, tick_count_map = maps
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
    best_thr, best_acc = 0.5, float(accuracy_score(y_true, (proba >= 0.5).astype(int)))
    for thr in np.arange(0.35, 0.66, 0.01):
        acc = float(accuracy_score(y_true, (proba >= thr).astype(int)))
        if acc > best_acc:
            best_thr, best_acc = float(round(thr, 2)), acc
    return best_thr, best_acc


def catboost_configs(quick: bool, threads: int) -> dict[str, dict]:
    common = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "allow_writing_files": False,
        "thread_count": threads,
        "verbose": False,
    }

    if quick:
        return {
            "CB_A": {
                **common,
                "iterations": 600,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 6,
                "random_strength": 1.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
            },
            "CB_B": {
                **common,
                "iterations": 450,
                "learning_rate": 0.05,
                "depth": 5,
                "l2_leaf_reg": 10,
                "random_strength": 2.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.85,
            },
            "CB_C": {
                **common,
                "iterations": 700,
                "learning_rate": 0.025,
                "depth": 7,
                "l2_leaf_reg": 12,
                "random_strength": 1.5,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 2.0,
            },
        }

    return {
        "CB_A": {
            **common,
            "iterations": 2000,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 8,
            "random_strength": 1.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
        },
        "CB_B": {
            **common,
            "iterations": 1400,
            "learning_rate": 0.03,
            "depth": 5,
            "l2_leaf_reg": 12,
            "random_strength": 2.0,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.85,
        },
        "CB_C": {
            **common,
            "iterations": 2400,
            "learning_rate": 0.015,
            "depth": 7,
            "l2_leaf_reg": 16,
            "random_strength": 1.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 2.0,
        },
        "CB_D": {
            **common,
            "iterations": 1600,
            "learning_rate": 0.025,
            "depth": 4,
            "l2_leaf_reg": 20,
            "random_strength": 3.0,
            "bootstrap_type": "MVS",
        },
    }


def save_submission(path: Path, passenger_ids: pd.Series, preds: np.ndarray) -> None:
    pd.DataFrame({"PassengerId": passenger_ids, "Survived": preds.astype(int)}).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.quick and args.n_splits == 10:
        args.n_splits = 5
    args.seeds_bagging = max(1, int(args.seeds_bagging))

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test CSV in {args.data_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    passenger_ids = test_df["PassengerId"].copy()
    y = train_df["Survived"].astype(int).to_numpy()
    train_len = len(train_df)
    test_len = len(test_df)

    base_all, keys_all, base_cols, cat_cols = build_base_features(train_df, test_df)
    X_train_base = base_all.iloc[:train_len].reset_index(drop=True)
    X_test_base = base_all.iloc[train_len:].reset_index(drop=True)
    keys_train = keys_all.iloc[:train_len].reset_index(drop=True)
    keys_test = keys_all.iloc[train_len:].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(skf.split(X_train_base, y))

    sample_maps = compute_group_maps(keys_train.iloc[splits[0][0]], y[splits[0][0]])
    group_cols = list(map_group_features(keys_train.iloc[splits[0][1]], sample_maps).columns)
    all_cols = base_cols + group_cols
    all_cat_cols = [c for c in cat_cols if c in all_cols]

    configs = catboost_configs(args.quick, args.threads)
    model_names = list(configs.keys())
    n_models = len(model_names)

    oof = np.zeros((train_len, n_models), dtype=float)
    test_fold_preds = np.zeros((test_len, n_models, args.n_splits), dtype=float)

    print("=" * 62)
    print("CATBOOST LOCAL (FOLD-SAFE GROUP FEATURES + SEED BAGGING)")
    print("=" * 62)
    print(f"Train={train_len} Test={test_len} Folds={args.n_splits}")
    print(f"Base features={len(base_cols)} Group features={len(group_cols)}")
    print(f"Configs={model_names} | Seeds bagging={args.seeds_bagging}")

    for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
        y_tr, y_val = y[tr_idx], y[val_idx]
        fold_maps = compute_group_maps(keys_train.iloc[tr_idx], y_tr)

        g_tr = map_group_features(keys_train.iloc[tr_idx], fold_maps).reset_index(drop=True)
        g_val = map_group_features(keys_train.iloc[val_idx], fold_maps).reset_index(drop=True)
        g_test = map_group_features(keys_test, fold_maps).reset_index(drop=True)

        xtr = pd.concat([X_train_base.iloc[tr_idx].reset_index(drop=True), g_tr], axis=1)[all_cols].copy()
        xval = pd.concat([X_train_base.iloc[val_idx].reset_index(drop=True), g_val], axis=1)[all_cols].copy()
        xtest = pd.concat([X_test_base.reset_index(drop=True), g_test], axis=1)[all_cols].copy()

        for c in all_cat_cols:
            xtr[c] = xtr[c].astype(str)
            xval[c] = xval[c].astype(str)
            xtest[c] = xtest[c].astype(str)

        cat_idx = [xtr.columns.get_loc(c) for c in all_cat_cols]
        train_pool = Pool(xtr, y_tr, cat_features=cat_idx)
        val_pool = Pool(xval, y_val, cat_features=cat_idx)
        test_pool = Pool(xtest, cat_features=cat_idx)

        print(f"\nFold {fold_idx}/{args.n_splits} - train={len(tr_idx)} val={len(val_idx)}")
        for m_idx, name in enumerate(model_names):
            cfg = configs[name]
            val_bag = np.zeros(len(val_idx), dtype=float)
            test_bag = np.zeros(test_len, dtype=float)

            for seed_idx in range(args.seeds_bagging):
                seed = args.seed + fold_idx * 100 + m_idx * 1000 + seed_idx
                model = CatBoostClassifier(
                    **cfg,
                    random_seed=seed,
                )
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    use_best_model=True,
                    early_stopping_rounds=150 if not args.quick else 80,
                )
                val_bag += model.predict_proba(val_pool)[:, 1]
                test_bag += model.predict_proba(test_pool)[:, 1]

            val_proba = val_bag / args.seeds_bagging
            test_proba = test_bag / args.seeds_bagging

            oof[val_idx, m_idx] = val_proba
            test_fold_preds[:, m_idx, fold_idx - 1] = test_proba

            fold_acc = accuracy_score(y_val, (val_proba >= 0.5).astype(int))
            print(f"  {name:4s} fold acc={fold_acc:.4f}")

    s_test = test_fold_preds.mean(axis=2)

    print("\n" + "=" * 62)
    print("BASE CATBOOST VARIANTS (OOF)")
    print("=" * 62)
    base_scores: dict[str, float] = {}
    for i, name in enumerate(model_names):
        acc = float(accuracy_score(y, (oof[:, i] >= 0.5).astype(int)))
        base_scores[name] = acc
        print(f"  {name:4s} OOF acc={acc:.4f}")

    # Vote all
    vote_train = oof.mean(axis=1)
    vote_test = s_test.mean(axis=1)
    vote_thr, vote_oof = best_threshold(y, vote_train)
    vote_preds = (vote_test >= vote_thr).astype(int)

    # Top-2 weighted
    top2 = sorted(model_names, key=lambda n: base_scores[n], reverse=True)[:2]
    top2_idx = [model_names.index(n) for n in top2]
    top2_w_raw = np.array([base_scores[n] for n in top2], dtype=float)
    top2_w = top2_w_raw / top2_w_raw.sum()
    top2_train = (oof[:, top2_idx] * top2_w).sum(axis=1)
    top2_test = (s_test[:, top2_idx] * top2_w).sum(axis=1)
    top2_thr, top2_oof = best_threshold(y, top2_train)
    top2_preds = (top2_test >= top2_thr).astype(int)

    # Stacking over CatBoost variants
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    stack_oof = np.zeros(train_len, dtype=float)
    for tr_idx, val_idx in splits:
        meta_fold = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
        meta_fold.fit(oof[tr_idx], y[tr_idx])
        stack_oof[val_idx] = meta_fold.predict_proba(oof[val_idx])[:, 1]
    meta.fit(oof, y)
    stack_test = meta.predict_proba(s_test)[:, 1]
    stack_thr, stack_oof_acc = best_threshold(y, stack_oof)
    stack_preds = (stack_test >= stack_thr).astype(int)

    # Blend stack+vote
    blend_train = (stack_oof_acc * stack_oof + vote_oof * vote_train) / max(stack_oof_acc + vote_oof, 1e-9)
    blend_test = (stack_oof_acc * stack_test + vote_oof * vote_test) / max(stack_oof_acc + vote_oof, 1e-9)
    blend_thr, blend_oof = best_threshold(y, blend_train)
    blend_preds = (blend_test >= blend_thr).astype(int)

    # Majority among stack/vote/top2
    majority_train = (
        ((stack_oof >= stack_thr).astype(int) + (vote_train >= vote_thr).astype(int) + (top2_train >= top2_thr).astype(int)) >= 2
    ).astype(int)
    majority_test = ((stack_preds + vote_preds + top2_preds) >= 2).astype(int)
    majority_oof = float(accuracy_score(y, majority_train))

    print("\n" + "=" * 62)
    print("ENSEMBLES (HONEST OOF)")
    print("=" * 62)
    print(f"  Stacking            OOF={stack_oof_acc:.4f} thr={stack_thr:.2f} survived={int(stack_preds.sum())}")
    print(f"  Vote(all configs)   OOF={vote_oof:.4f} thr={vote_thr:.2f} survived={int(vote_preds.sum())}")
    print(f"  Top2({','.join(top2)})     OOF={top2_oof:.4f} thr={top2_thr:.2f} survived={int(top2_preds.sum())}")
    print(f"  Blend(stack+vote)   OOF={blend_oof:.4f} thr={blend_thr:.2f} survived={int(blend_preds.sum())}")
    print(f"  Majority(3x)        OOF={majority_oof:.4f} thr=n/a  survived={int(majority_test.sum())}")

    outputs = {
        "submission_catboost_stack.csv": (stack_oof_acc, stack_preds),
        "submission_catboost_vote.csv": (vote_oof, vote_preds),
        "submission_catboost_top2.csv": (top2_oof, top2_preds),
        "submission_catboost_blend.csv": (blend_oof, blend_preds),
        "submission_catboost_majority.csv": (majority_oof, majority_test),
    }

    print("\nAgreement on test set:")
    print(f"  stack vs vote : {(stack_preds == vote_preds).sum()}/{test_len}")
    print(f"  stack vs top2 : {(stack_preds == top2_preds).sum()}/{test_len}")
    print(f"  vote  vs top2 : {(vote_preds == top2_preds).sum()}/{test_len}")

    for name, (_, preds) in outputs.items():
        save_submission(args.data_dir / name, passenger_ids, preds)

    best_name = max(outputs, key=lambda k: outputs[k][0])
    save_submission(args.data_dir / "submission.csv", passenger_ids, outputs[best_name][1])

    print("\nSaved:")
    for name, (score, _) in sorted(outputs.items(), key=lambda kv: kv[1][0], reverse=True):
        print(f"  {name:30s} OOF={score:.4f}")
    print(f"  submission.csv -> {best_name}")


if __name__ == "__main__":
    main()
