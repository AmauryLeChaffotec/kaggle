"""
Kaggle-ready Titanic solver with TensorFlow Decision Forests (TF-DF).

Inspired by the notebook:
  titanic-competition-w-tensorflow-decision-forests-score-0.80.ipynb

This script is intended to run on Kaggle (where TF-DF is available), but it also
supports local CSV paths if the package is installed locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/kaggle/input/titanic/train.csv")
    parser.add_argument("--test", type=str, default="/kaggle/input/titanic/test.csv")
    parser.add_argument("--output", type=str, default="/kaggle/working/submission_tfdf.csv")
    parser.add_argument("--n-models", type=int, default=25, help="Average predictions across multiple seeds")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--quick", action="store_true", help="Fewer trees for faster iteration")
    return parser.parse_args()


def resolve_default_path(path_str: str, local_name: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    local = Path(__file__).resolve().parent / local_name
    return local


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def normalize_name(x: str) -> str:
        return " ".join([v.strip(",()[].\"'") for v in str(x).split(" ")])

    def ticket_number(x: str) -> str:
        parts = str(x).split(" ")
        return parts[-1] if parts else "NONE"

    def ticket_item(x: str) -> str:
        parts = str(x).split(" ")
        if len(parts) <= 1:
            return "NONE"
        return "_".join(parts[:-1])

    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)
    return df


def build_feature_list(train_df: pd.DataFrame) -> list[str]:
    input_features = list(train_df.columns)
    for col in ["Ticket", "PassengerId", "Survived"]:
        if col in input_features:
            input_features.remove(col)
    return input_features


def make_model(tfdf, input_features: list[str], seed: int, quick: bool):
    kwargs = dict(
        verbose=0,
        features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
        exclude_non_specified_features=True,
        random_seed=seed,
        honest=True,
    )

    # Tuned-ish config taken from the notebook family (good LB behavior on Titanic).
    if quick:
        kwargs.update(
            num_trees=600,
            shrinkage=0.05,
            min_examples=1,
            categorical_algorithm="RANDOM",
            split_axis="SPARSE_OBLIQUE",
            sparse_oblique_normalization="MIN_MAX",
            sparse_oblique_num_projections_exponent=1.5,
        )
    else:
        kwargs.update(
            num_trees=2000,
            shrinkage=0.05,
            min_examples=1,
            categorical_algorithm="RANDOM",
            split_axis="SPARSE_OBLIQUE",
            sparse_oblique_normalization="MIN_MAX",
            sparse_oblique_num_projections_exponent=2.0,
        )

    return tfdf.keras.GradientBoostedTreesModel(**kwargs)


def main() -> None:
    args = parse_args()

    try:
        import tensorflow as tf
        import tensorflow_decision_forests as tfdf
    except Exception as exc:
        raise SystemExit(
            "tensorflow_decision_forests is required. Run this on Kaggle or install TF-DF locally. "
            f"Import error: {type(exc).__name__}: {exc}"
        ) from exc

    train_path = resolve_default_path(args.train, "train.csv")
    test_path = resolve_default_path(args.test, "test.csv")
    output_path = Path(args.output)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"CSV not found. train={train_path} test={test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    serving_ids = test_df["PassengerId"].copy()

    train_prep = preprocess(train_df)
    test_prep = preprocess(test_df)
    input_features = build_feature_list(train_prep)

    def tokenize_names(features, labels=None):
        # TF-DF can use tokenized text directly.
        features["Name"] = tf.strings.split(features["Name"])
        return features, labels

    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_prep, label="Survived").map(tokenize_names)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_prep).map(tokenize_names)

    print(f"TF-DF version: {tfdf.__version__}")
    print(f"Input features ({len(input_features)}): {input_features}")
    print(f"Averaging {args.n_models} model(s), seeds {args.start_seed}..{args.start_seed + args.n_models - 1}")

    preds_sum = None
    accs = []

    for i in range(args.n_models):
        seed = args.start_seed + i
        model = make_model(tfdf, input_features, seed=seed, quick=args.quick)
        model.fit(train_ds, verbose=0)

        try:
            ev = model.make_inspector().evaluation()
            if ev is not None and getattr(ev, "accuracy", None) is not None:
                accs.append(float(ev.accuracy))
                print(f"  seed={seed:3d} self-eval acc={ev.accuracy:.4f}")
            else:
                print(f"  seed={seed:3d} trained")
        except Exception:
            print(f"  seed={seed:3d} trained")

        p = model.predict(test_ds, verbose=0)[:, 0]
        if preds_sum is None:
            preds_sum = p
        else:
            preds_sum += p

    proba = preds_sum / float(args.n_models)
    preds = (proba >= args.threshold).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"PassengerId": serving_ids, "Survived": preds})
    sub.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(sub["Survived"].value_counts())
    if accs:
        print(f"Mean self-eval accuracy across seeds: {np.mean(accs):.4f}")


if __name__ == "__main__":
    main()
