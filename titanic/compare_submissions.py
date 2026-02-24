"""
Compare Titanic submission files and list disagreements.

Examples:
  python compare_submissions.py
  python compare_submissions.py submission_tfdf.csv submission_stack.csv
  python compare_submissions.py submission_tfdf.csv submission_stack.csv --show-all-cols
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd


DEFAULT_FILES = [
    "submission_tfdf.csv",
    "submission.csv",
    "submission_stack.csv",
    "submission_top3.csv",
    "submission_blend.csv",
    "submission_vote.csv",
    "submission_majority.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        help="Submission CSV files to compare. If omitted, common files in current folder are auto-detected.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing test.csv (used to enrich disagreement output).",
    )
    parser.add_argument(
        "--show-all-cols",
        action="store_true",
        help="Show more columns from test.csv in the disagreement table.",
    )
    return parser.parse_args()


def resolve_files(args: argparse.Namespace) -> list[Path]:
    if args.files:
        return [Path(f) if Path(f).is_absolute() else (args.data_dir / f) for f in args.files]

    found = []
    for name in DEFAULT_FILES:
        p = args.data_dir / name
        if p.exists():
            found.append(p)
    return found


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"PassengerId", "Survived"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {required}")
    return df[["PassengerId", "Survived"]].copy()


def main() -> None:
    args = parse_args()
    files = resolve_files(args)
    if len(files) < 2:
        raise SystemExit("Need at least 2 submission files to compare.")

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"Missing file(s): {missing}")

    subs: dict[str, pd.DataFrame] = {}
    for path in files:
        name = path.name
        subs[name] = load_submission(path).rename(columns={"Survived": name})

    merged = None
    for i, (name, df) in enumerate(subs.items()):
        merged = df if i == 0 else merged.merge(df, on="PassengerId", how="inner")

    if merged is None or merged.empty:
        raise SystemExit("No rows after merge.")

    pred_cols = [c for c in merged.columns if c != "PassengerId"]
    print(f"Rows compared: {len(merged)}")
    print("\nSurvived counts:")
    for col in pred_cols:
        print(f"  {col:24s} {int(merged[col].sum())}")

    print("\nPairwise differences:")
    for a, b in itertools.combinations(pred_cols, 2):
        diff = int((merged[a] != merged[b]).sum())
        print(f"  {a:24s} vs {b:24s}  diff={diff}")

    # Focus disagreement report on the first two files requested/resolved.
    left, right = pred_cols[0], pred_cols[1]
    diff_mask = merged[left] != merged[right]
    diff_rows = merged.loc[diff_mask, ["PassengerId", left, right]].copy()
    print(f"\nDisagreements ({left} vs {right}): {len(diff_rows)}")

    test_path = args.data_dir / "test.csv"
    if test_path.exists() and len(diff_rows):
        test_df = pd.read_csv(test_path)
        base_cols = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
        if args.show_all_cols:
            ctx_cols = [c for c in test_df.columns if c in base_cols]
        else:
            ctx_cols = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
        ctx = test_df[ctx_cols].copy()
        diff_rows = diff_rows.merge(ctx, on="PassengerId", how="left")

    if len(diff_rows):
        print(diff_rows.sort_values("PassengerId").to_string(index=False))
    else:
        print("  (none)")

    if len(pred_cols) >= 3:
        consensus = merged[pred_cols].nunique(axis=1)
        print("\nConsensus summary:")
        print(f"  all agree rows    : {int((consensus == 1).sum())}")
        print(f"  partial disagree  : {int((consensus > 1).sum())}")


if __name__ == "__main__":
    main()
