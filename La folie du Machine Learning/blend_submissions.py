#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "march-machine-learning-mania-2026"


def resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(df.columns) != {"ID", "Pred"}:
        raise ValueError(f"{path.name}: colonnes attendues {{'ID','Pred'}}, trouvé {list(df.columns)}")
    if df["ID"].duplicated().any():
        n_dup = int(df["ID"].duplicated().sum())
        raise ValueError(f"{path.name}: {n_dup} IDs dupliqués")
    df = df[["ID", "Pred"]].copy()
    df["Pred"] = df["Pred"].astype(float)
    return df


def align_submissions(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    merged = df_a.merge(df_b, on="ID", how="inner", suffixes=("_a", "_b"))
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        missing_a = len(df_a) - len(merged)
        missing_b = len(df_b) - len(merged)
        raise ValueError(
            f"IDs non alignés entre {label_a} et {label_b} "
            f"(manquants après merge: {missing_a} / {missing_b})"
        )
    return merged


def build_tourney_labels(data_dir: Path) -> pd.DataFrame:
    """Construit les labels réels de tournoi NCAA orientés comme Kaggle (TeamA=min TeamID)."""
    rows = []
    for fname in ("MNCAATourneyCompactResults.csv", "WNCAATourneyCompactResults.csv"):
        df = pd.read_csv(data_dir / fname, usecols=["Season", "WTeamID", "LTeamID"])
        for _, row in df.iterrows():
            season = int(row["Season"])
            w_id = int(row["WTeamID"])
            l_id = int(row["LTeamID"])
            if w_id < l_id:
                team_a, team_b, target = w_id, l_id, 1
            else:
                team_a, team_b, target = l_id, w_id, 0
            rows.append({"ID": f"{season}_{team_a}_{team_b}", "Target": target, "Season": season})

    labels = pd.DataFrame(rows)
    labels = labels.drop_duplicates(subset=["ID"]).reset_index(drop=True)
    return labels


def stage1_alpha_search(
    stage1_a: pd.DataFrame,
    stage1_b: pd.DataFrame,
    labels: pd.DataFrame,
    alphas: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    merged = align_submissions(stage1_a, stage1_b, "stage1_a", "stage1_b")
    scored = merged.merge(labels[["ID", "Target", "Season"]], on="ID", how="inner")
    if scored.empty:
        raise ValueError("Aucun label de tournoi trouvé dans le Stage 1 pour scorer localement les alphas.")

    y = scored["Target"].to_numpy(dtype=float)
    p_a = scored["Pred_a"].to_numpy(dtype=float)
    p_b = scored["Pred_b"].to_numpy(dtype=float)

    results = []
    for alpha in alphas:
        pred = alpha * p_a + (1.0 - alpha) * p_b
        brier = float(np.mean((pred - y) ** 2))
        results.append({"alpha": float(alpha), "brier_local": brier, "n_games": int(len(scored))})

    res_df = pd.DataFrame(results).sort_values(["brier_local", "alpha"]).reset_index(drop=True)
    best_alpha = float(res_df.iloc[0]["alpha"])
    return res_df, best_alpha


def blend_df(df_a: pd.DataFrame, df_b: pd.DataFrame, alpha: float) -> pd.DataFrame:
    merged = align_submissions(df_a, df_b, "A", "B")
    out = merged[["ID"]].copy()
    out["Pred"] = alpha * merged["Pred_a"].to_numpy() + (1.0 - alpha) * merged["Pred_b"].to_numpy()
    out["Pred"] = out["Pred"].clip(0.0, 1.0)
    return out


def infer_stage2_pair(stage1_path: Path | None) -> Path | None:
    if stage1_path is None:
        return None
    name = stage1_path.name
    if "stage1" not in name:
        return None
    candidate = stage1_path.with_name(name.replace("stage1", "stage2", 1))
    return candidate if candidate.exists() else None


def parse_alpha_grid(args: argparse.Namespace) -> np.ndarray:
    if args.alphas:
        vals = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
        alphas = np.array(vals, dtype=float)
    else:
        step = float(args.grid_step)
        n = int(round(1.0 / step))
        alphas = np.linspace(0.0, 1.0, n + 1)
    alphas = np.unique(np.clip(alphas, 0.0, 1.0))
    return np.sort(alphas)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blend 2 soumissions March Madness (Stage1/Stage2) et cherche un alpha sur Stage1 local."
    )
    parser.add_argument("--stage1-a", required=True, help="CSV soumission Stage1 A (ID,Pred)")
    parser.add_argument("--stage1-b", required=True, help="CSV soumission Stage1 B (ID,Pred)")
    parser.add_argument("--stage2-a", help="CSV soumission Stage2 A (ID,Pred). Auto-déduit si omis.")
    parser.add_argument("--stage2-b", help="CSV soumission Stage2 B (ID,Pred). Auto-déduit si omis.")
    parser.add_argument("--alpha", type=float, help="Alpha fixe (si non fourni, recherche sur grille via Stage1 local)")
    parser.add_argument("--alphas", help="Liste explicite d'alphas, ex: 0,0.1,0.2,...,1")
    parser.add_argument("--grid-step", type=float, default=0.01, help="Pas de grille alpha si --alphas absent (défaut: 0.01)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dossier des CSV Kaggle (pour labels Stage1 locaux)")
    parser.add_argument("--out-tag", help="Suffixe de sortie (défaut: horodatage)")
    parser.add_argument("--write-top-k", type=int, default=1, help="Nombre de meilleurs alphas à exporter (défaut: 1)")
    args = parser.parse_args()

    stage1_a_path = resolve_path(args.stage1_a)
    stage1_b_path = resolve_path(args.stage1_b)
    stage2_a_path = resolve_path(args.stage2_a) if args.stage2_a else infer_stage2_pair(stage1_a_path)
    stage2_b_path = resolve_path(args.stage2_b) if args.stage2_b else infer_stage2_pair(stage1_b_path)
    data_dir = resolve_path(args.data_dir)

    if stage1_a_path is None or stage1_b_path is None:
        raise SystemExit("Stage1 A/B requis.")
    if not stage1_a_path.exists() or not stage1_b_path.exists():
        raise SystemExit("Un des fichiers Stage1 n'existe pas.")
    if stage2_a_path and not stage2_a_path.exists():
        raise SystemExit(f"Stage2 A introuvable: {stage2_a_path}")
    if stage2_b_path and not stage2_b_path.exists():
        raise SystemExit(f"Stage2 B introuvable: {stage2_b_path}")

    print("Chargement des soumissions...")
    s1_a = load_submission(stage1_a_path)
    s1_b = load_submission(stage1_b_path)
    s2_a = load_submission(stage2_a_path) if stage2_a_path else None
    s2_b = load_submission(stage2_b_path) if stage2_b_path else None

    out_tag = args.out_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    alphas = parse_alpha_grid(args)

    # Recherche alpha sur Stage1 local (ou alpha fixe)
    search_df = None
    if args.alpha is None:
        if data_dir is None or not data_dir.exists():
            raise SystemExit("Dossier data Kaggle introuvable pour la recherche alpha locale. Fournis --alpha manuellement.")
        print(f"Construction des labels tournoi depuis {data_dir}...")
        labels = build_tourney_labels(data_dir)
        search_df, best_alpha = stage1_alpha_search(s1_a, s1_b, labels, alphas)
        search_path = ROOT / f"blend_alpha_search_{out_tag}.csv"
        search_df.to_csv(search_path, index=False)
        print(f"Recherche alpha sauvegardée: {search_path.name}")
        print("Top 10 alphas (score local Stage1):")
        print(search_df.head(10).to_string(index=False))
    else:
        best_alpha = float(np.clip(args.alpha, 0.0, 1.0))
        print(f"Alpha fixe utilisé: {best_alpha:.4f}")

    # Export des blends (un ou plusieurs meilleurs alphas)
    export_alphas: list[float]
    if search_df is not None:
        k = max(1, int(args.write_top_k))
        export_alphas = [float(a) for a in search_df.head(k)["alpha"].tolist()]
    else:
        export_alphas = [best_alpha]

    for alpha in export_alphas:
        alpha_tag = f"{alpha:.3f}".replace(".", "p")
        s1_blend = blend_df(s1_a, s1_b, alpha)
        s1_out = ROOT / f"submission_stage1_blend_{out_tag}_a{alpha_tag}.csv"
        s1_blend.to_csv(s1_out, index=False)
        print(f"Stage1 blend -> {s1_out.name} ({len(s1_blend):,} lignes)")

        if s2_a is not None and s2_b is not None:
            s2_blend = blend_df(s2_a, s2_b, alpha)
            s2_out = ROOT / f"submission_stage2_blend_{out_tag}_a{alpha_tag}.csv"
            s2_blend.to_csv(s2_out, index=False)
            print(f"Stage2 blend -> {s2_out.name} ({len(s2_blend):,} lignes)")

    print("\nTerminé.")
    if search_df is not None:
        print(f"Alpha localement optimal (sur matchs tournoi connus du Stage1) : {best_alpha:.4f}")


if __name__ == "__main__":
    main()
