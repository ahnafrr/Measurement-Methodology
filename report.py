"""
Usage:
  python3 report.py --r R_ransomware_keep.csv --label keep --out-prefix step4_ransomware_keep

"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

GROUPS = ["R_pktlen", "R_timing", "R_periodicity"]
WHY = {"R_pktlen": "packet size", "R_timing": "timing pattern (IAT)",
       "R_periodicity": "beacon periodicity"}


def heatmap(piv, path, label):
    cmap = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#f7f7a0", "#1a9850"])
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * piv.shape[1], 1.2 + 0.6 * piv.shape[0]))
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v < 0.18 else "black")
    ax.set_title(f"R(tool, phase) against manual - {label}", fontsize=11)
    fig.colorbar(im, ax=ax, label="similarity (1 = identical to manual)", shrink=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", nargs="+", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-prefix", default="step4")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in a.r], ignore_index=True)
    df = df[df["R_combined"].notna()].copy()
    if df.empty:
        raise SystemExit("No rows with R_combined found. Check scoring.py output files.")

    piv = df.pivot_table(index="tool", columns="phase", values="R_combined")
    piv = piv.fillna(0.0)
    heatmap(piv, f"{a.out_prefix}_heatmap.png", a.label)

    # Root cause for each cell: feature group with lowest score
    have = [g for g in GROUPS if g in df.columns]
    df["cause"] = df[have].idxmin(axis=1).map(WHY)
    df["cause_score"] = df[have].min(axis=1)

    # Per-tool summary
    summ = (df.groupby("tool")
              .agg(mean_R=("R_combined", "mean"),
                   min_R=("R_combined", "min"),
                   n_phases=("phase", "nunique"))
              .reset_index()
              .sort_values("mean_R", ascending=False))
    worst = df.loc[df.groupby("tool")["R_combined"].idxmin(),
                   ["tool", "phase", "R_combined", "cause"]]
    worst = worst.rename(columns={"phase": "worst_phase",
                                  "R_combined": "worst_R",
                                  "cause": "worst_cause"})
    summ = summ.merge(worst, on="tool")

    cell = df[["tool", "phase", "R_combined"] + have + ["cause"]].copy()
    cell.to_csv(f"{a.out_prefix}_cells.csv", index=False)
    summ.to_csv(f"{a.out_prefix}_ranking.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("== RANKING (sorted by mean R) ==")
        print(summ.round(3).to_string(index=False))
        print("\n== ROOT CAUSE PER CELL ==")
        print(cell.round(3).to_string(index=False))
    print("\nsaved:", f"{a.out_prefix}_heatmap.png,",
          f"{a.out_prefix}_ranking.csv,", f"{a.out_prefix}_cells.csv")


if __name__ == "__main__":
    main()
