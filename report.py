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
WHY = {"R_pktlen": "packet size", "R_timing": "timing (IAT)",
       "R_periodicity": "beacon regularity"}


def heatmap(piv, hatch, path, label):
    cmap = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#f7f7a0", "#1a9850"])
    fig, ax = plt.subplots(figsize=(1.8 + 1.15 * piv.shape[1], 1.3 + 0.6 * piv.shape[0]))
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if bool(hatch.values[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                             hatch="///", edgecolor="gray", linewidth=0))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="dimgray")
            else:
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v < 0.18 else "black")
    ax.set_title(f"R(tool, phase) against manual - {label}", fontsize=11)
    ax.text(0.5, -0.32, "hatched n/a = phase absent from ground truth (cannot be scored)",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="dimgray")
    fig.colorbar(im, ax=ax, label="similarity (1 = like manual)", shrink=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", nargs="+", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-prefix", default="step4")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in a.r], ignore_index=True)
    df["n_manual"] = df.get("n_manual", 0)

    # phases that have ground truth (appear in manual at least once)
    manual_phases = sorted(set(df.loc[df["n_manual"].fillna(0) > 0, "phase"]))
    other_phases = [p for p in dict.fromkeys(df["phase"]) if p not in manual_phases]
    phases = manual_phases + other_phases
    tools = sorted(df["tool"].dropna().unique())

    scored = {(r.tool, r.phase): r.R_combined
              for r in df.itertuples() if pd.notna(r.R_combined)}

    mat = pd.DataFrame(0.0, index=tools, columns=phases)
    hatch = pd.DataFrame(False, index=tools, columns=phases)
    for t in tools:
        for p in phases:
            if p not in manual_phases:
                hatch.loc[t, p] = True          # no ground truth -> n/a
            else:
                mat.loc[t, p] = scored.get((t, p), 0.0)  # score, or solid 0

    heatmap(mat, hatch, f"{a.out_prefix}_heatmap.png", a.label)

    # ranking: ground-truth phases only; a phase not run counts as 0
    rank_rows = []
    for t in tools:
        vals = [scored.get((t, p), 0.0) for p in manual_phases]
        covered = [scored[(t, p)] for p in manual_phases if (t, p) in scored]
        rank_rows.append({
            "tool": t,
            "mean_R_all_phases": np.mean(vals),
            "mean_R_phases_run": np.mean(covered) if covered else np.nan,
            "min_R": min(vals),
            "n_phases_run": len(covered),
            "n_phases_manual": len(manual_phases),
        })
    rank = pd.DataFrame(rank_rows).sort_values("mean_R_all_phases", ascending=False)

    # per-cell driver (scored cells only)
    have = [g for g in GROUPS if g in df.columns]
    cells = df[df["R_combined"].notna()].copy()
    if have:
        cells["driver"] = cells[have].idxmin(axis=1).map(WHY)
    cells = cells[["tool", "phase", "R_combined"] + have +
                  (["driver"] if have else [])]

    rank.to_csv(f"{a.out_prefix}_ranking.csv", index=False)
    cells.to_csv(f"{a.out_prefix}_cells.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("== RANKING ==")
        print(rank.round(3).to_string(index=False))
        print("\nground-truth phases:", manual_phases)
        print("phases without ground truth (n/a):", other_phases or "(none)")
    print("\nsaved:", f"{a.out_prefix}_heatmap.png,",
          f"{a.out_prefix}_ranking.csv,", f"{a.out_prefix}_cells.csv")


if __name__ == "__main__":
    main()
