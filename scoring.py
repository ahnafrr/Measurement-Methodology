"""Step 3 - Similarity score R(tool, phase) against manual ground truth.

A tool is compared against manual ground truth for ONE same scenario.
Directory structure: each tool/scenario has its own flow and sum files,
so the script accepts tool and manual file paths separately.

Usage:
  python3 scoring.py \
      --tool-flows "A.csv" "B.csv" "C.csv" "D.csv" "E.csv" \
      --tool-sum "A_sum.csv" "B_sum.csv" "C_sum.csv" "D_sum.csv" "E_sum.csv" \
      --manual-flows "output_all_flows.csv" \
      --manual-sum "output_all_summary.csv" \
      --metric wasserstein \
      --out R_all.csv

Comparison is done per phase (phase column). Tool phases absent from manual
are flagged, not forced.

Feature groups:
  R_pktlen      : mean_ps per flow (3 directions)  -> distribution
  R_timing      : mean_piat_ms, cv_piat per flow   -> distribution
  R_periodicity : gap_dom_median_s, gap_dom_robust_cv (bucket scalar)
R_combined = average of available feature groups.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon

DIST_FEATS = {
    "R_pktlen": ["bidirectional_mean_ps", "src2dst_mean_ps", "dst2src_mean_ps"],
    "R_timing": ["bidirectional_mean_piat_ms", "bidirectional_cv_piat"],
}
SCALAR_FEATS = {"R_periodicity": ["gap_dom_median_s", "gap_dom_robust_cv"]}


def d_to_sim(d):
    return float(np.exp(-d))


def js_dist(a, b, bins=30):
    lo, hi = np.nanmin(np.concatenate([a, b])), np.nanmax(np.concatenate([a, b]))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, edges)[0] + 1e-9
    pb = np.histogram(b, edges)[0] + 1e-9
    return float(jensenshannon(pa / pa.sum(), pb / pb.sum()))


def dist_feature(tool_vals, man_vals, metric, log=True):
    a = pd.Series(tool_vals).dropna().to_numpy(float)
    b = pd.Series(man_vals).dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    p = ks_2samp(a, b).pvalue  # KS on original scale
    if log and np.nanmin(np.concatenate([a, b])) >= 0:
        a, b = np.log1p(a), np.log1p(b)  # heavy-tailed features
    if metric == "js":
        d = js_dist(a, b)
    else:
        scale = b.std() or np.subtract(*np.percentile(b, [75, 25])) or 1.0
        d = wasserstein_distance(a, b) / scale
    return d_to_sim(d), p


def scalar_sim(t, m):
    if pd.isna(t) or pd.isna(m):
        return np.nan
    scale = (abs(m) + abs(t)) / 2 or 1.0
    return d_to_sim(abs(t - m) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool-flows", nargs="+", required=True)
    ap.add_argument("--tool-sum", nargs="+", required=True)
    ap.add_argument("--manual-flows", nargs="+", required=True)
    ap.add_argument("--manual-sum", nargs="+", required=True)
    ap.add_argument("--metric", choices=["wasserstein", "js"], default="wasserstein")
    ap.add_argument("--out", default="R_scores.csv")
    a = ap.parse_args()

    load = lambda paths: pd.concat([pd.read_csv(p, low_memory=False) for p in paths],
                                   ignore_index=True)
    tf, ts = load(a.tool_flows), load(a.tool_sum)
    mf, ms = load(a.manual_flows), load(a.manual_sum)  # manual = combined pool
    if "tool" not in tf:
        tf["tool"] = "tool"
    if "tool" not in ts:
        ts["tool"] = tf["tool"].iloc[0]

    rows = []
    # Each (tool, phase) is evaluated individually -> multiple tools allowed in tool-flows.
    for (tool_name, phase), g in tf.groupby(["tool", "phase"]):
        mg = mf[mf.phase == phase]
        row = {"tool": tool_name, "phase": phase, "n_tool": len(g), "n_manual": len(mg)}
        if mg.empty:
            row["note"] = "no matching phase in manual"
            rows.append(row)
            continue
        group_sims = {}
        for grp, feats in DIST_FEATS.items():
            sims, ps = [], []
            for f in feats:
                if f in g and f in mg:
                    s, p = dist_feature(g[f], mg[f], a.metric)
                    sims.append(s)
                    ps.append(p)
            group_sims[grp] = np.nanmean(sims) if sims else np.nan
            row[grp] = group_sims[grp]
            row[grp + "_ks_minp"] = np.nanmin(ps) if ps and not all(pd.isna(ps)) else np.nan
        trow = ts[(ts.tool == tool_name) & (ts.phase == phase)]
        mrow = ms[ms.phase == phase]
        for grp, feats in SCALAR_FEATS.items():
            sims = [scalar_sim(trow.iloc[0][f], mrow.iloc[0][f])
                    for f in feats if len(trow) and len(mrow)
                    and f in trow.columns and f in mrow.columns]
            group_sims[grp] = np.nanmean(sims) if sims else np.nan
            row[grp] = group_sims[grp]
        vals = [v for v in group_sims.values() if not pd.isna(v)]
        row["R_combined"] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    cols = [c for c in ["tool", "phase", "n_tool", "n_manual", "R_pktlen",
                        "R_timing", "R_periodicity", "R_combined", "note"] if c in out.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out[cols].round(3).to_string(index=False))
    print("\nsaved:", a.out)


if __name__ == "__main__":
    main()


