"""
Input : One or more CSV files from labelFlows.py (columns: tool, scenario, phase).
        For --mode keep, CSV must contain the keep column from flowClass.py.
Output: <prefix>_<mode>_flows.csv    One row per flow (input for Step 3)
        <prefix>_<mode>_summary.csv  One row per (tool, scenario, phase)

Usage:
    python3 buildFeatures.py --mode all  --out-prefix output a.csv b.csv c.csv
    python3 buildFeatures.py --mode keep --out-prefix output a.csv b.csv c.csv

"""
import argparse
import sys
import numpy as np
import pandas as pd

DIRS = ["bidirectional", "src2dst", "dst2src"]
ID = ["tool", "scenario", "phase"]
MIN_PKTS = 3

NEED = set(ID + ["flow_class", "keep", "src_ip", "dst_ip", "dst_port",
                 "bidirectional_first_seen_ms", "bidirectional_bytes", "src2dst_bytes"])
for d in DIRS:
    NEED |= {f"{d}_packets", f"{d}_mean_ps", f"{d}_stddev_ps",
             f"{d}_mean_piat_ms", f"{d}_stddev_piat_ms"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load(paths, mode):
    frames = [pd.read_csv(p, usecols=lambda c: c in NEED, low_memory=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    miss = [c for c in ID if c not in df.columns]
    if miss:
        sys.exit(f"Missing label column(s): {miss}. Run labelFlows.py first.")
    if mode == "keep":
        if "keep" not in df.columns:
            sys.exit("Missing 'keep' column. Run flowClass.py first.")
        df = df[df["keep"].astype(str).str.lower() == "true"].copy()
    return df


def flow_features(df):
    cols = ID + [c for c in ("flow_class", "src_ip", "dst_ip", "dst_port",
                             "bidirectional_first_seen_ms") if c in df.columns]
    out = df[cols].copy()
    out["bidirectional_bytes"] = num(df["bidirectional_bytes"])
    out["src2dst_bytes"] = num(df["src2dst_bytes"])
    out["byte_ratio_src2dst"] = out["src2dst_bytes"] / out["bidirectional_bytes"].where(out["bidirectional_bytes"] > 0)
    for d in DIRS:
        pk = num(df[f"{d}_packets"])
        out[f"{d}_packets"] = pk
        out[f"{d}_mean_ps"] = num(df[f"{d}_mean_ps"]).where(pk >= 1)
        out[f"{d}_std_ps"] = num(df[f"{d}_stddev_ps"]).where(pk >= 1)
        ok = pk >= MIN_PKTS
        mean_p = num(df[f"{d}_mean_piat_ms"]).where(ok)
        std_p = num(df[f"{d}_stddev_piat_ms"]).where(ok)
        out[f"{d}_mean_piat_ms"] = mean_p
        out[f"{d}_std_piat_ms"] = std_p
        out[f"{d}_cv_piat"] = std_p / mean_p.where(mean_p > 0)
    return out


def dist(s, prefix):
    s = s.dropna()
    return {
        f"{prefix}_mean": s.mean(), f"{prefix}_std": s.std(),
        f"{prefix}_p25": s.quantile(.25), f"{prefix}_p50": s.median(),
        f"{prefix}_p75": s.quantile(.75), f"{prefix}_p95": s.quantile(.95),
    }


def gap_stats(ms, prefix):
    keys = ["n", "median_s", "mean_s", "cv", "robust_cv"]
    if len(ms) < 3:
        return {f"{prefix}_{k}": np.nan for k in keys} | {f"{prefix}_n": len(ms)}
    g = np.diff(np.sort(ms.to_numpy(dtype=float))) / 1000.0
    med, mean = np.median(g), g.mean()
    mad = np.median(np.abs(g - med))
    return {
        f"{prefix}_n": len(g), f"{prefix}_median_s": med, f"{prefix}_mean_s": mean,
        f"{prefix}_cv": g.std(ddof=1) / mean if mean > 0 else np.nan,
        f"{prefix}_robust_cv": 1.4826 * mad / med if med > 0 else np.nan,
    }


def summarize(g):
    row = {"n_flows": len(g)}
    for d in DIRS:
        row.update(dist(g[f"{d}_mean_ps"], f"{d}_mean_ps"))
        row.update(dist(g[f"{d}_mean_piat_ms"], f"{d}_mean_piat_ms"))
        cv = g[f"{d}_cv_piat"].dropna()
        row[f"{d}_cv_piat_median"] = cv.median()
        row[f"{d}_cv_piat_mean"] = cv.mean()
        row[f"{d}_n_iat_valid"] = len(cv)
    tot = g["bidirectional_bytes"].sum()
    row["byte_ratio_src2dst"] = g["src2dst_bytes"].sum() / tot if tot > 0 else np.nan
    row.update(gap_stats(g["bidirectional_first_seen_ms"], "gap_all"))
    ch = g.groupby(["src_ip", "dst_ip", "dst_port"], dropna=False).size()
    top = ch.idxmax()
    sub = g[(g["src_ip"] == top[0]) & (g["dst_ip"] == top[1]) & (g["dst_port"] == top[2])]
    row["dom_channel"] = f"{top[0]}>{top[1]}:{top[2]}"
    row["dom_channel_flows"] = len(sub)
    row.update(gap_stats(sub["bidirectional_first_seen_ms"], "gap_dom"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--mode", choices=["all", "keep"], default="all")
    ap.add_argument("--out-prefix", default="step2")
    a = ap.parse_args()

    flows = flow_features(load(a.files, a.mode))
    rows = []
    for key, g in flows.groupby(ID):
        rows.append(dict(zip(ID, key)) | summarize(g))
    summ = pd.DataFrame(rows)

    f_out = f"{a.out_prefix}_{a.mode}_flows.csv"
    s_out = f"{a.out_prefix}_{a.mode}_summary.csv"
    flows.to_csv(f_out, index=False)
    summ.to_csv(s_out, index=False)

    show = ID + ["n_flows", "bidirectional_mean_ps_p50", "bidirectional_mean_piat_ms_p50",
                 "bidirectional_cv_piat_median", "byte_ratio_src2dst",
                 "gap_dom_median_s", "gap_dom_robust_cv"]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summ[show].round(3).to_string(index=False))
    print(f"\nmode={a.mode}  flow={len(flows)}  bucket={len(summ)}")
    print("saved:", f_out, s_out)


if __name__ == "__main__":
    main()
