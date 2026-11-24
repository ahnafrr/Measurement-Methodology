"""
Use:
    python3 labelFlows.py <flows.csv> <manifest.csv> <output.csv> <ip1,ip2,...>

"""
import ipaddress
import sys
import pandas as pd

def to_ms(ts):
    """ISO8601 -> epoch UTC in milliseconds."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.timestamp() * 1000)

def load_manifest(path):
    m = pd.read_csv(path)
    m["start_ms"] = m["start_ts"].map(to_ms)
    m["end_ms"] = m["end_ts"].map(to_ms)
    m["mid_ms"] = (m["start_ms"] + m["end_ms"]) / 2
    return m

def match_window(first_ms, last_ms, manifest):
    hit = manifest[(first_ms <= manifest["end_ms"]) & (last_ms >= manifest["start_ms"])]
    if hit.empty:
        return None, 0
    n = len(hit)
    if n > 1:
        flow_mid = (first_ms + last_ms) / 2
        hit = hit.iloc[[(hit["mid_ms"] - flow_mid).abs().argmin()]]
    return hit.iloc[0], n

def is_loopback(ip):
    try:
        return ipaddress.ip_address(str(ip)).is_loopback
    except ValueError:
        return False


def classify(src, dst, hosts):
    if is_loopback(src) or is_loopback(dst):
        return "loopback"
    if src in hosts and dst in hosts:
        return "lab_traffic"
    return "background"


def main(flows_path, manifest_path, out_path, hosts_arg):
    flows = pd.read_csv(flows_path, low_memory=False)
    manifest = load_manifest(manifest_path)
    hosts = {h.strip() for h in hosts_arg.split(",") if h.strip()}

    kept, dropped, ties = [], 0, 0
    for idx, r in flows.iterrows():
        w, n = match_window(r["bidirectional_first_seen_ms"],
                            r["bidirectional_last_seen_ms"], manifest)
        if w is None:
            dropped += 1  # background noise
            continue
        if n > 1:
            ties += 1
        kept.append((idx, w["tool"], w["scenario"], w["phase"]))

    idx = [k[0] for k in kept]
    out = flows.loc[idx].copy()
    out.insert(0, "phase", [k[3] for k in kept])
    out.insert(0, "scenario", [k[2] for k in kept])
    out.insert(0, "tool", [k[1] for k in kept])

    out["flow_class"] = [classify(s, d, hosts) for s, d in zip(out["src_ip"], out["dst_ip"])]
    out["keep"] = out["flow_class"] == "lab_traffic"
    out.to_csv(out_path, index=False)

    print(f"input flow        : {len(flows)}")
    print(f"label (kept)      : {len(out)}")
    print(f"drop (no window)  : {dropped}")
    print(f"tie-break         : {ties}")
    print("lab hosts         :", sorted(hosts))
    keys = ["tool", "scenario", "phase"]
    tab = out.groupby(keys + ["flow_class"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    print(tab.to_string())
    print(f"\nkeep=True: {int(out['keep'].sum())} from {len(out)}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
