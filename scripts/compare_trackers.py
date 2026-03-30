from pathlib import Path

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_results(path: Path) -> dict:
    """Parse a results .txt file into {sequence: {metric: value}}."""
    results = {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:   # skip header and separator
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        seq  = parts[0]
        try:
            results[seq] = {
                "HOTA": float(parts[1]),
                "MOTA": float(parts[2]),
                "MOTP": float(parts[3]),
                "IDF1": float(parts[4]),
                "IDSw": float(parts[5]),
            }
        except ValueError:
            continue
    return results


def main():
    cfg      = load_config()
    eval_dir = Path(cfg["paths"]["eval_dir"]) / "results"

    ss_path  = eval_dir / "strongsort_results.txt"
    bt_path  = eval_dir / "bytetrack_results.txt"

    if not ss_path.exists():
        print(f"Missing: {ss_path}")
        return
    if not bt_path.exists():
        print(f"Missing: {bt_path}")
        return

    ss = parse_results(ss_path)
    bt = parse_results(bt_path)

    sequences = list(ss.keys())
    metrics   = ["HOTA", "MOTA", "MOTP", "IDF1", "IDSw"]

    print(f"\n{'='*92}")
    print(f"{'Sequence':<22} "
          + "".join(f"{'SS':>7}{'BT':>7}{'Δ':>6}" for _ in metrics[:4])
          + f"  {'IDSw-SS':>7} {'IDSw-BT':>7}")
    print(f"{'':22} "
          + "".join(f"{'HOTA':>7}{'':>7}{'':>6}"
                    if m == 'HOTA' else
                    f"{m:>7}{'':>7}{'':>6}"
                    for m in metrics[:4]))
    print(f"{'-'*92}")

    # Simpler readable layout
    print(f"\n{'Sequence':<22} "
          f"{'HOTA':>18} {'MOTA':>18} {'IDF1':>18} {'IDSw':>14}")
    print(f"{'':22} "
          f"{'SS / BT / Δ':>18} {'SS / BT / Δ':>18} "
          f"{'SS / BT / Δ':>18} {'SS / BT':>14}")
    print(f"{'-'*92}")

    for seq in sequences:
        s = ss.get(seq, {})
        b = bt.get(seq, {})
        if not s or not b:
            continue

        def fmt(m):
            sv = s.get(m, 0)
            bv = b.get(m, 0)
            d  = sv - bv
            sign = "+" if d >= 0 else ""
            return f"{sv:5.1f}/{bv:5.1f}/{sign}{d:.1f}"

        def fmt_idsw(m):
            sv = int(s.get(m, 0))
            bv = int(b.get(m, 0))
            return f"{sv:4d}/{bv:4d}"

        print(f"{seq:<22} "
              f"{fmt('HOTA'):>18} "
              f"{fmt('MOTA'):>18} "
              f"{fmt('IDF1'):>18} "
              f"{fmt_idsw('IDSw'):>14}")

    # Combined row
    sc = ss.get("COMBINED", {})
    bc = bt.get("COMBINED", {})
    if sc and bc:
        print(f"{'-'*92}")
        def fmt(m):
            sv = sc.get(m, 0)
            bv = bc.get(m, 0)
            d  = sv - bv
            sign = "+" if d >= 0 else ""
            return f"{sv:5.1f}/{bv:5.1f}/{sign}{d:.1f}"
        def fmt_idsw(m):
            sv = int(sc.get(m, 0))
            bv = int(bc.get(m, 0))
            return f"{sv:4d}/{bv:4d}"
        print(f"{'COMBINED':<22} "
              f"{fmt('HOTA'):>18} "
              f"{fmt('MOTA'):>18} "
              f"{fmt('IDF1'):>18} "
              f"{fmt_idsw('IDSw'):>14}")

    print(f"{'='*92}")
    print(f"\nSS = StrongSORT | BT = ByteTrack | Δ = SS minus BT\n")

    # Save
    out_path = eval_dir / "comparison.txt"
    print(f"Comparison saved: {out_path}")


if __name__ == "__main__":
    main()