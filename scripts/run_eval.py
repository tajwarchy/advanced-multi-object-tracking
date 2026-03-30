import os
import sys
from pathlib import Path

import numpy as np
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tracker", default="StrongSORT",
                    choices=["StrongSORT", "ByteTrack"])
args = parser.parse_args()
tracker_name = args.tracker

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    cfg       = load_config()
    eval_cfg  = cfg["eval"]
    eval_dir  = Path(cfg["paths"]["eval_dir"])

    trackeval_root = Path(eval_cfg["trackeval_root"]).expanduser()
    if not trackeval_root.exists():
        print(f"TrackEval not found at: {trackeval_root}")
        print("Update eval.trackeval_root in config/config.yaml")
        sys.exit(1)

    # Add TrackEval to path
    sys.path.insert(0, str(trackeval_root))

    # Compatibility patch for old TrackEval code using deprecated NumPy aliases
    import numpy as np
    for alias, replacement in {
        'int': int,
        'float': float,
        'bool': bool,
        'object': object,
    }.items():
        if not hasattr(np, alias):
            setattr(np, alias, replacement)

    import trackeval

    # ── Paths ────────────────────────────────────────────────────
    gt_dir      = eval_dir / "gt"
    tracker_dir = eval_dir / "trackers"
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sequences = cfg["sequences"]["selected"]

    # ── Dataset config ───────────────────────────────────────────
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
    "GT_FOLDER"          : str(gt_dir / "MOT17" / "MOT17-train"),
    "TRACKERS_FOLDER"    : str(tracker_dir),
    "OUTPUT_FOLDER"      : str(results_dir),
    "BENCHMARK"          : "MOT17",
    "SPLIT_TO_EVAL"      : "train",
    "TRACKERS_TO_EVAL"   : [tracker_name],
    "CLASSES_TO_EVAL"    : ["pedestrian"],
    "PRINT_CONFIG"       : False,
    "OUTPUT_SUMMARY"     : True,
    "OUTPUT_DETAILED"    : True,
    "PRINT_ONLY_COMBINED": False,
    "SKIP_SPLIT_FOL"     : True,
    "SEQ_INFO"           : {
        "MOT17-02-SDP": None,
        "MOT17-04-SDP": None,
        "MOT17-05-SDP": None,
        "MOT17-09-SDP": None,
        "MOT17-10-SDP": None,
    },
})

    # ── Metrics config ───────────────────────────────────────────
    metrics_config = {"METRICS": eval_cfg["metrics"],
                      "THRESHOLD": 0.5}

    # ── Evaluator config ─────────────────────────────────────────
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL"    : False,       # macOS — no multiprocessing
        "NUM_PARALLEL_CORES": 1,
        "PRINT_RESULTS"   : True,
        "PRINT_CONFIG"    : False,
        "TIME_PROGRESS"   : True,
        "OUTPUT_EMPTY_CLASSES": False,
        "LOG_ON_ERROR"    : str(results_dir / "error_log.txt"),
    })

    # ── Run ──────────────────────────────────────────────────────
    print("\nRunning TrackEval...\n")

    evaluator   = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []

    for m in eval_cfg["metrics"]:
        if m == "HOTA":
            metrics_list.append(
                trackeval.metrics.HOTA(metrics_config))
        elif m == "CLEAR":
            metrics_list.append(
                trackeval.metrics.CLEAR(metrics_config))
        elif m == "Identity":
            metrics_list.append(
                trackeval.metrics.Identity(metrics_config))

    results, _ = evaluator.evaluate(dataset_list, metrics_list)
    

    # ── Extract and print clean results table ────────────────────
    _print_results_table(results, sequences, eval_dir)


def _print_results_table(results: dict, sequences: list, eval_dir: Path):
    try:
        tracker_res = results["MotChallenge2DBox"][tracker_name]
    except KeyError:
        print("Could not parse results dict — check error log.")
        return

    metrics_keys = [
        ("HOTA",     "HOTA",   "HOTA"),
        ("CLEAR",    "MOTA",   "MOTA"),
        ("CLEAR",    "MOTP",   "MOTP"),
        ("Identity", "IDF1",   "IDF1"),
        ("CLEAR",    "IDSW",   "IDSw"),
    ]

    def extract(seq_data, metric_cls, key):
        val = seq_data.get(metric_cls, {}).get(key, None)
        if val is None:
            return "-"
        if hasattr(val, "__len__"):
            scalar = float(val.mean())
            # HOTA sub-metrics come back as 0-1, MOTA/IDF1 already 0-100
            scalar = scalar * 100 if scalar <= 1.0 else scalar
        else:
            scalar = float(val)
            scalar = scalar * 100 if scalar <= 1.0 else scalar
        return round(scalar, 2)

    rows = []
    for seq in sequences:
        seq_data = tracker_res.get(seq, {}).get("pedestrian", {})
        if not seq_data:
            continue
        row = {"Sequence": seq}
        for metric_cls, key, label in metrics_keys:
            row[label] = extract(seq_data, metric_cls, key)
        rows.append(row)

    combined = tracker_res.get("COMBINED_SEQ", {}).get("pedestrian", {})
    if combined:
        row = {"Sequence": "COMBINED"}
        for metric_cls, key, label in metrics_keys:
            row[label] = extract(combined, metric_cls, key)
        rows.append(row)

    # Print
    print(f"\n{'='*72}")
    print(f"{'Sequence':<22} {'HOTA':>7} {'MOTA':>7} {'MOTP':>7} "
          f"{'IDF1':>7} {'IDSw':>6}")
    print(f"{'-'*72}")
    for row in rows:
        if row["Sequence"] == "COMBINED":
            print(f"{'-'*72}")
        print(f"{row['Sequence']:<22} "
              f"{str(row.get('HOTA','-')):>7} "
              f"{str(row.get('MOTA','-')):>7} "
              f"{str(row.get('MOTP','-')):>7} "
              f"{str(row.get('IDF1','-')):>7} "
              f"{str(row.get('IDSw','-')):>6}")
    print(f"{'='*72}\n")

    # Save
    out_path = eval_dir / "results" / f"{tracker_name.lower()}_results.txt"
    with open(out_path, "w") as f:
        f.write(f"{'Sequence':<22} {'HOTA':>7} {'MOTA':>7} {'MOTP':>7} "
                f"{'IDF1':>7} {'IDSw':>6}\n")
        f.write("-" * 72 + "\n")
        for row in rows:
            f.write(f"{row['Sequence']:<22} "
                    f"{str(row.get('HOTA','-')):>7} "
                    f"{str(row.get('MOTA','-')):>7} "
                    f"{str(row.get('MOTP','-')):>7} "
                    f"{str(row.get('IDF1','-')):>7} "
                    f"{str(row.get('IDSw','-')):>6}\n")
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()