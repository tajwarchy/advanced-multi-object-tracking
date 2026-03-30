import os
import configparser
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def inspect_sequence(seq_path: Path) -> dict:
    # Parse seqinfo.ini
    ini_path = seq_path / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    info = parser["Sequence"]

    seq_len = int(info["seqLength"])
    fps = float(info["frameRate"])
    img_width = int(info["imWidth"])
    img_height = int(info["imHeight"])

    # Count actual frames
    img_dir = seq_path / "img1"
    frames = sorted(img_dir.glob("*.jpg"))
    actual_frames = len(frames)

    # Inspect GT
    gt_path = seq_path / "gt" / "gt.txt"
    gt_lines = 0
    unique_ids = set()
    if gt_path.exists():
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                # MOT GT format: frame, id, x, y, w, h, conf, cls, vis
                if int(parts[6]) == 1:  # conf=1 means active annotation
                    gt_lines += 1
                    unique_ids.add(int(parts[1]))

    return {
        "sequence": seq_path.name,
        "declared_length": seq_len,
        "actual_frames": actual_frames,
        "fps": fps,
        "resolution": f"{img_width}x{img_height}",
        "gt_annotations": gt_lines,
        "unique_gt_ids": len(unique_ids),
    }


def main():
    cfg = load_config()
    data_root = Path(cfg["paths"]["data_root"]) / "train"
    selected = cfg["sequences"]["selected"]

    print(f"\n{'='*65}")
    print(f"{'Sequence':<22} {'Frames':>7} {'FPS':>5} {'Resolution':>12} "
          f"{'GT Ann':>8} {'GT IDs':>7}")
    print(f"{'='*65}")

    for seq_name in selected:
        seq_path = data_root / seq_name
        if not seq_path.exists():
            print(f"{seq_name:<22}  NOT FOUND")
            continue
        s = inspect_sequence(seq_path)
        print(f"{s['sequence']:<22} {s['actual_frames']:>7} {s['fps']:>5.1f} "
              f"{s['resolution']:>12} {s['gt_annotations']:>8} "
              f"{s['unique_gt_ids']:>7}")

    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()