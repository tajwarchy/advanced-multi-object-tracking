from pathlib import Path

import configparser

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class EvalFormatter:
    """
    Writes tracker output in MOT Challenge submission format.
    Rescales coordinates from inference resolution back to original resolution.
        frame, id, x, y, w, h, conf, -1, -1, -1
    """

    def __init__(self, cfg: dict, seq_name: str, seq_path: Path,
             tracker_name: str = "StrongSORT"):
        
        tracker_dir = (Path(cfg["paths"]["eval_dir"])
                        / "trackers" / tracker_name / "data")
        tracker_dir.mkdir(parents=True, exist_ok=True)
        self._path = tracker_dir / f"{seq_name}.txt"
        self._rows: list[str] = []

        # Read original resolution from seqinfo.ini
        ini = configparser.ConfigParser()
        ini.read(seq_path / "seqinfo.ini")
        orig_w = int(ini["Sequence"]["imWidth"])
        orig_h = int(ini["Sequence"]["imHeight"])

        # Compute scale factor — same logic as SequenceLoader._resize()
        target_res = cfg["detector"]["input_resolution"]
        self._scale = 1.0 / (target_res / max(orig_w, orig_h))

    def update(self, tracks: list, frame_id: int):
        for t in tracks:
            if t["bbox"] is None:
                continue
            x1, y1, x2, y2 = t["bbox"]

            # Rescale back to original resolution
            x1 = int(x1 * self._scale)
            y1 = int(y1 * self._scale)
            x2 = int(x2 * self._scale)
            y2 = int(y2 * self._scale)

            w    = x2 - x1
            h    = y2 - y1
            conf = round(t["conf"], 4)
            tid  = t["track_id"]
            self._rows.append(
                f"{frame_id},{tid},{x1},{y1},{w},{h},{conf},-1,-1,-1"
            )

    def save(self) -> Path:
        with open(self._path, "w") as f:
            f.write("\n".join(self._rows))
        return self._path