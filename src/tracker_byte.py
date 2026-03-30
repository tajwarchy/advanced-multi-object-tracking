from pathlib import Path

import numpy as np
import yaml
from boxmot import ByteTrack


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class ByteTrackerWrapper:
    """
    ByteTrack wrapper matching the TrackerWrapper interface exactly.
    Outputs the same list-of-dicts format so all downstream code
    (visualizer, reporter, formatter) works without modification.
    """

    def __init__(self, cfg: dict):
        bt_cfg = cfg["bytetrack"]
        self.tracker = ByteTrack(
            device=bt_cfg["device"],
            track_thresh=bt_cfg["track_thresh"],
            track_buffer=bt_cfg["track_buffer"],
            match_thresh=bt_cfg["match_thresh"],
            half=False,
        )
        self._track_registry: dict[int, dict] = {}
        self._dead_ids: set[int]              = set()
        self.min_hits                          = 3

    def update(self, dets: np.ndarray, frame: np.ndarray,
               frame_id: int) -> list[dict]:
        if dets.shape[0] == 0:
            empty = np.empty((0, 6), dtype=np.float32)
            raw   = self.tracker.update(empty, frame)
        else:
            raw = self.tracker.update(dets, frame)

        tracks    = []
        active_ids = set()

        if raw is not None and len(raw) > 0:
            for row in raw:
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                tid             = int(row[4])
                conf            = float(row[5])
                active_ids.add(tid)

                if tid not in self._track_registry:
                    self._track_registry[tid] = {
                        "first_frame": frame_id,
                        "hit_count"  : 1,
                        "last_seen"  : frame_id,
                    }
                    state = "born"
                else:
                    reg = self._track_registry[tid]
                    reg["hit_count"] += 1
                    reg["last_seen"]  = frame_id
                    state = ("active"
                             if reg["hit_count"] > self.min_hits
                             else "born")

                age = frame_id - self._track_registry[tid]["first_frame"] + 1
                tracks.append({
                    "track_id" : tid,
                    "bbox"     : [x1, y1, x2, y2],
                    "conf"     : conf,
                    "state"    : state,
                    "frame_id" : frame_id,
                    "age"      : age,
                })

        # Lost / dead
        max_age = self.tracker.track_buffer
        for tid, reg in self._track_registry.items():
            if tid in active_ids or tid in self._dead_ids:
                continue
            frames_missing = frame_id - reg["last_seen"]
            if frames_missing > max_age:
                self._dead_ids.add(tid)
                state = "dead"
            else:
                state = "lost"
            tracks.append({
                "track_id" : tid,
                "bbox"     : None,
                "conf"     : 0.0,
                "state"    : state,
                "frame_id" : frame_id,
                "age"      : frame_id - reg["first_frame"] + 1,
            })

        return tracks

    def reset(self):
        self.tracker.reset()
        self._track_registry.clear()
        self._dead_ids.clear()