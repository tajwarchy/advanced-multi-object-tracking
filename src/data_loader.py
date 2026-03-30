import configparser
from pathlib import Path
from typing import Generator, Tuple

import cv2
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class SequenceLoader:
    """
    Iterates frames for a single MOT17 sequence.
    Handles resolution downscaling per config.
    """

    def __init__(self, seq_path: Path, cfg: dict):
        self.seq_path = seq_path
        self.img_dir = seq_path / "img1"
        self.target_res = cfg["detector"]["input_resolution"]

        # Parse seqinfo
        parser = configparser.ConfigParser()
        parser.read(seq_path / "seqinfo.ini")
        info = parser["Sequence"]
        self.fps = float(info["frameRate"])
        self.orig_width = int(info["imWidth"])
        self.orig_height = int(info["imHeight"])
        self.seq_len = int(info["seqLength"])
        self.name = info["name"]

        self.frames = sorted(self.img_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Generator[Tuple[int, any], None, None]:
        """Yields (frame_id, frame_bgr) — frame_id is 1-indexed (MOT convention)."""
        for frame_path in self.frames:
            frame_id = int(frame_path.stem)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            frame = self._resize(frame)
            yield frame_id, frame

    def _resize(self, frame):
        h, w = frame.shape[:2]
        scale = self.target_res / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_LINEAR)
        return frame

    @property
    def scale_factor(self) -> float:
        return self.target_res / max(self.orig_height, self.orig_width)


def get_sequence_paths(cfg: dict) -> list[Path]:
    data_root = Path(cfg["paths"]["data_root"]) / "train"
    return [data_root / s for s in cfg["sequences"]["selected"]]


