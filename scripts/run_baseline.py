import time
from pathlib import Path

from src.data_loader import SequenceLoader, get_sequence_paths, load_config
from src.detector import Detector
from src.eval_formatter import EvalFormatter
from src.reporter import Reporter
from src.tracker_byte import ByteTrackerWrapper
from src.visualizer import Visualizer
from src.video_writer import VideoWriter


def run_sequence_byte(seq_path: Path, detector: Detector,
                      cfg: dict) -> dict:
    loader     = SequenceLoader(seq_path, cfg)
    tracker    = ByteTrackerWrapper(cfg)
    visualizer = Visualizer(cfg)
    reporter   = Reporter(cfg, loader.name)
    formatter  = EvalFormatter(cfg, loader.name, seq_path,
                                tracker_name="ByteTrack")

    first_loader       = SequenceLoader(seq_path, cfg)
    _, first_frame     = next(iter(first_loader))
    h, w               = first_frame.shape[:2]
    frame_size         = (w, h)

    out_video = (Path(cfg["paths"]["videos_dir"])
                 / f"{loader.name}_bytetrack.mp4")

    print(f"\n  Running: {loader.name}  [ByteTrack]")
    print(f"  Frames : {len(loader)} | Resolution: {w}x{h}")

    with VideoWriter(cfg, out_video, loader.fps, frame_size) as vw:
        for frame_id, frame in loader:
            t0   = time.perf_counter()
            dets = detector.detect(frame)
            trks = tracker.update(dets, frame, frame_id)
            vis  = visualizer.draw(frame, trks)
            vw.write(vis)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            reporter.update(trks, frame_id, len(dets), elapsed_ms)
            formatter.update(trks, frame_id)

            if frame_id % 100 == 0:
                print(f"    frame {frame_id:04d}/{len(loader)} "
                      f"| {elapsed_ms:.0f}ms")

    report_path = reporter.save()
    fmt_path    = formatter.save()
    print(f"  Report : {report_path}")
    print(f"  MOT txt: {fmt_path}")

    import json
    with open(report_path) as f:
        data = json.load(f)
    return data["summary"] | {"sequence": loader.name}


def print_summary_table(summaries: list[dict]):
    print(f"\n{'='*80}")
    print(f"{'Sequence':<22} {'Frames':>7} {'IDs':>5} {'AvgActive':>10} "
          f"{'AvgLife':>8} {'IDSw':>5} {'FPS':>6}")
    print(f"{'-'*80}")
    for s in summaries:
        print(f"{s['sequence']:<22} "
              f"{s['total_frames']:>7} "
              f"{s['unique_track_ids']:>5} "
              f"{s['avg_active_per_frame']:>10.1f} "
              f"{s['avg_track_lifetime_frames']:>8.1f} "
              f"{s['id_switch_events']:>5} "
              f"{s['avg_fps']:>6.1f}")
    print(f"{'='*80}\n")


def main():
    cfg      = load_config()
    detector = Detector(cfg)
    paths    = get_sequence_paths(cfg)

    print(f"\nRunning ByteTrack baseline on {len(paths)} sequence(s)...")

    summaries = []
    for seq_path in paths:
        summary = run_sequence_byte(seq_path, detector, cfg)
        summaries.append(summary)

    print_summary_table(summaries)
    print("ByteTrack baseline complete.")
    print(f"Videos  : {cfg['paths']['videos_dir']}")
    print(f"Reports : {cfg['paths']['reports_dir']}\n")


if __name__ == "__main__":
    main()