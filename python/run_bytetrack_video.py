import csv
import ctypes
import os
import sys
from collections import defaultdict
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_float,
    c_int,
    c_size_t,
    c_void_p,
)

import cv2


TRACK_STATE_TRACKED = 1
TRACK_STATE_LOST = 2


# ============================================================
# C structs matching bytetrack_c.h
# ============================================================


class ByteTrackObject(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("width", c_float),
        ("height", c_float),
        ("label", c_int),
        ("prob", c_float),
    ]


class ByteTrackTrack(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("width", c_float),
        ("height", c_float),
        ("score", c_float),
        ("track_id", c_size_t),
        ("frame_id", c_size_t),
        ("start_frame_id", c_size_t),
        ("tracklet_length", c_size_t),
        ("is_activated", c_int),
        ("state", c_int),
    ]


# ============================================================
# Shared library resolution
# ============================================================


def _platform_library_extension():
    if sys.platform.startswith("win"):
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _resolve_library_path(library_path):
    ext = _platform_library_extension()
    candidates = []

    if library_path:
        candidates.append(library_path)
        root, old_ext = os.path.splitext(library_path)
        if old_ext:
            if old_ext.lower() != ext:
                candidates.append(root + ext)
        else:
            candidates.append(library_path + ext)

    # Common local locations when running from python/
    candidates.extend(
        [
            os.path.join("artifacts", "libbytetrack" + ext),
            os.path.join("..", "build", "libbytetrack" + ext),
            os.path.join("..", "build-win", "libbytetrack" + ext),
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        abs_candidate = os.path.abspath(candidate)
        if os.path.exists(abs_candidate):
            return abs_candidate

    searched = "\n".join(f"  - {os.path.abspath(p)}" for p in candidates if p)
    raise FileNotFoundError(
        "Could not find ByteTrack shared library for this platform. "
        f"Expected extension: {ext}\nSearched:\n{searched}"
    )


# ============================================================
# ByteTrack wrapper
# ============================================================


class ByteTracker:
    def __init__(
        self,
        dll_path,
        frame_rate=30,
        track_buffer=30,
        track_thresh=0.5,
        high_thresh=0.6,
        match_thresh=0.8,
    ):
        resolved_path = _resolve_library_path(dll_path)

        dll_dir = os.path.abspath(os.path.dirname(resolved_path))
        dll_name = os.path.basename(resolved_path)

        if sys.platform.startswith("win"):
            os.add_dll_directory(dll_dir)
            self.lib = ctypes.CDLL(os.path.join(dll_dir, dll_name))
        else:
            self.lib = ctypes.CDLL(os.path.abspath(resolved_path))

        self.lib.bytetrack_create.argtypes = [c_int, c_int, c_float, c_float, c_float]
        self.lib.bytetrack_create.restype = c_void_p

        self.lib.bytetrack_destroy.argtypes = [c_void_p]
        self.lib.bytetrack_destroy.restype = None

        self.lib.bytetrack_update.argtypes = [
            c_void_p,
            POINTER(ByteTrackObject),
            c_size_t,
            POINTER(ByteTrackTrack),
            c_size_t,
            POINTER(c_size_t),
        ]
        self.lib.bytetrack_update.restype = c_int

        self.handle = self.lib.bytetrack_create(
            int(frame_rate),
            int(track_buffer),
            float(track_thresh),
            float(high_thresh),
            float(match_thresh),
        )

        if not self.handle:
            raise RuntimeError("bytetrack_create failed")

    def update(self, detections):
        object_count = len(detections)

        if object_count > 0:
            objects_array = (ByteTrackObject * object_count)(*detections)
        else:
            objects_array = None

        out_capacity = max(128, object_count * 4)
        out_tracks = (ByteTrackTrack * out_capacity)()
        out_count = c_size_t(0)

        ret = self.lib.bytetrack_update(
            self.handle,
            objects_array,
            c_size_t(object_count),
            out_tracks,
            c_size_t(out_capacity),
            byref(out_count),
        )

        if ret == 1:
            out_capacity = out_count.value
            out_tracks = (ByteTrackTrack * out_capacity)()
            out_count = c_size_t(0)

            ret = self.lib.bytetrack_update(
                self.handle,
                objects_array,
                c_size_t(object_count),
                out_tracks,
                c_size_t(out_capacity),
                byref(out_count),
            )

        if ret != 0:
            raise RuntimeError(f"bytetrack_update failed with return code {ret}")

        return list(out_tracks[: out_count.value])

    def close(self):
        if getattr(self, "handle", None):
            self.lib.bytetrack_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


# ============================================================
# CSV loader
# Expected CSV:
# frame,x,y,w,h,score,class_id
# ============================================================


def load_detections_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    detections_by_frame = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {"frame", "x", "y", "w", "h", "score"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for row in reader:
            frame_id = int(float(row["frame"]))

            obj = ByteTrackObject(
                x=float(row["x"]),
                y=float(row["y"]),
                width=float(row["w"]),
                height=float(row["h"]),
                label=int(float(row.get("class_id", 0) or 0)),
                prob=float(row["score"]),
            )

            detections_by_frame[frame_id].append(obj)

    return detections_by_frame


# ============================================================
# Draw
# ============================================================


def draw_detections(frame, detections):
    for d in detections:
        x1 = int(round(d.x))
        y1 = int(round(d.y))
        x2 = int(round(d.x + d.width))
        y2 = int(round(d.y + d.height))

        # Detections (from detector/NN): cyan, text below the box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        det_text_y = min(frame.shape[0] - 5, y2 + 14)

        cv2.putText(
            frame,
            f"det {d.prob:.2f}",
            (x1, det_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def draw_tracks(frame, tracks, current_frame_idx):
    tracked_count = 0
    predicted_count = 0

    for t in tracks:
        x1 = int(round(t.x))
        y1 = int(round(t.y))
        x2 = int(round(t.x + t.width))
        y2 = int(round(t.y + t.height))

        if t.state == TRACK_STATE_LOST:
            predicted_count += 1
            color = (0, 165, 255)  # orange
            thickness = 1
            lost_age = max(0, int(current_frame_idx - t.frame_id))
            label = f"ID {t.track_id} pred +{lost_age}"
        else:
            tracked_count += 1
            color = (0, 0, 255)
            thickness = 2
            label = f"ID {t.track_id} {t.score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return tracked_count, predicted_count


# ============================================================
# Main processing
# ============================================================


def run(
    video_path,
    csv_path,
    dll_path,
    output_path,
    detection_interval=1,
    draw_dets=True,
    draw_tracks_flag=True,
):
    detection_interval = max(1, int(detection_interval))
    # Keep predicted output across skipped detector frames.
    os.environ["BYTETRACK_LOST_OUTPUT_TTL"] = str(max(1, detection_interval - 1))

    print("trying to load detections from CSV...")
    detections_by_frame = load_detections_csv(csv_path)
    print("detections loaded successfully.")

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    print(f"Initializing tracker with DLL: {dll_path}")
    tracker = ByteTracker(
        dll_path=dll_path,
        frame_rate=int(round(fps)),
        track_buffer=30,
        track_thresh=0.5,
        high_thresh=0.6,
        match_thresh=0.8,
    )

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Emulates running detector every N frames.
            if frame_idx % detection_interval == 0:
                detections = detections_by_frame.get(frame_idx, [])
            else:
                detections = []

            tracks = tracker.update(detections)

            if draw_dets:
                draw_detections(frame, detections)

            if draw_tracks_flag:
                tracked_count, predicted_count = draw_tracks(frame, tracks, frame_idx)
            else:
                tracked_count, predicted_count = 0, 0

            cv2.putText(
                frame,
                f"Frame {frame_idx}/{total_frames} | Dets {len(detections)} | Tracked {tracked_count} | Pred {predicted_count} | DetEvery {detection_interval}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

            if frame_idx % 100 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")

            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        tracker.close()

    print(f"Done. Saved output video to: {output_path}")


# ============================================================
# Edit these paths and run
# ============================================================

if __name__ == "__main__":
    run(
        video_path="data/input.MOV",
        csv_path="data/detections.csv",
        dll_path="../build/libbytetrack",  # load rebuilt library directly
        output_path="output/tracked.mp4",
        detection_interval=10,  # try: 2, 4, or any X>=1
    )
