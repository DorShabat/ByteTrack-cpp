import csv
import ctypes
import os
import sys
from collections import defaultdict
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint8,
    c_void_p,
    byref,
)

import cv2
import numpy as np


# ============================================================
# C structs matching cvtracker_c.h
# ============================================================


class CVTrackerObject(Structure):
    """Input detection (matches cvtracker_object_t)."""
    _fields_ = [
        ("x",     c_float),
        ("y",     c_float),
        ("width", c_float),
        ("height",c_float),
        ("score", c_float),
        ("label", c_int),
    ]


class CVTrackerTrack(Structure):
    """Output track (matches cvtracker_track_t)."""
    _fields_ = [
        ("x",        c_float),
        ("y",        c_float),
        ("width",    c_float),
        ("height",   c_float),
        ("score",    c_float),
        ("track_id", c_int),
        ("is_active",c_int),   # 1 = cv::Tracker success, 0 = kept-alive only
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

    candidates.extend(
        [
            os.path.join("..", "tracker", "build", "libcvtracker" + ext),
            os.path.join("artifacts",   "libcvtracker" + ext),
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
        "Could not find cvtracker shared library for this platform. "
        f"Expected extension: {ext}\nSearched:\n{searched}"
    )


# ============================================================
# CVTracker wrapper
# ============================================================


class CVTracker:
    def __init__(
        self,
        dll_path,
        tracker_type="CSRT",   # "CSRT" (quality) or "KCF" (speed)
        iou_thresh=0.3,
        lost_ttl=5,
    ):
        resolved_path = _resolve_library_path(dll_path)

        dll_dir  = os.path.abspath(os.path.dirname(resolved_path))
        dll_name = os.path.basename(resolved_path)

        if sys.platform.startswith("win"):
            os.add_dll_directory(dll_dir)
            self.lib = ctypes.CDLL(os.path.join(dll_dir, dll_name))
        else:
            self.lib = ctypes.CDLL(os.path.abspath(resolved_path))

        self.lib.cvtracker_create.argtypes = [c_char_p, c_float, c_int]
        self.lib.cvtracker_create.restype  = c_void_p

        self.lib.cvtracker_destroy.argtypes = [c_void_p]
        self.lib.cvtracker_destroy.restype  = None

        self.lib.cvtracker_update.argtypes = [
            c_void_p,
            POINTER(c_uint8),          # frame_data
            c_int,                     # width
            c_int,                     # height
            POINTER(CVTrackerObject),  # objects
            c_size_t,                  # object_count
            POINTER(CVTrackerTrack),   # out_tracks
            c_size_t,                  # out_capacity
            POINTER(c_size_t),         # out_count
        ]
        self.lib.cvtracker_update.restype = c_int

        self.handle = self.lib.cvtracker_create(
            tracker_type.encode(),
            float(iou_thresh),
            int(lost_ttl),
        )

        if not self.handle:
            raise RuntimeError("cvtracker_create failed")

    def update(self, frame_bgr, detections):
        """
        frame_bgr  : numpy array (H, W, 3) uint8 BGR – always required.
        detections : list of CVTrackerObject (may be empty on skipped frames).
        """
        # Ensure contiguous BGR bytes.
        frame_c = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        h, w    = frame_c.shape[:2]
        frame_ptr = frame_c.ctypes.data_as(POINTER(c_uint8))

        det_count = len(detections)
        if det_count > 0:
            objects_array = (CVTrackerObject * det_count)(*detections)
        else:
            objects_array = None

        out_capacity = max(128, det_count * 4)
        out_tracks   = (CVTrackerTrack * out_capacity)()
        out_count    = c_size_t(0)

        ret = self.lib.cvtracker_update(
            self.handle,
            frame_ptr,
            c_int(w),
            c_int(h),
            objects_array,
            c_size_t(det_count),
            out_tracks,
            c_size_t(out_capacity),
            byref(out_count),
        )

        if ret != 0:
            raise RuntimeError(f"cvtracker_update failed with return code {ret}")

        return list(out_tracks[: out_count.value])

    def close(self):
        if getattr(self, "handle", None):
            self.lib.cvtracker_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


# ============================================================
# CSV loader (same format as run_bytetrack_video.py)
# frame,x,y,w,h,score,class_id
# ============================================================


def load_detections_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    detections_by_frame = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {"frame", "x", "y", "w", "h", "score"}
        missing  = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for row in reader:
            frame_id = int(float(row["frame"]))
            obj = CVTrackerObject(
                x=float(row["x"]),
                y=float(row["y"]),
                width=float(row["w"]),
                height=float(row["h"]),
                score=float(row["score"]),
                label=int(float(row.get("class_id", 0) or 0)),
            )
            detections_by_frame[frame_id].append(obj)

    return detections_by_frame


# ============================================================
# Draw
# ============================================================


def draw_detections(frame, detections):
    """Cyan box + label below the box."""
    for d in detections:
        x1 = int(round(d.x))
        y1 = int(round(d.y))
        x2 = int(round(d.x + d.width))
        y2 = int(round(d.y + d.height))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        det_text_y = min(frame.shape[0] - 5, y2 + 14)
        cv2.putText(
            frame,
            f"det {d.score:.2f}",
            (x1, det_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def draw_tracks(frame, tracks):
    """
    Green  (solid, thick)  = actively tracked by cv::Tracker.
    Orange (dashed, thin)  = cv::Tracker lost it but track still alive.
    Label above the box in both cases.
    """
    active_count = 0
    lost_count   = 0

    for t in tracks:
        x1 = int(round(t.x))
        y1 = int(round(t.y))
        x2 = int(round(t.x + t.width))
        y2 = int(round(t.y + t.height))

        if t.is_active:
            active_count += 1
            color     = (0, 200, 0)   # green
            thickness = 2
            label     = f"ID {t.track_id} {t.score:.2f}"
        else:
            lost_count += 1
            color     = (0, 140, 255)  # orange
            thickness = 1
            label     = f"ID {t.track_id} lost"

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

    return active_count, lost_count


# ============================================================
# Main processing
# ============================================================


def run(
    video_path,
    csv_path,
    dll_path,
    output_path,
    tracker_type="CSRT",     # "CSRT" or "KCF"
    iou_thresh=0.3,
    lost_ttl=5,
    detection_interval=1,    # run detector every N frames
    draw_dets=True,
    draw_tracks_flag=True,
):
    detection_interval = max(1, int(detection_interval))

    print("Loading detections from CSV...")
    detections_by_frame = load_detections_csv(csv_path)
    print("Detections loaded.")

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    print(f"Initializing CVTracker ({tracker_type}) with DLL: {dll_path}")
    tracker = CVTracker(
        dll_path=dll_path,
        tracker_type=tracker_type,
        iou_thresh=iou_thresh,
        lost_ttl=lost_ttl,
    )

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detections only on scheduled frames.
            if frame_idx % detection_interval == 0:
                detections = detections_by_frame.get(frame_idx, [])
            else:
                detections = []

            # cvtracker always receives the frame (cv::Tracker needs pixels).
            tracks = tracker.update(frame, detections)

            if draw_dets:
                draw_detections(frame, detections)

            if draw_tracks_flag:
                active_count, lost_count = draw_tracks(frame, tracks)
            else:
                active_count = lost_count = 0

            cv2.putText(
                frame,
                (
                    f"Frame {frame_idx}/{total_frames} | "
                    f"Dets {len(detections)} | "
                    f"Tracked {active_count} | "
                    f"Lost {lost_count} | "
                    f"DetEvery {detection_interval} | "
                    f"{tracker_type}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
        dll_path="../tracker/build/libcvtracker",  # built from tracker/CMakeLists.txt
        output_path="output/cvtracked.mp4",
        tracker_type="CSRT",    # "CSRT" (best quality) or "KCF" (faster)
        iou_thresh=0.3,
        lost_ttl=5,             # frames to keep a track alive after cv::Tracker fails
        detection_interval=30,   # run detector every N frames; cv::Tracker fills the gaps
    )
