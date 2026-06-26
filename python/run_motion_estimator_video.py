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
    c_ubyte,
    c_void_p,
)

import cv2
import numpy as np


class MotionBBox(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("width", c_float),
        ("height", c_float),
    ]


class MotionResult(Structure):
    _fields_ = [
        ("matrix", c_float * 9),
        ("dx", c_float),
        ("dy", c_float),
        ("unit_x", c_float),
        ("unit_y", c_float),
        ("angle_rad", c_float),
        ("angle_deg", c_float),
        ("scale_x", c_float),
        ("scale_y", c_float),
        ("confidence", c_float),
        ("valid", c_int),
        ("num_features", c_int),
        ("num_tracked", c_int),
        ("num_inliers", c_int),
    ]


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
            os.path.join("..", "motion_camera_tracking", "build", "libMotionEstimator" + ext),
            os.path.join("artifacts", "libMotionEstimator" + ext),
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
        "Could not find MotionEstimator shared library for this platform. "
        f"Expected extension: {ext}\nSearched:\n{searched}"
    )


class MotionEstimator:
    def __init__(self, dll_path, width, height, history_size=5):
        resolved_path = _resolve_library_path(dll_path)

        dll_dir = os.path.abspath(os.path.dirname(resolved_path))
        dll_name = os.path.basename(resolved_path)

        if sys.platform.startswith("win"):
            os.add_dll_directory(dll_dir)
            self.lib = ctypes.CDLL(os.path.join(dll_dir, dll_name))
        else:
            self.lib = ctypes.CDLL(os.path.abspath(resolved_path))

        self.lib.motion_create.argtypes = [c_int, c_int, c_int]
        self.lib.motion_create.restype = c_void_p

        self.lib.motion_destroy.argtypes = [c_void_p]
        self.lib.motion_destroy.restype = None

        self.lib.motion_reset.argtypes = [c_void_p]
        self.lib.motion_reset.restype = None

        self.lib.motion_update.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_int,
            c_int,
            c_int,
            POINTER(MotionBBox),
            c_size_t,
            POINTER(MotionResult),
        ]
        self.lib.motion_update.restype = c_int

        self.handle = self.lib.motion_create(int(width), int(height), int(history_size))
        if not self.handle:
            raise RuntimeError("motion_create failed")

    def reset(self):
        self.lib.motion_reset(self.handle)

    def update(self, frame_bgr, ignore_boxes):
        frame = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        h, w = frame.shape[:2]
        stride = int(frame.strides[0])

        frame_ptr = frame.ctypes.data_as(POINTER(c_ubyte))

        box_count = len(ignore_boxes)
        if box_count > 0:
            boxes_arr = (MotionBBox * box_count)(*ignore_boxes)
        else:
            boxes_arr = None

        out = MotionResult()

        ret = self.lib.motion_update(
            self.handle,
            frame_ptr,
            c_int(w),
            c_int(h),
            c_int(stride),
            boxes_arr,
            c_size_t(box_count),
            byref(out),
        )

        if ret != 0:
            raise RuntimeError(f"motion_update failed with return code {ret}")

        return out

    def close(self):
        if getattr(self, "handle", None):
            self.lib.motion_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def load_detection_boxes_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    boxes_by_frame = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {"frame", "x", "y", "w", "h"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for row in reader:
            frame_id = int(float(row["frame"]))
            boxes_by_frame[frame_id].append(
                MotionBBox(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    width=float(row["w"]),
                    height=float(row["h"]),
                )
            )

    return boxes_by_frame


def draw_ignore_boxes(frame, boxes):
    for b in boxes:
        x1 = int(round(b.x))
        y1 = int(round(b.y))
        x2 = int(round(b.x + b.width))
        y2 = int(round(b.y + b.height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)


def draw_motion_overlay(frame, result, trail_points):
    h, w = frame.shape[:2]

    center = (w // 2, h // 2)
    arrow_scale = 8.0
    tip = (
        int(round(center[0] + result.dx * arrow_scale)),
        int(round(center[1] + result.dy * arrow_scale)),
    )

    color = (0, 200, 0) if result.valid else (0, 0, 255)
    cv2.arrowedLine(frame, center, tip, color, 2, cv2.LINE_AA, tipLength=0.2)

    if trail_points:
        for i in range(1, len(trail_points)):
            cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 180, 255), 1, cv2.LINE_AA)

    line1 = (
        f"dx={result.dx:+.2f} dy={result.dy:+.2f} "
        f"ang={result.angle_deg:+.2f}deg conf={result.confidence:.2f} valid={result.valid}"
    )
    line2 = (
        f"feat={result.num_features} tracked={result.num_tracked} inliers={result.num_inliers} "
        f"scale=({result.scale_x:.3f},{result.scale_y:.3f})"
    )

    cv2.putText(frame, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)


def run(
    video_path,
    detections_csv_path,
    dll_path,
    output_path,
    detection_interval=1,
    history_size=5,
    draw_boxes=True,
):
    detection_interval = max(1, int(detection_interval))

    print("Loading detection boxes from CSV...")
    boxes_by_frame = load_detection_boxes_csv(detections_csv_path)
    print("Boxes loaded.")

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

    print(f"Initializing MotionEstimator with DLL: {dll_path}")
    estimator = MotionEstimator(
        dll_path=dll_path,
        width=width,
        height=height,
        history_size=history_size,
    )

    frame_idx = 0
    trail_points = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % detection_interval == 0:
                ignore_boxes = boxes_by_frame.get(frame_idx, [])
            else:
                ignore_boxes = []

            result = estimator.update(frame, ignore_boxes)

            if draw_boxes:
                draw_ignore_boxes(frame, ignore_boxes)

            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            plot_scale = 2.0
            pt = (
                int(round(center[0] + result.dx * plot_scale)),
                int(round(center[1] + result.dy * plot_scale)),
            )
            trail_points.append(pt)
            if len(trail_points) > 40:
                trail_points.pop(0)

            draw_motion_overlay(frame, result, trail_points)

            cv2.putText(
                frame,
                f"Frame {frame_idx}/{total_frames} | IgnoreBoxes {len(ignore_boxes)} | BoxEvery {detection_interval}",
                (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
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
        estimator.close()

    print(f"Done. Saved output video to: {output_path}")


if __name__ == "__main__":
    run(
        video_path="data/input.MOV",
        detections_csv_path="data/detections.csv",
        dll_path="../motion_camera_tracking/build/libMotionEstimator",
        output_path="output/motion_estimated.mp4",
        detection_interval=2,
        history_size=5,
        draw_boxes=True,
    )
