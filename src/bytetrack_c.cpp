#include "ByteTrack/bytetrack_c.h"

#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/Object.h"
#include "ByteTrack/Rect.h"
#include "ByteTrack/STrack.h"

#include <algorithm>
#include <memory>
#include <vector>

struct bytetrack_handle {
  std::unique_ptr<byte_track::BYTETracker> impl;
};

namespace {

void fill_track(const byte_track::STrack &src, bytetrack_track_t &dst) {
  const auto &rect = src.getRect();
  dst.x = rect.x();
  dst.y = rect.y();
  dst.width = rect.width();
  dst.height = rect.height();
  dst.score = src.getScore();
  dst.track_id = src.getTrackId();
  dst.frame_id = src.getFrameId();
  dst.start_frame_id = src.getStartFrameId();
  dst.tracklet_length = src.getTrackletLength();
  dst.is_activated = src.isActivated() ? 1 : 0;
  dst.state = static_cast<int>(src.getSTrackState());
}

}  // namespace

bytetrack_handle *bytetrack_create(int frame_rate,
                                   int track_buffer,
                                   float track_thresh,
                                   float high_thresh,
                                   float match_thresh) {
  try {
    bytetrack_handle *handle = new bytetrack_handle();
    handle->impl = std::make_unique<byte_track::BYTETracker>(
        frame_rate, track_buffer, track_thresh, high_thresh, match_thresh);
    return handle;
  } catch (...) {
    return nullptr;
  }
}

void bytetrack_destroy(bytetrack_handle *handle) {
  delete handle;
}

int bytetrack_update(bytetrack_handle *handle,
                     const bytetrack_object_t *objects,
                     size_t object_count,
                     bytetrack_track_t *out_tracks,
                     size_t out_capacity,
                     size_t *out_count) {
  if (handle == nullptr || handle->impl == nullptr || out_count == nullptr) {
    return -1;
  }
  if (objects == nullptr && object_count > 0) {
    return -1;
  }

  try {
    std::vector<byte_track::Object> input;
    input.reserve(object_count);
    for (size_t i = 0; i < object_count; ++i) {
      const auto &obj = objects[i];
      input.emplace_back(byte_track::Rect<float>(obj.x, obj.y, obj.width, obj.height),
                         obj.label,
                         obj.prob);
    }

    const auto tracks = handle->impl->update(input);
    *out_count = tracks.size();

    if (out_tracks == nullptr || out_capacity == 0) {
      return tracks.empty() ? 0 : 1;
    }

    const size_t write_count = std::min(out_capacity, tracks.size());
    for (size_t i = 0; i < write_count; ++i) {
      fill_track(*tracks[i], out_tracks[i]);
    }

    return (out_capacity < tracks.size()) ? 1 : 0;
  } catch (...) {
    return -3;
  }
}