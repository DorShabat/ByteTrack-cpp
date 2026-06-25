#include "cvtracker_c.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

namespace {

static float rect_iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    float ix = std::max(0.f, std::min(a.x + a.width,  b.x + b.width)  - std::max(a.x, b.x));
    float iy = std::max(0.f, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
    float inter = ix * iy;
    if (inter <= 0.f) return 0.f;
    return inter / (a.area() + b.area() - inter);
}

static cv::Ptr<cv::Tracker> make_cv_tracker(const std::string& type)
{
    if (type == "KCF") return cv::TrackerKCF::create();
    return cv::TrackerCSRT::create(); /* default */
}

struct Track {
    int                    id;
    cv::Rect2f             rect;
    float                  score;
    cv::Ptr<cv::Tracker>   cv_tracker;
    int                    lost_frames; /* frames since last matched detection */
    int                    is_active;   /* cv tracker succeeded this frame     */
};

} /* anonymous namespace */

/* ------------------------------------------------------------------ */
/* Handle definition                                                    */
/* ------------------------------------------------------------------ */

struct cvtracker_handle {
    std::string        tracker_type;
    float              iou_thresh;
    int                lost_ttl;
    int                next_id;
    std::vector<Track> tracks;
};

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

cvtracker_handle_t* cvtracker_create(const char* tracker_type,
                                     float       iou_thresh,
                                     int         lost_ttl_frames)
{
    auto* h = new cvtracker_handle;
    h->tracker_type = tracker_type ? tracker_type : "CSRT";
    h->iou_thresh   = (iou_thresh > 0.f) ? iou_thresh : 0.3f;
    h->lost_ttl     = (lost_ttl_frames > 0) ? lost_ttl_frames : 5;
    h->next_id      = 1;
    return h;
}

void cvtracker_destroy(cvtracker_handle_t* handle)
{
    delete handle;
}

int cvtracker_update(cvtracker_handle_t*       handle,
                     const uint8_t*            frame_data,
                     int                       width,
                     int                       height,
                     const cvtracker_object_t* objects,
                     size_t                    object_count,
                     cvtracker_track_t*        out_tracks,
                     size_t                    out_capacity,
                     size_t*                   out_count)
{
    if (!handle || !frame_data || !out_count) return -1;

    /* Wrap raw BGR bytes as cv::Mat (zero-copy). */
    cv::Mat frame(height, width, CV_8UC3, const_cast<uint8_t*>(frame_data));

    /* ---- Step 1: advance every existing cv::Tracker ---- */
    for (auto& t : handle->tracks) {
        if (!t.cv_tracker) {
            t.is_active = 0;
            t.lost_frames++;
            continue;
        }
        cv::Rect2i roi;
        bool ok = t.cv_tracker->update(frame, roi);
        if (ok) {
            t.rect      = cv::Rect2f(roi);
            t.is_active = 1;
        } else {
            t.is_active = 0;
            t.lost_frames++;
        }
    }

    /* ---- Step 2: match detections to tracks by IoU (greedy) ---- */
    std::vector<bool> det_matched(object_count, false);
    std::vector<bool> trk_matched(handle->tracks.size(), false);

    if (object_count > 0 && !handle->tracks.empty()) {
        /* Build candidate list sorted by descending IoU. */
        std::vector<std::tuple<float, int, int>> candidates;
        for (size_t d = 0; d < object_count; ++d) {
            cv::Rect2f dr(objects[d].x, objects[d].y,
                          objects[d].width, objects[d].height);
            for (size_t t = 0; t < handle->tracks.size(); ++t) {
                float v = rect_iou(dr, handle->tracks[t].rect);
                if (v >= handle->iou_thresh)
                    candidates.emplace_back(v, static_cast<int>(d), static_cast<int>(t));
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                      return std::get<0>(a) > std::get<0>(b);
                  });

        for (const auto& [score_val, di, ti] : candidates) {
            if (det_matched[di] || trk_matched[ti]) continue;
            det_matched[di] = trk_matched[ti] = true;

            auto& tr    = handle->tracks[ti];
            tr.rect     = cv::Rect2f(objects[di].x, objects[di].y,
                                     objects[di].width, objects[di].height);
            tr.score    = objects[di].score;
            tr.lost_frames = 0;
            tr.is_active   = 1;

            /* Re-initialise cv::Tracker with the fresh detection box. */
            tr.cv_tracker = make_cv_tracker(handle->tracker_type);
            cv::Rect2i roi(static_cast<int>(tr.rect.x),
                           static_cast<int>(tr.rect.y),
                           static_cast<int>(tr.rect.width),
                           static_cast<int>(tr.rect.height));
            tr.cv_tracker->init(frame, roi);
        }
    }

    /* ---- Step 3: spawn new tracks for unmatched detections ---- */
    for (size_t d = 0; d < object_count; ++d) {
        if (det_matched[d]) continue;
        Track tr;
        tr.id          = handle->next_id++;
        tr.rect        = cv::Rect2f(objects[d].x, objects[d].y,
                                    objects[d].width, objects[d].height);
        tr.score       = objects[d].score;
        tr.lost_frames = 0;
        tr.is_active   = 1;
        tr.cv_tracker  = make_cv_tracker(handle->tracker_type);
        cv::Rect2i roi(static_cast<int>(tr.rect.x),
                       static_cast<int>(tr.rect.y),
                       static_cast<int>(tr.rect.width),
                       static_cast<int>(tr.rect.height));
        tr.cv_tracker->init(frame, roi);
        handle->tracks.push_back(std::move(tr));
    }

    /* ---- Step 4: remove stale tracks ---- */
    handle->tracks.erase(
        std::remove_if(handle->tracks.begin(), handle->tracks.end(),
                       [&](const Track& t) {
                           return t.lost_frames > handle->lost_ttl;
                       }),
        handle->tracks.end());

    /* ---- Step 5: fill output ---- */
    *out_count = handle->tracks.size();
    if (!out_tracks || out_capacity == 0) return 0;

    const size_t n = std::min(out_capacity, handle->tracks.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& t    = handle->tracks[i];
        out_tracks[i].x         = t.rect.x;
        out_tracks[i].y         = t.rect.y;
        out_tracks[i].width     = t.rect.width;
        out_tracks[i].height    = t.rect.height;
        out_tracks[i].score     = t.score;
        out_tracks[i].track_id  = t.id;
        out_tracks[i].is_active = t.is_active;
    }
    return 0;
}
