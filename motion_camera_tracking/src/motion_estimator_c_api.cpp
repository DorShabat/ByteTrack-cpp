#include "motion_estimator_c_api.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

struct motion_handle {
    int width = 0;
    int height = 0;
    int history_size = 5;

    cv::Mat prev_gray;

    std::deque<cv::Matx33f> history;
};

static void set_identity(motion_result_t* out) {
    for (int i = 0; i < 9; ++i) out->matrix[i] = 0.0f;

    out->matrix[0] = 1.0f;
    out->matrix[4] = 1.0f;
    out->matrix[8] = 1.0f;

    out->dx = 0.0f;
    out->dy = 0.0f;
    out->unit_x = 0.0f;
    out->unit_y = 0.0f;
    out->angle_rad = 0.0f;
    out->angle_deg = 0.0f;
    out->scale_x = 1.0f;
    out->scale_y = 1.0f;
    out->confidence = 0.0f;
    out->valid = 0;
    out->num_features = 0;
    out->num_tracked = 0;
    out->num_inliers = 0;
}

static bool point_in_box(const cv::Point2f& p, const motion_bbox_t& b) {
    return p.x >= b.x &&
           p.x <= b.x + b.width &&
           p.y >= b.y &&
           p.y <= b.y + b.height;
}

static bool should_ignore_point(
    const cv::Point2f& p,
    const motion_bbox_t* boxes,
    size_t count
) {
    if (!boxes) return false;

    for (size_t i = 0; i < count; ++i) {
        if (point_in_box(p, boxes[i])) {
            return true;
        }
    }

    return false;
}

static float median(std::vector<float> values) {
    if (values.empty()) return 0.0f;

    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;

    if (values.size() % 2 == 1) {
        return values[mid];
    }

    return 0.5f * (values[mid - 1] + values[mid]);
}

static cv::Matx33f median_matrix(const std::deque<cv::Matx33f>& history) {
    if (history.empty()) {
        return cv::Matx33f::eye();
    }

    cv::Matx33f out;

    for (int idx = 0; idx < 9; ++idx) {
        std::vector<float> vals;
        vals.reserve(history.size());

        for (const auto& m : history) {
            vals.push_back(m.val[idx]);
        }

        out.val[idx] = median(vals);
    }

    out(2, 0) = 0.0f;
    out(2, 1) = 0.0f;
    out(2, 2) = 1.0f;

    return out;
}

static void fill_result_from_matrix(
    const cv::Matx33f& M,
    motion_result_t* out,
    int valid,
    float confidence,
    int num_features,
    int num_tracked,
    int num_inliers
) {
    for (int i = 0; i < 9; ++i) {
        out->matrix[i] = M.val[i];
    }

    const float a  = M(0, 0);
    const float b  = M(0, 1);
    const float tx = M(0, 2);

    const float c  = M(1, 0);
    const float d  = M(1, 1);
    const float ty = M(1, 2);

    out->dx = tx;
    out->dy = ty;

    const float norm = std::sqrt(tx * tx + ty * ty);
    if (norm > 1e-6f) {
        out->unit_x = tx / norm;
        out->unit_y = ty / norm;
    } else {
        out->unit_x = 0.0f;
        out->unit_y = 0.0f;
    }

    out->angle_rad = std::atan2(c, a);
    out->angle_deg = out->angle_rad * 180.0f / 3.1415926535f;

    out->scale_x = std::sqrt(a * a + c * c);
    out->scale_y = std::sqrt(b * b + d * d);

    out->confidence = confidence;
    out->valid = valid;

    out->num_features = num_features;
    out->num_tracked = num_tracked;
    out->num_inliers = num_inliers;
}

motion_handle* motion_create(
    int width,
    int height,
    int history_size
) {
    if (width <= 0 || height <= 0) {
        return nullptr;
    }

    auto* h = new motion_handle();
    h->width = width;
    h->height = height;
    h->history_size = std::max(1, history_size);

    return h;
}

void motion_destroy(motion_handle* handle) {
    delete handle;
}

void motion_reset(motion_handle* handle) {
    if (!handle) return;

    handle->prev_gray.release();
    handle->history.clear();
}

int motion_update(
    motion_handle* handle,
    const unsigned char* bgr,
    int width,
    int height,
    int stride,
    const motion_bbox_t* ignore_boxes,
    size_t ignore_count,
    motion_result_t* out_result
) {
    if (!handle || !bgr || !out_result) {
        return -1;
    }

    if (width <= 0 || height <= 0 || stride < width * 3) {
        return -1;
    }

    set_identity(out_result);

    cv::Mat curr_bgr(height, width, CV_8UC3, const_cast<unsigned char*>(bgr), stride);
    cv::Mat curr_gray;
    cv::cvtColor(curr_bgr, curr_gray, cv::COLOR_BGR2GRAY);

    if (handle->prev_gray.empty()) {
        handle->prev_gray = curr_gray.clone();
        return 0;
    }

    std::vector<cv::Point2f> prev_pts;
    cv::goodFeaturesToTrack(
        handle->prev_gray,
        prev_pts,
        1000,
        0.01,
        8.0
    );

    std::vector<cv::Point2f> filtered_prev_pts;
    filtered_prev_pts.reserve(prev_pts.size());

    for (const auto& p : prev_pts) {
        if (!should_ignore_point(p, ignore_boxes, ignore_count)) {
            filtered_prev_pts.push_back(p);
        }
    }

    if (filtered_prev_pts.size() < 8) {
        handle->prev_gray = curr_gray.clone();
        return 0;
    }

    std::vector<cv::Point2f> curr_pts;
    std::vector<unsigned char> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        handle->prev_gray,
        curr_gray,
        filtered_prev_pts,
        curr_pts,
        status,
        err
    );

    std::vector<cv::Point2f> good_prev;
    std::vector<cv::Point2f> good_curr;

    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) continue;

        if (should_ignore_point(curr_pts[i], ignore_boxes, ignore_count)) {
            continue;
        }

        good_prev.push_back(filtered_prev_pts[i]);
        good_curr.push_back(curr_pts[i]);
    }

    if (good_prev.size() < 8) {
        handle->prev_gray = curr_gray.clone();
        return 0;
    }

    cv::Mat inliers;
    cv::Mat affine = cv::estimateAffinePartial2D(
        good_prev,
        good_curr,
        inliers,
        cv::RANSAC,
        3.0,
        2000,
        0.99,
        10
    );

    if (affine.empty()) {
        handle->prev_gray = curr_gray.clone();
        return 0;
    }

    int num_inliers = 0;
    for (int i = 0; i < inliers.rows; ++i) {
        if (inliers.at<unsigned char>(i, 0)) {
            ++num_inliers;
        }
    }

    const float confidence =
        good_prev.empty() ? 0.0f : static_cast<float>(num_inliers) / static_cast<float>(good_prev.size());

    cv::Matx33f raw_M = cv::Matx33f::eye();

    raw_M(0, 0) = static_cast<float>(affine.at<double>(0, 0));
    raw_M(0, 1) = static_cast<float>(affine.at<double>(0, 1));
    raw_M(0, 2) = static_cast<float>(affine.at<double>(0, 2));
    raw_M(1, 0) = static_cast<float>(affine.at<double>(1, 0));
    raw_M(1, 1) = static_cast<float>(affine.at<double>(1, 1));
    raw_M(1, 2) = static_cast<float>(affine.at<double>(1, 2));

    handle->history.push_back(raw_M);

    while (static_cast<int>(handle->history.size()) > handle->history_size) {
        handle->history.pop_front();
    }

    cv::Matx33f filtered_M = median_matrix(handle->history);

    fill_result_from_matrix(
        filtered_M,
        out_result,
        confidence > 0.25f ? 1 : 0,
        confidence,
        static_cast<int>(filtered_prev_pts.size()),
        static_cast<int>(good_prev.size()),
        num_inliers
    );

    handle->prev_gray = curr_gray.clone();

    return 0;
}