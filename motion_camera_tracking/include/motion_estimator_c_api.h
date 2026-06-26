#pragma once
#include <stddef.h>

#ifdef _WIN32
#ifdef MOTION_ESTIMATOR_EXPORTS
#define MOTION_API __declspec(dllexport)
#else
#define MOTION_API __declspec(dllimport)
#endif
#else
#define MOTION_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct motion_handle motion_handle;

typedef struct motion_bbox_t {
    float x;
    float y;
    float width;
    float height;
} motion_bbox_t;

typedef struct motion_result_t {
    float matrix[9];      // row-major 3x3: [a b tx; c d ty; 0 0 1]

    float dx;
    float dy;

    float unit_x;
    float unit_y;

    float angle_rad;
    float angle_deg;

    float scale_x;
    float scale_y;

    float confidence;
    int valid;

    int num_features;
    int num_tracked;
    int num_inliers;
} motion_result_t;

MOTION_API motion_handle* motion_create(
    int width,
    int height,
    int history_size
);

MOTION_API void motion_destroy(motion_handle* handle);

MOTION_API void motion_reset(motion_handle* handle);

MOTION_API int motion_update(
    motion_handle* handle,
    const unsigned char* bgr,
    int width,
    int height,
    int stride,
    const motion_bbox_t* ignore_boxes,
    size_t ignore_count,
    motion_result_t* out_result
);

#ifdef __cplusplus
}
#endif