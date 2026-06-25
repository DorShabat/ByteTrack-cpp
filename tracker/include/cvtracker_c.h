#ifndef CVTRACKER_C_H_
#define CVTRACKER_C_H_

#include <stddef.h>
#include <stdint.h>

/* Symbol visibility */
#ifndef CVTRACKER_API
#  if defined(_WIN32) || defined(__CYGWIN__)
#    ifdef CVTRACKER_EXPORTS
#      define CVTRACKER_API __declspec(dllexport)
#    else
#      define CVTRACKER_API __declspec(dllimport)
#    endif
#  elif __GNUC__ >= 4
#    define CVTRACKER_API __attribute__((visibility("default")))
#  else
#    define CVTRACKER_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Structs                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    float x;
    float y;
    float width;
    float height;
    float score;
    int   label;
} cvtracker_object_t;

typedef struct {
    float x;
    float y;
    float width;
    float height;
    float score;
    int   track_id;
    /* 1 = actively tracked by cv::Tracker, 0 = kept alive (detector absent) */
    int   is_active;
} cvtracker_track_t;

/* ------------------------------------------------------------------ */
/* Opaque handle                                                        */
/* ------------------------------------------------------------------ */

typedef struct cvtracker_handle cvtracker_handle_t;

/* ------------------------------------------------------------------ */
/* API                                                                  */
/* ------------------------------------------------------------------ */

/*
 * Create a multi-object tracker.
 *
 * tracker_type  : "CSRT" (best quality), "KCF" (fast).  NULL = "CSRT".
 * iou_thresh    : min IoU to match a detection to an existing track (e.g. 0.3f).
 * lost_ttl      : frames to keep a track alive when the cv::Tracker fails and
 *                 no matching detection arrives.
 */
CVTRACKER_API cvtracker_handle_t* cvtracker_create(const char* tracker_type,
                                                   float       iou_thresh,
                                                   int         lost_ttl_frames);

CVTRACKER_API void cvtracker_destroy(cvtracker_handle_t* handle);

/*
 * Process one video frame.
 *
 * frame_data    : raw BGR bytes, exactly width * height * 3 bytes.
 * width, height : frame dimensions.
 * objects       : detections for this frame (may be NULL / 0 when skipped).
 * out_tracks    : caller-allocated output array.
 * out_capacity  : number of elements in out_tracks.
 * out_count     : set to the number of active tracks on return.
 *
 * Returns 0 on success, -1 on error.
 */
CVTRACKER_API int cvtracker_update(cvtracker_handle_t*       handle,
                                   const uint8_t*            frame_data,
                                   int                       width,
                                   int                       height,
                                   const cvtracker_object_t* objects,
                                   size_t                    object_count,
                                   cvtracker_track_t*        out_tracks,
                                   size_t                    out_capacity,
                                   size_t*                   out_count);

#ifdef __cplusplus
}
#endif

#endif /* CVTRACKER_C_H_ */
