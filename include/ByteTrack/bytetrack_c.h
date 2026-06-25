#ifndef BYTE_TRACK_C_API_H_
#define BYTE_TRACK_C_API_H_

#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef BYTETRACK_EXPORTS
#    define BYTETRACK_C_API __declspec(dllexport)
#  else
#    define BYTETRACK_C_API __declspec(dllimport)
#  endif
#else
#  if __GNUC__ >= 4
#    define BYTETRACK_C_API __attribute__((visibility("default")))
#  else
#    define BYTETRACK_C_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bytetrack_handle bytetrack_handle;

typedef struct bytetrack_object_t {
  float x;
  float y;
  float width;
  float height;
  int label;
  float prob;
} bytetrack_object_t;

typedef struct bytetrack_track_t {
  float x;
  float y;
  float width;
  float height;
  float score;
  size_t track_id;
  size_t frame_id;
  size_t start_frame_id;
  size_t tracklet_length;
  int is_activated;
  int state;
} bytetrack_track_t;

/*
 * Return codes:
 *  0: success
 *  1: output buffer too small (out_count contains required count)
 * -1: invalid argument
 * -2: allocation/creation failure
 * -3: internal runtime error
 */
BYTETRACK_C_API bytetrack_handle *bytetrack_create(int frame_rate,
                                                    int track_buffer,
                                                    float track_thresh,
                                                    float high_thresh,
                                                    float match_thresh);

BYTETRACK_C_API void bytetrack_destroy(bytetrack_handle *handle);

BYTETRACK_C_API int bytetrack_update(bytetrack_handle *handle,
                                     const bytetrack_object_t *objects,
                                     size_t object_count,
                                     bytetrack_track_t *out_tracks,
                                     size_t out_capacity,
                                     size_t *out_count);

#ifdef __cplusplus
}
#endif

#endif  // BYTE_TRACK_C_API_H_