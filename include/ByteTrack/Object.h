#ifndef BYTE_TRACK_OBJECT_H_
#define BYTE_TRACK_OBJECT_H_

#include "ByteTrack/Rect.h"

namespace byte_track
{
struct Object
{
    Rect<float> rect;
    int label;
    float prob;

    Object(const Rect<float> &_rect,
           const int &_label,
           const float &_prob);
};
}

#endif  // BYTE_TRACK_OBJECT_H_