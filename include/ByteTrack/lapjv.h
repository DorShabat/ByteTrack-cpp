#ifndef BYTE_TRACK_LAPJV_H_
#define BYTE_TRACK_LAPJV_H_

#include <cstddef>

namespace byte_track
{
int lapjv_internal(const size_t n, double *cost[], int *x, int *y);
}

#endif  // BYTE_TRACK_LAPJV_H_