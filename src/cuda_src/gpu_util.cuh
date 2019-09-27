#ifndef __NTHU_GPU_UTIL_H__
#define __NTHU_GPU_UTIL_H__

#define MATH_CEIL(a, b) (((a) + (b)-1) / (b))

#define GPU_DISTRIBUTE(np, num_threads, block_index, i, n) \
  BEGIN_PRIMITIVE {                                        \
    int _nb = MATH_CEIL(np, num_threads);                  \
    i = num_threads * block_index;                         \
    n = (block_index == _nb - 1) ? (np - i) : num_threads; \
  }                                                        \
  END_PRIMITIVE

#endif