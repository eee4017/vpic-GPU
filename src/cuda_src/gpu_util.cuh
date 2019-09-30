#ifndef __NTHU_GPU_UTIL_H__
#define __NTHU_GPU_UTIL_H__
#include <cstdio>
#include <cstdlib>
#include <CppToolkit/color.h>

#define MATH_CEIL(a, b) (((a) + (b)-1) / (b))

#define GPU_DISTRIBUTE(np, num_threads, block_index, i, n) \
  BEGIN_PRIMITIVE {                                        \
    int _nb = MATH_CEIL(np, num_threads);                  \
    i = num_threads * block_index;                         \
    n = (block_index == _nb - 1) ? (np - i) : num_threads; \
  }                                                        \
  END_PRIMITIVE


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,LIGHT_RED "GPUassert: %s %s %d\n" COLOR_END, cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


#endif


