#ifndef __NTHU_GPU_UTIL_H__
#define __NTHU_GPU_UTIL_H__
#include <CppToolkit/color.h>
#include <execinfo.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#define MATH_CEIL(a, b) (((a) + (b)-1) / (b))

#define GPU_DISTRIBUTE(np, stride_size, block_index, i, n) \
  BEGIN_PRIMITIVE {                                        \
    int _nb = MATH_CEIL(np, stride_size);                  \
    i = stride_size * block_index;                         \
    n = (block_index == _nb - 1) ? (np - i) : stride_size; \
  }                                                        \
  END_PRIMITIVE

#define BACK_TRACE_BUFFER_SIZE 1005

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    void *buffer[BACK_TRACE_BUFFER_SIZE];
    int nptrs = backtrace(buffer, BACK_TRACE_BUFFER_SIZE);
    fprintf(stderr, LIGHT_RED "GPUassert: %s %s %d\n" COLOR_END, cudaGetErrorString(code), file, line);
    backtrace_symbols_fd(buffer, nptrs, stderr->_fileno);
    if (abort) exit(code);
  }
}

class cudaTimer {
  cudaEvent_t _start, _stop;

 public:
  cudaTimer() {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }
  void start() {
    cudaEventRecord(_start);
  }
  void end() {
    cudaEventRecord(_stop);
  }
  float getTime() {
    cudaEventSynchronize(_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, _start, _stop);
    return milliseconds;
  }

  void printTime(const char *name) {
    fprintf(stderr, YELLOW "[Timer %s]: %f ms\n" COLOR_END, name, getTime());
  }
};



#endif
