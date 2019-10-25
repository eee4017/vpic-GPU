#ifndef __NTHU_GPU_UTIL_H__
#define __NTHU_GPU_UTIL_H__
#include <cstdio>
#include <cstdlib>
#include <string>
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

class cudaTimer{
  cudaEvent_t _start, _stop;
public:
  cudaTimer(){
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }
  void start(){
    cudaEventRecord(_start);
  }
  void end(){
    cudaEventRecord(_stop);
  }
  float getTime(){
    cudaEventSynchronize(_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, _start, _stop);
    return milliseconds;
  }

  void printTime(const char * name){
    fprintf(stderr,YELLOW "[Timer %s]: %f ms\n" COLOR_END, name, getTime());
  }
};

// template <typename T>
// class cudaGlobalVector{
// private:
//   T * vecctor_ptr;
//   size_t *counter;
// public:
//   __host__ cudaGlobalVector(size_t max_size){
//     cudaMalloc();
//   }
//   __device__ inline void push_back(const T& the){
//     size_t idx = atomicAdd(counter)
//   }

// };



#endif


