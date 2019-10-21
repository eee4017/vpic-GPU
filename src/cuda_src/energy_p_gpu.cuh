#ifndef __NTHU_ENERGY_P_GPU_H__
#define __NTHU_ENERGY_P_GPU_H__
#include "gpu.cuh"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
unsigned long long int* address_as_ull = (unsigned long long int*)address;
unsigned long long int old = *address_as_ull, assumed;
if (val==0.0)
    return __longlong_as_double(old);
do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
} while (assumed != old);
return __longlong_as_double(old);
}

#endif

struct energy_p_gpu_args {
  particle_t *p;  
  interpolator_t* f;     
  double *en;     
  float  qdt_2mc; 
  float  msp;    
  int np;      
  int block_size;
};

__global__ void 
energy_p_gpu(energy_p_gpu_args args);

#endif
