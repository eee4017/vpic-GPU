#include <cub/cub.cuh>

#include "gpu.cuh"
#include "gpu_util.cuh"
#include "sort_p_gpu.cuh"

__global__ void copy_particle(particle_t* dst, particle_t* src, int num_items) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < num_items; i += stride) {
    dst[i] = src[i];
  }
}

__global__ void copy_particle_index(int32_t* dst, particle_t* src, int num_items) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < num_items; i += stride) {
    dst[i] = src[i].i;
  }
}