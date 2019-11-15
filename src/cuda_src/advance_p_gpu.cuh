#ifndef __NTHU_ADVANCE_P_GPU_H__
#define __NTHU_ADVANCE_P_GPU_H__

#include "gpu.cuh"

typedef struct advance_p_gpu_args {
  particle_t* p0;
  particle_mover_t* pm_array;
  particle_mover_t* temp_pm_array;
  accumulator_t* a0;
  interpolator_t* f0;
  int* nm;

  float qdt_2mc, cdt_dx, cdt_dy, cdt_dz, qsp;
  int np;
  int stride_size;

  // const int64_t* g_neighbor;
  // int64_t g_rangel, g_rangeh;
} advance_p_gpu_args_t;

typedef struct handle_args {
  particle_t* p0;
  particle_mover_t* pm_array;
  particle_mover_t* temp_pm_array;
  accumulator_t* a0;
  interpolator_t* f0;
  int* nm;

  float qdt_2mc, cdt_dx, cdt_dy, cdt_dz, qsp;
  int np;
  int stride_size;

  const int64_t* g_neighbor;
  int64_t g_rangel, g_rangeh;
} handle_args_t;

__global__ void
advance_p_gpu(advance_p_gpu_args args);

__global__ void
// handle_particle_movers(advance_p_gpu_args args, int temp_nm);
handle_particle_movers(handle_args args, int temp_nm);

#endif
