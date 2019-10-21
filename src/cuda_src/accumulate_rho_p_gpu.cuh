#ifndef __NTHU_ACCUMULATE_RHO_P_GPU_H__
#define __NTHU_ACCUMULATE_RHO_P_GPU_H__

#include "gpu.cuh"

struct accumulate_rho_p_gpu_args{
  field_t    *f;
  particle_t *p;

  float q_8V;
  int np;
  int sy;
  int sz;  
  int block_size;
};

__global__ void accumulate_rho_p_gpu(accumulate_rho_p_gpu_args);

#endif