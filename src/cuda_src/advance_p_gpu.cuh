#ifndef __NTHU_ADVANCE_P_GPU_H__
#define __NTHU_ADVANCE_P_GPU_H__


typedef struct advance_p_gpu_args{
  particle_t* p0;       
  particle_mover_t* pm;      
  accumulator_t* a0;      
  interpolator_t* f0;      

  float qdt_2mc, cdt_dx, cdt_dy, cdt_dz, qsp;
  int np;
  // original: int np, max_nm, nx, ny, nz; 
  
  int block_size;
  
  const int64_t *g_neighbor;
  int64_t g_rangel, g_rangeh; 
} advance_p_gpu_args_t;

__global__
void 
advance_p_gpu(advance_p_gpu_args args);

#endif