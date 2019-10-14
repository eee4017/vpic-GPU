#ifndef __NTHU_MOVE_P_GPU_H__
#define __NTHU_MOVE_P_GPU_H__

#include "gpu.cuh"

__device__ 
int move_p_gpu(particle_t *p_register, particle_mover_t *pm,
accumulator_t *a0, const int64_t *g_neighbor,
int64_t g_rangel, int64_t g_rangeh, const float qsp);

#endif