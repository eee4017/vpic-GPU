#ifndef __NTHU_BACKFILL_GPU_H__
#define __NTHU_BACKFILL_GPU_H__

#include "gpu.cuh"

__global__ void findPAndPm(particle_t* device_p, particle_mover_t* device_pm,
                           particle_t* d_p0, int np, int nm);

__global__ void back_fill_stage_1(particle_t* device_p, particle_mover_t* device_pm,
                                  particle_t* device_particle_temp, int* device_particle_counter,
                                  particle_t* particle_selected, int np, int nm, const int block_size);

__global__ void back_fill_stage_2(particle_t* device_p, particle_mover_t* device_pm,
                                  particle_t* device_particle_temp, int* device_particle_counter,
                                  particle_t* particle_selected, int np, int nm, const int block_size);

__global__ void back_fill_stage_3(particle_t* device_p, particle_mover_t* device_pm,
                                  particle_t* device_particle_temp, int* device_particle_counter,
                                  particle_t* particle_selected, int np, int nm, const int block_size);

#endif