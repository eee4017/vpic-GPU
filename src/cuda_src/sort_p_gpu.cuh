#ifndef __NTHU_SORT_P_GPU_H__
#define __NTHU_SORT_P_GPU_H__


__global__
void  copy_particle(particle_t *src, particle_t* dst, int num_items);

__global__
void  copy_particle_index(int32_t *dst, particle_t* src, int num_items);

#endif