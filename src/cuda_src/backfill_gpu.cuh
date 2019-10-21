#ifndef __NTHU_BACKFILL_GPU_H__
#define __NTHU_BACKFILL_GPU_H__

__global__
void findPAndPm(particle_t * device_p, particle_mover_t * device_pm, particle_t * d_p0, particle_mover_t * d_pm, int np, int nm);

#endif