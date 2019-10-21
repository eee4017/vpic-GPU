#include "gpu.cuh"
#include "backfill_gpu.cuh"
#include "gpu_util.cuh"


__global__
void findPAndPm(particle_t * device_p, particle_mover_t * device_pm, particle_t * d_p0, particle_mover_t * d_pm, int np, int nm)
{
  int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if(tid == 0){
    particle_t *         p = device_p;
    particle_mover_t *  pm = device_pm + nm - 1;

    for( ; nm; pm--, nm-- ) {
        int i = pm->i;

        //copy p/pm
        d_p0[nm-1] = p[i];
        d_pm[nm-1] = *pm;

        np--;
        p[i] = p[np];
      }
  }
}