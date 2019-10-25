#include "gpu.cuh"
#include "backfill_gpu.cuh"
#include "gpu_util.cuh"
#include <cooperative_groups.h>


__global__
void back_fill_read(particle_t* device_p, particle_mover_t* device_pm, 
               particle_t* device_particle_temp, int *device_particle_counter,
               particle_t* particle_selected, int np, int nm, const int block_size){

  const int block_rank = blockIdx.x;
  const int n_block = gridDim.x;
  const int thread_rank = threadIdx.x;
  const int n_thread = blockDim.x;

  int itmp, n;
  GPU_DISTRIBUTE(nm, block_size, block_rank, itmp, n);
  
  particle_mover_t pm;
  particle_t p, p_backfill;
  if(thread_rank < n){
    pm = device_pm[itmp + thread_rank];
    p = device_p[pm.i];
    particle_selected[itmp + thread_rank] = p;
    device_p[pm.i].i = -31;
  }
  __syncthreads();
  if(thread_rank < n){
    p_backfill = device_p[np - nm + (itmp + thread_rank)];
    if(p_backfill.i != -31){
      if (pm.i < np - nm){
        device_p[pm.i] = p_backfill;
      } else {
        int idx = atomicAdd(device_particle_counter, 1);
        // printf("INC: %d %d\n", idx, p_backfill.i);
        device_particle_temp[idx] = p_backfill;
      }
    }
  }
}

__global__
void back_fill_write(particle_t* device_p, particle_mover_t* device_pm, 
               particle_t* device_particle_temp, int *device_particle_counter,
               particle_t* particle_selected, int np, int nm, const int block_size){

  const int block_rank = blockIdx.x;
  const int n_block = gridDim.x;
  const int thread_rank = threadIdx.x;
  const int n_thread = blockDim.x;

  int itmp, n;
  GPU_DISTRIBUTE(nm, block_size, block_rank, itmp, n);

  particle_mover_t pm;
  particle_t p;
  if(thread_rank < n){
    pm = device_pm[itmp + thread_rank];
    p = device_p[pm.i];
    if(pm.i < np - nm && p.i == -31){
      int idx = atomicAdd(device_particle_counter, -1); 
      // printf("DEC: %d %d\n", idx, device_particle_temp[idx].i);
      device_p[pm.i] = device_particle_temp[idx];
    }
  }
}

__global__
void findPAndPm(particle_t * device_p, particle_mover_t * device_pm, 
                particle_t * d_p0, particle_mover_t * d_pm, int np, int nm)
{
  int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if(tid == 0){
    particle_t *         p = device_p;
    particle_mover_t *  pm = device_pm + nm - 1;

    for( ; nm; pm--, nm-- ) {
        int i = pm->i;

        //copy p/pm
        d_p0[nm-1] = p[i];
        // d_pm[nm-1] = *pm;

        np--;
        p[i] = p[np];
      }
  }
}
