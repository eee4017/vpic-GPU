#include "energy_p_gpu.cuh"
#include "gpu_util.cuh"
#include <cub/cub.cuh>



__global__
void 
energy_p_gpu(energy_p_gpu_args args) {
  const int block_rank = blockIdx.x;
  const int n_block = gridDim.x;
  const int thread_rank = threadIdx.x;
  const int n_thread = blockDim.x;
  const int block_size = args.block_size;

  const interpolator_t *f_global = args.f;
  const particle_t     *p_global = args.p;

  const double qdt_2mc = args.qdt_2mc;
  const double msp     = args.msp;
  const double one     = 1.0;

  double dx, dy, dz;
  double v0, v1, v2;

  double en = 0.0;

  int i, itmp, size;
  GPU_DISTRIBUTE(args.np, block_size, block_rank, itmp, size);
  // printf("%d, %d, %d, %d, %d \n",args.np, block_size, block_rank, itmp, size);

  int prev_i = -1;
  interpolator_t f;

for(int n = itmp; n < itmp + size; n += n_thread){
  if(n + thread_rank < itmp + size) {
    particle_t p = p_global[n + thread_rank];
    dx  = p.dx;
    dy  = p.dy;
    dz  = p.dz;
    i   = p.i;
    if( i != prev_i){
      f = f_global[i];
      prev_i = i;
    }

    v0  = p.ux + qdt_2mc*(    ( f.ex    + dy*f.dexdy    ) +
                              dz*( f.dexdz + dy*f.d2exdydz ) );

    v1  = p.uy + qdt_2mc*(    ( f.ey    + dz*f.deydz    ) +
                              dx*( f.deydx + dz*f.d2eydzdx ) );

    v2  = p.uz + qdt_2mc*(    ( f.ez    + dx*f.dezdx    ) +
                              dy*( f.dezdy + dx*f.d2ezdxdy ) );

    v0  = v0*v0 + v1*v1 + v2*v2;

    v0  = (msp * p.w) * (v0 / (one + sqrtf(one + v0)));

    en += ( double ) v0;
  }
}

  __syncthreads();
  typedef cub::WarpReduce<double> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  double sum_en = WarpReduce(temp_storage).Sum(en);
  if(thread_rank == 0) atomicAdd(args.en, sum_en);
}
