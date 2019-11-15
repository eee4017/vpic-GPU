#include <cub/cub.cuh>

#include "energy_p_gpu.cuh"
#include "gpu_util.cuh"

__global__ void
energy_p_gpu(energy_p_gpu_args args) {
  const int bstart = blockIdx.x * args.stride_size;
  const int bend = min((blockIdx.x + 1) * args.stride_size, args.np);
  const int stride_size = args.stride_size;

  const interpolator_t *f_global = args.f;
  const particle_t *p_global = args.p;

  const double qdt_2mc = args.qdt_2mc;
  const double msp = args.msp;
  const double one = 1.0;

  // double dx, dy, dz;
  // double v0, v1, v2;

  double en = 0.0;


  for (int pid = bstart + threadIdx.x; pid < bend; pid += blockDim.x) {
      const particle_t& p = p_global[pid];
      const double dx = p.dx;
      const double dy = p.dy;
      const double dz = p.dz;

      const interpolator_t& f = f_global[p.i];
      // f = f_global[p.i];
      
      double v0 = p.ux + qdt_2mc * ((f.ex + dy * f.dexdy) +
                             dz * (f.dexdz + dy * f.d2exdydz));

      double v1 = p.uy + qdt_2mc * ((f.ey + dz * f.deydz) +
                             dx * (f.deydx + dz * f.d2eydzdx));

      double v2 = p.uz + qdt_2mc * ((f.ez + dx * f.dezdx) +
                             dy * (f.dezdy + dx * f.d2ezdxdy));

      v0 = v0 * v0 + v1 * v1 + v2 * v2;

      v0 = (msp * p.w) * (v0 / (one + sqrtf(one + v0)));

      en += (double)v0;
  }

  atomicAdd(args.en, en);
  // __syncthreads();
  // typedef cub::WarpReduce<double> WarpReduce;
  // __shared__ typename WarpReduce::TempStorage temp_storage;
  // double sum_en = WarpReduce(temp_storage).Sum(en);
  // if (thread_rank == 0) atomicAdd(args.en, sum_en);  // TODO:sum_en
}
