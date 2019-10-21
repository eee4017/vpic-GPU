#include "accumulate_rho_p_gpu.cuh"
#include "gpu_util.cuh"

__global__ void accumulate_rho_p_gpu(accumulate_rho_p_gpu_args args){
  const int block_rank = blockIdx.x;
  const int n_block = gridDim.x;
  const int thread_rank = threadIdx.x;
  const int n_thread = blockDim.x;
  const int block_size = args.block_size;

  field_t    *f = args.f;
  particle_t *p_global = args.p;

  const float q_8V = args.q_8V;
  const int np = args.np;
  const int sy = args.sy;
  const int sz = args.sz;

  float w0, w1, w2, w3, w4, w5, w6, w7, dz;
  int v;
  
  int i, itmp, size;
  GPU_DISTRIBUTE(args.np, block_size, block_rank, itmp, size);

for(int n = itmp; n < itmp + size; n += n_thread){
    if(n + thread_rank < itmp + size) {
      particle_t p = p_global[n + thread_rank];
      w0 = p.dx;
      w1 = p.dy;
      dz = p.dz;
      v  = p.i;
      w7 = p.w*q_8V;

#   define FMA( x,y,z) ((z)+(x)*(y))
#   define FNMS(x,y,z) ((z)-(x)*(y))
    w6=FNMS(w0,w7,w7);                    // q(1-dx)
    w7=FMA( w0,w7,w7);                    // q(1+dx)
    w4=FNMS(w1,w6,w6); w5=FNMS(w1,w7,w7); // q(1-dx)(1-dy), q(1+dx)(1-dy)
    w6=FMA( w1,w6,w6); w7=FMA( w1,w7,w7); // q(1-dx)(1+dy), q(1+dx)(1+dy)
    w0=FNMS(dz,w4,w4); w1=FNMS(dz,w5,w5); w2=FNMS(dz,w6,w6); w3=FNMS(dz,w7,w7);
    w4=FMA( dz,w4,w4); w5=FMA( dz,w5,w5); w6=FMA( dz,w6,w6); w7=FMA( dz,w7,w7);
#   undef FNMS
#   undef FMA

      atomicAdd(&f[v      ].rhof,w0); atomicAdd(&f[v      +1].rhof,w1);
      atomicAdd(&f[v   +sy].rhof,w2); atomicAdd(&f[v   +sy+1].rhof,w3);
      atomicAdd(&f[v+sz   ].rhof,w4); atomicAdd(&f[v+sz   +1].rhof,w5);
      atomicAdd(&f[v+sz+sy].rhof,w6); atomicAdd(&f[v+sz+sy+1].rhof,w7);
      
  }
}

}
