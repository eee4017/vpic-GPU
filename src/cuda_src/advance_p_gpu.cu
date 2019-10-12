#define IN_spa

#include <cub/cub.cuh>
#include "gpu.cuh"
#include "advance_p_gpu.cuh"
#include "gpu_util.cuh"

#ifdef __USE_LESS_
__device__ int move_p_gpu(particle_t *p, particle_t *p_global, particle_mover_t *pm,
                          float *a, const int64_t *g_neighbor,
                          int64_t g_rangel, int64_t g_rangeh, const float qsp) {
  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, v4, v5, q;
  int axis, face;
  int64_t neighbor;

  q = qsp * p->w;

  for (;;) {
    s_midx = p->dx;
    s_midy = p->dy;
    s_midz = p->dz;

    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    s_dir[0] = (s_dispx > 0.0f) ? 1.0f : -1.0f;
    s_dir[1] = (s_dispy > 0.0f) ? 1.0f : -1.0f;
    s_dir[2] = (s_dispz > 0.0f) ? 1.0f : -1.0f;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx == 0.0f) ? 3.4e38f : (s_dir[0] - s_midx) / s_dispx;
    v1 = (s_dispy == 0.0f) ? 3.4e38f : (s_dir[1] - s_midy) / s_dispy;
    v2 = (s_dispz == 0.0f) ? 3.4e38f : (s_dir[2] - s_midz) / s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
    v3 = 2.0f, axis = 3;
    if (v0 < v3) v3 = v0, axis = 0;
    if (v1 < v3) v3 = v1, axis = 1;
    if (v2 < v3) v3 = v2, axis = 2;
    v3 *= 0.5f;

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
    v5 = q * s_dispx * s_dispy * s_dispz * (1. / 3.);

#define accumulate_j(X, Y, Z)                                    \
  v4 = q * s_disp##X; /* v2 = q ux                            */ \
  v1 = v4 * s_mid##Y; /* v1 = q ux dy                         */ \
  v0 = v4 - v1;       /* v0 = q ux (1-dy)                     */ \
  v1 += v4;           /* v1 = q ux (1+dy)                     */ \
  v4 = 1 + s_mid##Z;  /* v4 = 1+dz                            */ \
  v2 = v0 * v4;       /* v2 = q ux (1-dy)(1+dz)               */ \
  v3 = v1 * v4;       /* v3 = q ux (1+dy)(1+dz)               */ \
  v4 = 1 - s_mid##Z;  /* v4 = 1-dz                            */ \
  v0 *= v4;           /* v0 = q ux (1-dy)(1-dz)               */ \
  v1 *= v4;           /* v1 = q ux (1+dy)(1-dz)               */ \
  v0 += v5;           /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */ \
  v1 -= v5;           /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */ \
  v2 -= v5;           /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */ \
  v3 += v5;           /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */ \
  a[0] += v0;                                                    \
  a[1] += v1;                                                    \
  a[2] += v2;                                                    \
  a[3] += v3
    accumulate_j(x, y, z);
    a += 4;
    accumulate_j(y, z, x);
    a += 4;
    accumulate_j(z, x, y);
#undef accumulate_j

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    // Compute the new particle offset
    p->dx += s_dispx + s_dispx;
    p->dy += s_dispy + s_dispy;
    p->dz += s_dispz + s_dispz;

    // If an end streak, return success (should be ~50% of the time)

    if (axis == 3) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    v0 = s_dir[axis];
    (&(p->dx))[axis] = v0;  // Avoid roundoff fiascos--put the particle
                            // _exactly_ on the boundary.
    face = axis;
    if (v0 > 0) face += 3;
    neighbor = g_neighbor[6 * p->i + face];

    if (UNLIKELY(neighbor == reflect_particles)) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      (&(p->ux))[axis] = -(&(p->ux))[axis];
      (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
      continue;
    }

    if (UNLIKELY(neighbor < g_rangel || neighbor > g_rangeh)) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      p->i = 8 * p->i + face;
      return 1;  // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    p->i = neighbor - g_rangel;  // Compute local index of neighbor
    (&(p->dx))[axis] = -v0;      // Convert coordinate system
  }

  return 0;  // Return "mover not in use"
}

#endif

#define timer_start(elt) \
  int timer_##elt = clock();

#define  timer_end(elt) \
  timer_##elt = clock() - timer_##elt; \
  if(threadIdx.x == 0) printf(#elt": %d\n", timer_##elt); 

// Total __shared__ size = 11298 * sizeof(int)
// #define BLOCK_SIZE 512
#define SHARE_MAX_VOXEL_SIZE 1    // 18
#define SHARE_MAX_PM_SIZE 10  // > (1024 * 0.2%) // 10 * 4

__inline__ __device__
int warpAllReduceSum(int val) {
  for (int mask = warpSize/2; mask > 0; mask /= 2) 
    val += __shfl_xor(val, mask);
  return val;
}

__global__ void advance_p_gpu(advance_p_gpu_args args) {
  int block_rank = blockIdx.x;
  int n_block = gridDim.x;
  int thread_rank = threadIdx.x;
  int n_thread = blockDim.x;
  int block_size = args.block_size;
  
  const float qdt_2mc = args.qdt_2mc;
  const float cdt_dx = args.cdt_dx;
  const float cdt_dy = args.cdt_dy;
  const float cdt_dz = args.cdt_dz;
  const float qsp = args.qsp;
  const float one = 1.0;
  const float one_third = 1.0 / 3.0;
  const float two_fifteenths = 2.0 / 15.0;
  
  
  float dx, dy, dz, ux, uy, uz, q;
  float hax, hay, haz, cbx, cby, cbz;
  float v0, v1, v2, v3, v4, v5;
  float a[SHARE_MAX_VOXEL_SIZE][12];

  int itmp, n, nm, max_nm;

  particle_mover_t local_pm;
  __shared__ int f_shared_index[SHARE_MAX_VOXEL_SIZE];
  interpolator_t f_shared[SHARE_MAX_VOXEL_SIZE];

  GPU_DISTRIBUTE(args.np, block_size, block_rank, itmp, n);
  particle_t *p_global = args.p0 + itmp; 
  const interpolator_t *f_global = args.f0;

  // __syncthreads();
  // timer_start(loadf); // 32269
  if (thread_rank < SHARE_MAX_VOXEL_SIZE) {  // the first thread
    int my_index = thread_rank * ((n + SHARE_MAX_VOXEL_SIZE - 1) / SHARE_MAX_VOXEL_SIZE);
    // if(my_index > args.np) my_index = 0;
    // int my_index = thread_rank ? (n-1): 0;
    particle_t p = p_global[my_index];
    f_shared_index[thread_rank] = p.i;
    f_shared[thread_rank] = f_global[p.i];
  }
  // printf("%d %d %d %d %d\n", thread_rank, block_rank, f_shared_index[0], f_shared_index[1], f_shared_index[2]);
  // timer_end(loadf);

  __syncthreads();

  int ii = 0;
  interpolator_t f;


for(int i = 0;i < n; i+= n_thread){
  if (thread_rank < n) {
    particle_t p = p_global[i + thread_rank];

    if (f_shared_index[ii] != p.i){
      while(ii < SHARE_MAX_VOXEL_SIZE && f_shared_index[ii] != p.i) ii++;
      f = f_shared[ii];
    }

    dx = p.dx;  // Load position
    dy = p.dy;
    dz = p.dz;

    hax = qdt_2mc *
          ((f.ex + dy * f.dexdy) + dz * (f.dexdz + dy * f.d2exdydz));

    hay = qdt_2mc *
          ((f.ey + dz * f.deydz) + dx * (f.deydx + dz * f.d2eydzdx));

    haz = qdt_2mc *
          ((f.ez + dx * f.dezdx) + dy * (f.dezdy + dx * f.d2ezdxdy));

    cbx = f.cbx + dx * f.dcbxdx;  // Interpolate B
    cby = f.cby + dy * f.dcbydy;
    cbz = f.cbz + dz * f.dcbzdz;

    ux = p.ux;  // Load momentum
    uy = p.uy;
    uz = p.uz;
    q = p.w;

    ux += hax;  // Half advance E
    uy += hay;
    uz += haz;

    v0 = qdt_2mc /
         sqrtf(one + (ux * ux + (uy * uy + uz * uz)));  

    // Boris - scalars
    v1 = cbx * cbx + (cby * cby + cbz * cbz);
    v2 = (v0 * v0) * v1;
    v3 = v0 * (one + v2 * (one_third + v2 * two_fifteenths));
    v4 = v3 / (one + v1 * (v3 * v3));
    v4 += v4;

    v0 = ux + v3 * (uy * cbz - uz * cby);  // Boris - uprime
    v1 = uy + v3 * (uz * cbx - ux * cbz);
    v2 = uz + v3 * (ux * cby - uy * cbx);

    ux += v4 * (v1 * cbz - v2 * cby);  // Boris - rotation
    uy += v4 * (v2 * cbx - v0 * cbz);
    uz += v4 * (v0 * cby - v1 * cbx);

    ux += hax;  // Half advance E
    uy += hay;
    uz += haz;

    p.ux = ux;  // Store momentum
    p.uy = uy;
    p.uz = uz;

    v0 = one / sqrtf(one + (ux * ux + (uy * uy + uz * uz)));
    // Get norm displacement

    ux *= cdt_dx;
    uy *= cdt_dy;
    uz *= cdt_dz;

    ux *= v0;
    uy *= v0;
    uz *= v0;

    v0 = dx + ux;  // Streak midpoint (inbnds)
    v1 = dy + uy;
    v2 = dz + uz;

    v3 = v0 + ux;  // New position
    v4 = v1 + uy;
    v5 = v2 + uz;

    // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
    if (v3 <= one && v4 <= one && v5 <= one &&  // Check if inbnds
        -v3 <= one && -v4 <= one && -v5 <= one) {
      q *= qsp;

      p.dx = v3;  // Store new position
      p.dy = v4;
      p.dz = v5;

      dx = v0;  // Streak midpoint
      dy = v1;
      dz = v2;

      v5 = q * ux * uy * uz * one_third;  // Compute correction
      
#define ACCUMULATE_J(X, Y, Z, offset)                         \
  v4 = q * u##X;   /* v2 = q ux                            */ \
  v1 = v4 * d##Y;  /* v1 = q ux dy                         */ \
  v0 = v4 - v1;    /* v0 = q ux (1-dy)                     */ \
  v1 += v4;        /* v1 = q ux (1+dy)                     */ \
  v4 = one + d##Z; /* v4 = 1+dz                            */ \
  v2 = v0 * v4;    /* v2 = q ux (1-dy)(1+dz)               */ \
  v3 = v1 * v4;    /* v3 = q ux (1+dy)(1+dz)               */ \
  v4 = one - d##Z; /* v4 = 1-dz                            */ \
  v0 *= v4;        /* v0 = q ux (1-dy)(1-dz)               */ \
  v1 *= v4;        /* v1 = q ux (1+dy)(1-dz)               */ \
  v0 += v5;        /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */ \
  v1 -= v5;        /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */ \
  v2 -= v5;        /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */ \
  v3 += v5;        /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */ \
  a[ii][offset + 0] = v0;                           \
  a[ii][offset + 1] = v1;                           \
  a[ii][offset + 2] = v2;                           \
  a[ii][offset + 3] = v3; 

  // timer_start(save_to_local_a); // 9920
      ACCUMULATE_J(x, y, z, 0);
      ACCUMULATE_J(y, z, x, 4);
      ACCUMULATE_J(z, x, y, 8);
  // timer_end(save_to_local_a);

#undef ACCUMULATE_J
    }

    else{
      local_pm.dispx = ux;
      local_pm.dispy = uy;
      local_pm.dispz = uz;

      local_pm.i = itmp + i + thread_rank;

      // __device__ int move_p_gpu(particle_t *p0, particle_mover_t *pm,
      //   accumulator_t *a0, const int64_t *g_neighbor,
      //   int64_t g_rangel, int64_t g_rangeh, const float qsp) {

      /*if (move_p_gpu(p0, &local_pm, a0, g, qsp))  // Unlikely
      {
        if (nm < max_nm) {
          pm[nm++] = local_pm[0];
        }

        else {
          itmp++;  // Unlikely
        }
      }*/
    }

    p_global[i + thread_rank] = p;
  }
}
    
  __syncthreads();

  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;

  float my_a[SHARE_MAX_VOXEL_SIZE][12];
  __shared__ float res[SHARE_MAX_VOXEL_SIZE][12];
  #pragma unroll
  for(int i = 0;i < SHARE_MAX_VOXEL_SIZE; i++){
    #pragma unroll
    for(int j = 0;j < 12;j++){
      float aggregate = WarpReduce(temp_storage).Sum(my_a[i][j]);
    }
  }

  accumulator_t *a_global = args.a0;
  for(int i = 0;i < SHARE_MAX_VOXEL_SIZE; i++){
    if (thread_rank < 12) {
      float real_a = res[i][thread_rank];
      atomicAdd( ((float *)a_global + f_shared_index[i]) + thread_rank ,
      real_a );
    }
  }
/*  int my_a[SHARE_MAX_VOXEL_SIZE][12];
  __shared__ int res[SHARE_MAX_VOXEL_SIZE][12];
  const float two_p_28 = (float)(1<<28);
  #pragma unroll
  for(int i = 0;i < SHARE_MAX_VOXEL_SIZE; i++){
    #pragma unroll
    for(int j = 0;j < 12;j++){
      my_a[i][j] = a[i][j] * two_p_28;
    }
  }
  #pragma unroll
  for(int i = 0;i < SHARE_MAX_VOXEL_SIZE; i++){
    #pragma unroll
    for(int j = 0;j < 12;j++){
      // float aggregate = WarpReduce(temp_storage).Sum(my_a[i][j]);
      int aggregate = warpAllReduceSum(my_a[i][j]);
      if(thread_rank == 0) res[i][j] = aggregate;
      __syncthreads();
    }
  }

  accumulator_t *a_global = args.a0;
  for(int i = 0;i < SHARE_MAX_VOXEL_SIZE; i++){
    if (thread_rank < 12) {
      float real_a = (float)(res[i][thread_rank] >> 28);
      atomicAdd( ((float *)a_global + f_shared_index[i]) + thread_rank ,
      real_a );
    }
  }*/


}