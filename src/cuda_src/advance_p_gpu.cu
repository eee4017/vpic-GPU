#define IN_spa

#include <cub/cub.cuh>
#include "advance_p_gpu.cuh"
#include "gpu_util.cuh"

#define SHARE_MAX_VOXEL_SIZE 2    // 18

__global__ void advance_p_gpu(advance_p_gpu_args args) {
  const int block_rank = blockIdx.x;
  const int n_block = gridDim.x;
  const int thread_rank = threadIdx.x;
  const int n_thread = blockDim.x;
  const int block_size = args.block_size;

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
  int itmp, n, nm, max_nm;

  GPU_DISTRIBUTE(args.np, block_size, block_rank, itmp, n);
  particle_t *p_global = args.p0 + itmp;
  accumulator_t *a_global = args.a0;
  const interpolator_t *f_global = args.f0;
  int prev_i = -1;

  if (itmp + thread_rank < args.np) {
    particle_t p = p_global[thread_rank];
    interpolator_t f = f_global[p.i];

    dx = p.dx;  // Load position
    dy = p.dy;
    dz = p.dz;

    hax = qdt_2mc * ((f.ex + dy * f.dexdy) + dz * (f.dexdz + dy * f.d2exdydz));
    hay = qdt_2mc * ((f.ey + dz * f.deydz) + dx * (f.deydx + dz * f.d2eydzdx));
    haz = qdt_2mc * ((f.ez + dx * f.dezdx) + dy * (f.dezdy + dx * f.d2ezdxdy));

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

    v0 = qdt_2mc / sqrtf(one + (ux * ux + (uy * uy + uz * uz)));

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
      // Common case (inbnds).  Note: accumulator values are 4 times
      // the total physical charge that passed through the appropriate
      // current quadrant in a time-step.

      q *= qsp;

      p.dx = v3;  // Store new position
      p.dy = v4;
      p.dz = v5;

      dx = v0;  // Streak midpoint
      dy = v1;
      dz = v2;

      v5 = q * ux * uy * uz * one_third;  // Compute correction

      float *a = (float *)(a_global + p.i);  // Get accumulator


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
  atomicAdd( a + offset + 0, v0 ); \
  atomicAdd( a + offset + 1, v1 ); \
  atomicAdd( a + offset + 2, v2 ); \
  atomicAdd( a + offset + 3, v3 ); 

      ACCUMULATE_J(x, y, z, 0);
      ACCUMULATE_J(y, z, x, 4);
      ACCUMULATE_J(z, x, y, 8);

#undef ACCUMULATE_J


    }

    else {
      /*local_pm->dispx = ux;
      local_pm->dispy = uy;
      local_pm->dispz = uz;

      local_pm->i = p - p0;

      if (move_p(p0, local_pm, a0, g, qsp))  // Unlikely
      {
        if (nm < max_nm) {
          pm[nm++] = local_pm[0];
        }

        else {
          itmp++;  // Unlikely
        }
      }*/
    }

    p_global[thread_rank] = p;
  }

}