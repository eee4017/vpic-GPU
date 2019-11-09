#define IN_boundary
#include <CppToolkit/debug.h>

#include <vector>
#include <utility>
#include <algorithm>
#include <random>

#include "../cuda_src/gpu.cuh"
#include "boundary_private.h"
#ifdef V4_ACCELERATION
using namespace v4;
#endif

#ifndef MIN_NP
#define MIN_NP 128  // Default to 4kb (~1 page worth of memory)
//#define MIN_NP 32768 // 32768 particles is 1 MiB of memory.
#endif

enum { MAX_PBC = 32, MAX_SP = 32 };

inline void copy_pi_from_p(particle_injector_t *pi, particle_t *p,
                           particle_mover_t *pm) {
  // #ifdef V4_ACCELERATION
  //   copy_4x1(&pi->dx, &p->dx);
  //   copy_4x1(&pi->ux, &p->ux);
  //   copy_4x1(&pi->dispx, &pm->dispx);
  // #else
  pi->dx = p->dx;
  pi->dy = p->dy;
  pi->dz = p->dz;

  pi->ux = p->ux;
  pi->uy = p->uy;
  pi->uz = p->uz;
  pi->w = p->w;

  pi->dispx = pm->dispx;
  pi->dispy = pm->dispy;
  pi->dispz = pm->dispz;
  // #endif
}

inline void copy_p_from_pi(particle_t *p, const particle_injector_t *pi) {
  // #ifdef V4_ACCELERATION
  //   copy_4x1(&p->dx, &pi->dx);
  //   copy_4x1(&p->ux, &pi->ux);
  // #else
  p->dx = pi->dx;
  p->dy = pi->dy;
  p->dz = pi->dz;

  p->i = pi->i;
  p->ux = pi->ux;
  p->uy = pi->uy;
  p->uz = pi->uz;
  p->w = pi->w;
  // #endif
}

inline void copy_pm_from_pi(particle_mover_t *pm,
                            const particle_injector_t *pi) {
  // #ifdef V4_ACCELERATION
  //   copy_4x1(&pm->dispx, &pi->dispx);
  // #else
  pm->dispx = pi->dispx;
  pm->dispy = pi->dispy;
  pm->dispz = pi->dispz;
  // #endif
}


int
move_p_singlge_p (  particle_t       * ALIGNED(128) p_single,
                    particle_mover_t * ALIGNED(16)  pm,
                    accumulator_t    * ALIGNED(128) a0,
                    const grid_t     *              g,
                    const float                     qsp ) {
  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, v4, v5, q;
  int axis, face;
  int64_t neighbor;
  float *a;
  particle_t * ALIGNED(32) p = p_single;

  q = qsp*p->w;

  for(;;) {
    s_midx = p->dx;
    s_midy = p->dy;
    s_midz = p->dz;

    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    s_dir[0] = (s_dispx>0.0f) ? 1.0f : -1.0f;
    s_dir[1] = (s_dispy>0.0f) ? 1.0f : -1.0f;
    s_dir[2] = (s_dispz>0.0f) ? 1.0f : -1.0f;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0.0f) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0.0f) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0.0f) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
    /**/      v3=2.0f, axis=3;
    if(v0<v3) v3=v0,   axis=0;
    if(v1<v3) v3=v1,   axis=1;
    if(v2<v3) v3=v2,   axis=2;
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
    v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);
    a = (float *)(a0 + p->i);
#   define accumulate_j(X,Y,Z)                                        \
    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
    v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
    v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
    v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
    v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
    v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
    v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \
    a[0] += v0;                                                       \
    a[1] += v1;                                                       \
    a[2] += v2;                                                       \
    a[3] += v3
    accumulate_j(x,y,z); a += 4;
    accumulate_j(y,z,x); a += 4;
    accumulate_j(z,x,y);
#   undef accumulate_j

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    // Compute the new particle offset
    p->dx += s_dispx+s_dispx;
    p->dy += s_dispy+s_dispy;
    p->dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)

    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    v0 = s_dir[axis];
    (&(p->dx))[axis] = v0; // Avoid roundoff fiascos--put the particle
                           // _exactly_ on the boundary.
    face = axis; if( v0>0 ) face += 3;
    neighbor = g->neighbor[ 6*p->i + face ];

    if( UNLIKELY( neighbor==reflect_particles ) ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      (&(p->ux    ))[axis] = -(&(p->ux    ))[axis];
      (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
      continue;
    }

    if( UNLIKELY( neighbor<g->rangel || neighbor>g->rangeh ) ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      p->i = 8*p->i + face;
      return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    p->i = neighbor - g->rangel; // Compute local index of neighbor
    /**/                         // Note: neighbor - g->rangel < 2^31 / 6
    (&(p->dx))[axis] = -v0;      // Convert coordinate system
  }

  return 0; // Return "mover not in use"
}


void
boundary_p_gpu( particle_bc_t       * RESTRICT pbc_list,
            species_t           * RESTRICT sp_list,
            field_array_t       * RESTRICT fa,
            accumulator_array_t * RESTRICT aa ) {

  // Gives the local mp port associated with a local face
  static const int f2b[6]  = { BOUNDARY(-1, 0, 0),
                               BOUNDARY( 0,-1, 0),
                               BOUNDARY( 0, 0,-1),
                               BOUNDARY( 1, 0, 0),
                               BOUNDARY( 0, 1, 0),
                               BOUNDARY( 0, 0, 1) };

  // Gives the remote mp port associated with a local face
  static const int f2rb[6] = { BOUNDARY( 1, 0, 0),
                               BOUNDARY( 0, 1, 0),
                               BOUNDARY( 0, 0, 1),
                               BOUNDARY(-1, 0, 0),
                               BOUNDARY( 0,-1, 0),
                               BOUNDARY( 0, 0,-1) };

  // Gives the axis associated with a local face
  static const int axis[6]  = { 0, 1, 2,  0,  1,  2 };

  // Gives the location of sending face on the receiver
  static const float dir[6] = { 1, 1, 1, -1, -1, -1 };

  // Temporary store for local particle injectors
  // FIXME: Ugly static usage
  static particle_injector_t * RESTRICT ALIGNED(16) ci = NULL;
  static int max_ci = 0;

  int n_send[6], n_recv[6], n_ci;

  species_t * sp;
  int face;

  // Check input args

  if( !sp_list ) return; // Nothing to do if no species
  if( !fa || !aa || sp_list->g!=aa->g || fa->g!=aa->g )
    ERROR(( "Bad args" ));

  // Unpack the particle boundary conditions

  particle_bc_func_t pbc_interact[MAX_PBC];
  void * pbc_params[MAX_PBC];
  const int nb = num_particle_bc( pbc_list );
  if( nb>MAX_PBC ) ERROR(( "Update this to support more particle boundary conditions" ));
  for( particle_bc_t * pbc=pbc_list; pbc; pbc=pbc->next ) {
    pbc_interact[-pbc->id-3] = pbc->interact;
    pbc_params[  -pbc->id-3] = pbc->params;
   }

  // Unpack fields

  field_t * RESTRICT ALIGNED(128) f = fa->f;
  grid_t  * RESTRICT              g = fa->g;

  // Unpack accumulator

  accumulator_t * RESTRICT ALIGNED(128) a0 = aa->a;

  // Unpack the grid

  const int64_t * RESTRICT ALIGNED(128) neighbor = g->neighbor;
  /**/  mp_t    * RESTRICT              mp       = g->mp;
  const int64_t rangel = g->rangel;
  const int64_t rangeh = g->rangeh;
  const int64_t rangem = g->range[world_size];
  /*const*/ int bc[6], shared[6];
  /*const*/ int64_t range[6];
  for( face=0; face<6; face++ ) {
    bc[face] = g->bc[f2b[face]];
    shared[face] = (bc[face]>=0) && (bc[face]<world_size) &&
                   (bc[face]!=world_rank);
    if( shared[face] ) range[face] = g->range[bc[face]];
  }

  // Begin receiving the particle counts

  for( face=0; face<6; face++ )
    if( shared[face] ) {
      mp_size_recv_buffer( mp, f2b[face], sizeof(int) );
      mp_begin_recv( mp, f2b[face], sizeof(int), bc[face], f2rb[face] );
    }

  // Load the particle send and local injection buffers

  do {

    particle_injector_t * RESTRICT ALIGNED(16) pi_send[6];

    // Presize the send and injection buffers
    //
    // Each buffer is large enough to hold one injector corresponding
    // to every mover in use (worst case, but plausible scenario in
    // beam simulations, is one buffer gets all the movers).
    //
    // FIXME: We could be several times more efficient in our particle
    // injector buffer sizing here.  Namely, we could create on local
    // injector buffer of nm is size.  All injection for all
    // boundaries would be done here.  The local buffer would then be
    // counted to determine the size of each send buffer.  The local
    // buffer would then move all injectors into the approate send
    // buffers (leaving only the local injectors).  This would require
    // some extra data motion though.  (But would give a more robust
    // implementation against variations in MP implementation.)
    //
    // FIXME: This presizing assumes that custom boundary conditions
    // inject at most one particle per incident particle.  Currently,
    // the invocation of pbc_interact[*] insures that assumption will
    // be satisfied (if the handlers conform that it).  We should be
    // more flexible though in the future (especially given above the
    // above overalloc).

    int nm = 0; LIST_FOR_EACH( sp, sp_list ) nm += sp->nm;

    for( face=0; face<6; face++ )
      if( shared[face] ) {
        mp_size_send_buffer( mp, f2b[face], 16+nm*sizeof(particle_injector_t) );
        pi_send[face] = (particle_injector_t *)(((char *)mp_send_buffer(mp,f2b[face]))+16);
        n_send[face] = 0;
      }

    if( max_ci<nm ) {
      particle_injector_t * new_ci = ci;
      FREE_ALIGNED( new_ci );
      MALLOC_ALIGNED( new_ci, nm, 16 );
      ci     = new_ci;
      max_ci = nm;
    }
    n_ci = 0;

    // For each species, load the movers

    LIST_FOR_EACH(sp, sp_list) {
      const float sp_q = sp->q;
      const int32_t sp_id = sp->id;

      particle_t *RESTRICT ALIGNED(128) p0 = sp->p;
      int np = sp->np;

      particle_mover_t * RESTRICT ALIGNED(16)  pm = sp->pm;
      nm = sp->nm;

      particle_injector_t *RESTRICT ALIGNED(16) pi;
      int i, voxel;
      int64_t nn;

      // typedef std::pair<particle_t, particle_mover_t> pp;
      std::vector<particle_t> preload_p(sp->nm);
      std::vector<particle_mover_t> preload_pm(sp->nm);


      // for (int i = sp->nm - 1; i >= 0; i--) {
      //   int idx = pm[i].i;
      //   preload_p[i] = p0[idx];
      //   preload_pm[i] = pm[i];

      //   np--;
      //   p0[idx] = p0[np];
      // }

      vpic_gpu::boundary_p_get_p_pm(&preload_p[0], &preload_pm[0], sp);
      np -= sp->nm;


      for (int i = 0; i < sp->nm; i++) {
        particle_t *p = &preload_p[i];
        particle_mover_t *pm = &preload_pm[i];
        voxel = p->i;
        face = voxel & 7;
        voxel >>= 3;
        p->i = voxel;
        if(voxel > sp->g->nv){
          MY_MESSAGE(("ERROR voxel %d > %d", voxel, sp->g->nv));
        }
        nn = neighbor[6 * voxel + face];

        // Absorb
        if (nn == absorb_particles) {
          accumulate_rhob(f, p, g, sp_q);
          continue;
        }

        // Send to a neighboring node
        if (((nn >= 0) & (nn < rangel)) | ((nn > rangeh) & (nn <= rangem))) {
          pi = &pi_send[face][n_send[face]++];
          copy_pi_from_p(pi, p, pm);
          (&pi->dx)[axis[face]] = dir[face];
          pi->i = nn - range[face];
          pi->sp_id = sp_id;
          continue;
        }

        nn = -nn - 3;  // Assumes reflective/absorbing are -1, -2
        if ((nn >= 0) & (nn < nb)) {
          n_ci +=
              pbc_interact[nn](pbc_params[nn], sp, p, pm, ci + n_ci, 1, face);
          continue;
        }


        WARNING(
            ("Unknown boundary interaction ... dropping particle "
             "(species=%s)", sp->name) );
      }

      sp->np = np;
      sp->nm = 0;
    }


  } while(0);

  // Finish exchanging particle counts and start exchanging actual
  // particles.

  // Note: This is wasteful of communications.  A better protocol
  // would fuse the exchange of the counts with the exchange of the
  // messages.  in a slightly more complex protocol.  However, the MP
  // API prohibits such a model.  Unfortuantely, refining MP is not
  // much help here.  Under the hood on Roadrunner, the DaCS API also
  // prohibits such (specifically, in both, you can't do the
  // equilvanet of a MPI_Getcount to determine how much data you
  // actually received.

  for( face=0; face<6; face++ )
    if( shared[face] ) {
      *((int *)mp_send_buffer( mp, f2b[face] )) = n_send[face];
      mp_begin_send( mp, f2b[face], sizeof(int), bc[face], f2b[face] );
    }

  for( face=0; face<6; face++ )
    if( shared[face] )  {
      mp_end_recv( mp, f2b[face] );
      n_recv[face] = *((int *)mp_recv_buffer( mp, f2b[face] ));
      mp_size_recv_buffer( mp, f2b[face],
                           16+n_recv[face]*sizeof(particle_injector_t) );
      mp_begin_recv( mp, f2b[face], 16+n_recv[face]*sizeof(particle_injector_t),
                     bc[face], f2rb[face] );
    }

  for( face=0; face<6; face++ )
    if( shared[face] ) {
      mp_end_send( mp, f2b[face] );
      // FIXME: ASSUMES MP WON'T MUCK WITH REST OF SEND BUFFER. IF WE
      // DID MORE EFFICIENT MOVER ALLOCATION ABOVE, THIS WOULD BE
      // ROBUSTED AGAINST MP IMPLEMENTATION VAGARIES
      mp_begin_send( mp, f2b[face], 16+n_send[face]*sizeof(particle_injector_t),
                     bc[face], f2b[face] );
    }

# ifndef DISABLE_DYNAMIC_RESIZING

# endif

  do {

    // Unpack the species list for random acesss

    particle_t       * RESTRICT ALIGNED(32) sp_p[ MAX_SP];
    particle_mover_t * RESTRICT ALIGNED(32) sp_pm[MAX_SP];
    float sp_q[MAX_SP];
    int sp_np[MAX_SP];
    int sp_nm[MAX_SP];

#   ifdef DISABLE_DYNAMIC_RESIZING
    int sp_max_np[64], n_dropped_particles[64];
    int sp_max_nm[64], n_dropped_movers[64];
#   endif

    if( num_species( sp_list ) > MAX_SP )
      ERROR(( "Update this to support more species" ));
    LIST_FOR_EACH( sp, sp_list ) {
      sp_p[  sp->id ] = sp->p;
      sp_pm[ sp->id ] = sp->pm;
      sp_q[  sp->id ] = sp->q;
      sp_np[ sp->id ] = sp->np;
      sp_nm[ sp->id ] = sp->nm;
#     ifdef DISABLE_DYNAMIC_RESIZING
      sp_max_np[sp->id]=sp->max_np; n_dropped_particles[sp->id]=0;
      sp_max_nm[sp->id]=sp->max_nm; n_dropped_movers[sp->id]=0;
#     endif
    }

    // Inject particles.  We do custom local injection first to
    // increase message overlap opportunities.
    std::vector<particle_t> append_p[MAX_SP];
    std::vector<particle_mover_t> append_pm[MAX_SP];

    face = 5;
    do {
             particle_t          * RESTRICT ALIGNED(32) p;
             particle_mover_t    * RESTRICT ALIGNED(16) pm;
      const particle_injector_t * RESTRICT ALIGNED(16) pi;
      int np, nm, n, id;

      face++; if( face==7 ) face = 0;
      if( face==6 ) pi = ci, n = n_ci;
      else if( shared[face] ) {
        mp_end_recv( mp, f2b[face] );
        pi = (const particle_injector_t *)
          (((char *)mp_recv_buffer(mp,f2b[face]))+16);
        n  = n_recv[face];
      } else continue;

      // Reverse order injection is done to reduce thrashing of the
      // particle list (particles are removed reverse order so the
      // overall impact of removal + injection is to keep injected
      // particles in order).
      //
      // WARNING: THIS TRUSTS THAT THE INJECTORS (INCLUDING THOSE
      // RECEIVED FROM OTHER NODES) HAVE VALID PARTICLE IDS.

      
      pi += n-1;
      for( ; n; pi--, n-- ) {
        id = pi->sp_id;
        np = sp_np[id];
        nm = sp_nm[id];

        particle_t the_p;
        particle_mover_t the_pm;

        copy_p_from_pi(&the_p, pi);
        copy_pm_from_pi(&the_pm, pi);

        sp_np[id] = np + 1;
        the_pm.i = np;

        if( move_p_singlge_p(&the_p, &the_pm, a0, g, sp_q[id]) ){
          sp_nm[id] = nm + 1;
          append_pm[id].push_back(the_pm);
        }
        
        append_p[id].push_back(the_p);
      }
    } while(face!=5);


    LIST_FOR_EACH( sp, sp_list ) {
#     ifdef DISABLE_DYNAMIC_RESIZING
      if( n_dropped_particles[sp->id] )
        WARNING(( "Dropped %i particles from species \"%s\".  Use a larger "
                  "local particle allocation in your simulation setup for "
                  "this species on this node.",
                  n_dropped_particles[sp->id], sp->name ));
      if( n_dropped_movers[sp->id] )
        WARNING(( "%i particles were not completed moved to their final "
                  "location this timestep for species \"%s\".  Use a larger "
                  "local particle mover buffer in your simulation setup "
                  "for this species on this node.",
                  n_dropped_movers[sp->id], sp->name ));
#     endif
      sp->np=sp_np[sp->id];
      sp->nm=sp_nm[sp->id];
    }


    LIST_FOR_EACH( sp, sp_list ) {
      int i = sp->id;
      vpic_gpu::append_p_and_pm(&append_p[i][0], &append_pm[i][0],
                                 append_p[i].size(), append_pm[i].size(), sp);
      // if(append_p[i].size()){
      //   std::copy(append_p[i].begin(), append_p[i].end(),
      //             sp_p[i] + sp_np[i] - append_p[i].size()); 
      // }
      // if(append_pm[i].size()){
      //   std::copy(append_pm[i].begin(), append_pm[i].end(), 
      //             sp_pm[i] + sp_nm[i] - append_pm[i].size()); 
      // }
    }


  } while(0);

  for( face=0; face<6; face++ )
    if( shared[face] ) mp_end_send(mp,f2b[face]);
  

}
