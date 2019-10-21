#define IN_boundary
#include "boundary_private.h"
#include "../cuda_src/gpu.cuh"

#ifdef V4_ACCELERATION
using namespace v4;
#endif

#ifndef MIN_NP
#define MIN_NP 128 // Default to 4kb (~1 page worth of memory)
//#define MIN_NP 32768 // 32768 particles is 1 MiB of memory.
#endif


enum { MAX_PBC = 32, MAX_SP = 32 };

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

    LIST_FOR_EACH( sp, sp_list ) {

      const float   sp_q  = sp->q;
      const int32_t sp_id = sp->id;

      //--- particle_t * RESTRICT ALIGNED(128) p0 = sp->p;
      int np = sp->np;

      //--- particle_mover_t * RESTRICT ALIGNED(16)  pm = sp->pm + sp->nm - 1;
      nm = sp->nm;

      //++++++++++++++++++++++++++++++++++
      particle_t *  ALIGNED(128) p0;
      particle_mover_t *  ALIGNED(16) pm;
      particle_mover_t *  ALIGNED(16) pm_dul; //for free purpose

      //allocate host p0/pm
      MALLOC_ALIGNED(p0, nm, 128);
      MALLOC_ALIGNED(pm, nm, 128);
      pm_dul = pm;

      //This function transfers boundary p/pm to host with device back-filling 
      vpic_gpu::boundary_p_get_p_pm(p0, pm, sp);
      pm = pm + sp->nm - 1;

      //++++++++++++++++++++++++++++++++++

      particle_injector_t * RESTRICT ALIGNED(16) pi;
      int i, voxel;
      int64_t nn;


      for( ; nm; pm--, nm-- ) {
        //--- i = pm->i;
        //--- voxel = p0[i].i;
        //++++++++++++++++++++++++++++++++++
        voxel = p0[nm-1].i;
        //++++++++++++++++++++++++++++++++++
        face = voxel & 7;
        voxel >>= 3;
        //--- p0[i].i = voxel;
        //++++++++++++++++++++++++++++++++++
        p0[nm-1].i = voxel;
        //++++++++++++++++++++++++++++++++++
        nn = neighbor[ 6*voxel + face ];

        // Absorb

        if( nn==absorb_particles ) {
          // Ideally, we would batch all rhob accumulations together
          // for efficiency
          //--- accumulate_rhob( f, p0+i, g, sp_q );
          //--- goto backfill;
          //++++++++++++++++++++++++++++++++++


          accumulate_rhob( f, p0+(nm-1), g, sp_q );
          goto next;
          //++++++++++++++++++++++++++++++++++
        }

        // Send to a neighboring node

        if( ((nn>=0) & (nn< rangel)) | ((nn>rangeh) & (nn<=rangem)) ) {

          pi = &pi_send[face][n_send[face]++];
#         ifdef V4_ACCELERATION
          //--- copy_4x1( &pi->dx,    &p0[i].dx  );
          //--- copy_4x1( &pi->ux,    &p0[i].ux  );
          //--- copy_4x1( &pi->dispx, &pm->dispx );
          //++++++++++++++++++++++++++++++++++
          copy_4x1( &pi->dx,    &p0[nm-1].dx  );
          copy_4x1( &pi->ux,    &p0[nm-1].ux  );
          copy_4x1( &pi->dispx, &pm->dispx );
          //++++++++++++++++++++++++++++++++++
#         else
          //---pi->dx=p0[i].dx; pi->dy=p0[i].dy; pi->dz=p0[i].dz;
          //---pi->ux=p0[i].ux; pi->uy=p0[i].uy; pi->uz=p0[i].uz; pi->w=p0[i].w;
          //---pi->dispx = pm->dispx; pi->dispy = pm->dispy; pi->dispz = pm->dispz;
          //++++++++++++++++++++++++++++++++++
          pi->dx=p0[nm-1].dx; pi->dy=p0[nm-1].dy; pi->dz=p0[nm-1].dz;
          pi->ux=p0[nm-1].ux; pi->uy=p0[nm-1].uy; pi->uz=p0[nm-1].uz; pi->w=p0[nm-1].w;
          pi->dispx = pm->dispx; pi->dispy = pm->dispy; pi->dispz = pm->dispz;
          //++++++++++++++++++++++++++++++++++
#         endif
          (&pi->dx)[axis[face]] = dir[face];
          pi->i                 = nn - range[face];
          pi->sp_id             = sp_id;
          //---goto backfill;
          //++++++++++++++++++++++++++++++++++
          goto next;
          //++++++++++++++++++++++++++++++++++
        }

        // User-defined handling


        nn = -nn - 3; // Assumes reflective/absorbing are -1, -2
        if( (nn>=0) & (nn<nb) ) {
          //--- n_ci += pbc_interact[nn]( pbc_params[nn], sp, p0+i, pm,
          //---                          ci+n_ci, 1, face );
          //--- goto backfill;
          //++++++++++++++++++++++++++++++++++
          n_ci += pbc_interact[nn]( pbc_params[nn], sp, p0+(nm-1), pm,
                                    ci+n_ci, 1, face );
          goto next;
          //++++++++++++++++++++++++++++++++++
        }

        // Uh-oh: We fell through

        WARNING(( "Unknown boundary interaction ... dropping particle "
                  "(species=%s)", sp->name ));

      //--- backfill:
//#       //--- ifdef V4_ACCELERATION
        //--- copy_4x1( &p0[i].dx, &p0[np].dx );
        //--- copy_4x1( &p0[i].ux, &p0[np].ux );
//#       //--- else
        //--- p0[i] = p0[np];
//#       //--- endif

        //++++++++++++++++++++++++++++++++++
        next:
          np--;
        //++++++++++++++++++++++++++++++++++
      }

      sp->np = np;
      sp->nm = 0;

    }

  } while(0);


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

      mp_begin_send( mp, f2b[face], 16+n_send[face]*sizeof(particle_injector_t),
                     bc[face], f2b[face] );
    }

# ifndef DISABLE_DYNAMIC_RESIZING
  // Resize particle storage to accomodate worst case inject
    //++++
    int max_inj;
    //++++

  do {
    int n, nm;


    //--- int max_inj = n_ci;
    max_inj = n_ci;

    for( face=0; face<6; face++ )
      if( shared[face] ) max_inj += n_recv[face];

    LIST_FOR_EACH( sp, sp_list ) {
      particle_mover_t * new_pm;
      particle_t * new_p;

      n = sp->np + max_inj;
      if( n>sp->max_np ) {
        n += 0.3125*n;
        WARNING(( "Resizing local %s particle storage from %i to %i",
                  sp->name, sp->max_np, n ));

        sp->max_np = n;
        vpic_gpu::resize_on_device(sp->p, sp->np, sp->max_np);
      }

      nm = sp->nm + max_inj;
      if( nm>sp->max_nm ) {
        nm += 0.3125*nm; // See note above
        WARNING(( "This happened.  Resizing local %s mover storage from "
                    "%i to %i based on not enough movers",
                  sp->name, sp->max_nm, nm ));
        //++++++++++++++++++++++++++++++++++
        sp->max_nm = nm;
        vpic_gpu::resize_on_device(sp->pm, sp->nm, sp->max_nm);
        //++++++++++++++++++++++++++++++++++
      }
    }
  } while(0);
# endif


  //++++++++++++++++++++++++++++++++++
    //host temp storage of received p
    particle_t *  ALIGNED(32) temp_p[MAX_SP];

    //host temp storage of received pm
    particle_mover_t *  ALIGNED(32) temp_pm[MAX_SP];

    //host received pi count
    int pi_cnt[MAX_SP]; for(int i=0;i<MAX_SP;i++) pi_cnt[i] = 0;

    //host received pm count
    int pm_cnt[MAX_SP]; for(int i=0;i<MAX_SP;i++) pm_cnt[i] = 0;
    //++++++++++++++++++++++++++++++++++


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

    face = 5;
    do {
      /**/  particle_t          * RESTRICT ALIGNED(32) p;
      /**/  particle_mover_t    * RESTRICT ALIGNED(16) pm;
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

      pi += n-1;
      for( ; n; pi--, n-- ) {
        id = pi->sp_id;
        //--- p  = sp_p[id]; 
        np = sp_np[id];
        particle_t          * RESTRICT ALIGNED(32) current_temp_p = temp_p[id];
        

        //++++++++++++++++++++++++++++++++++
        //allocate temp storage of this species, if have'nt
        if(pi_cnt[id] == 0){
          MALLOC_ALIGNED(temp_p[id], max_inj, 128 );
        }
        if(pi_cnt[id] == 0){
          MALLOC_ALIGNED(temp_pm[id], max_inj, 128 );
        }
        //++++++++++++++++++++++++++++++++++
        
        //--- pm = sp_pm[id]; 
        nm = sp_nm[id];

#       ifdef DISABLE_DYNAMIC_RESIZING
        if( np>=sp_max_np[id] ) { n_dropped_particles[id]++; continue; }
#       endif
#       ifdef V4_ACCELERATION
        //--- copy_4x1(  &p[np].dx,    &pi->dx    );
        //--- copy_4x1(  &p[np].ux,    &pi->ux    );
        //++++++++++++++++++++++++++++++++++
        copy_4x1(  &temp_p[id][pi_cnt[id]].dx,    &pi->dx    );
        copy_4x1(  &temp_p[id][pi_cnt[id]].ux,    &pi->ux    );
        //++++++++++++++++++++++++++++++++++
#       else
        //--- p[np].dx=pi->dx; p[np].dy=pi->dy; p[np].dz=pi->dz; p[np].i=pi->i;
        //--- p[np].ux=pi->ux; p[np].uy=pi->uy; p[np].uz=pi->uz; p[np].w=pi->w;
        //++++++++++++++++++++++++++++++++++
        temp_p[id][pi_cnt[id]].dx=pi->dx; temp_p[id][pi_cnt[id]].dy=pi->dy; temp_p[id][pi_cnt[id]].dz=pi->dz; temp_p[id][pi_cnt[id]].i=pi->i;
        temp_p[id][pi_cnt[id]].ux=pi->ux; temp_p[id][pi_cnt[id]].uy=pi->uy; temp_p[id][pi_cnt[id]].uz=pi->uz; temp_p[id][pi_cnt[id]].w=pi->w;
        //++++++++++++++++++++++++++++++++++
#       endif
        sp_np[id] = np+1;

#       ifdef DISABLE_DYNAMIC_RESIZING
        if( nm>=sp_max_nm[id] ) { n_dropped_movers[id]++;    continue; }
#       endif
#       ifdef V4_ACCELERATION
        //--- copy_4x1( &pm[nm].dispx, &pi->dispx );
        //--- pm[nm].i = np;
        //++++++++++++++++++++++++++++++++++
        copy_4x1( &temp_pm[id][pm_cnt[id]].dispx, &pi->dispx );
        temp_pm[id][pm_cnt[id]].i = np;
        //++++++++++++++++++++++++++++++++++
#       else
        //--- pm[nm].dispx=pi->dispx; pm[nm].dispy=pi->dispy; pm[nm].dispz=pi->dispz;
        //--- pm[nm].i=np;
        //++++++++++++++++++++++++++++++++++
        temp_pm[id][pm_cnt[id]].dispx=pi->dispx; temp_pm[id][pm_cnt[id]].dispy=pi->dispy; temp_pm[id][pm_cnt[id]].dispz=pi->dispz;
        temp_pm[id][pm_cnt[id]].i = np;
        //++++++++++++++++++++++++++++++++++
#       endif
        //--- sp_nm[id] = nm + move_p( p, pm+nm, a0, g, sp_q[id] );
        //+++ 
        int tempi = temp_pm[id][pm_cnt[id]].i;
        temp_pm[id][pm_cnt[id]].i = pi_cnt[id];

        int move_p_cnt = move_p( temp_p[id], temp_pm[id]+pm_cnt[id], a0, g, sp_q[id] );

        // if((temp_p[id] + pi_cnt[id])->i > 161994){
        //   printf(RED"Error!: %d"COLOR_END"\n", (temp_p[id] + pi_cnt[id])->i);
        // }

        sp_nm[id] = nm + move_p_cnt;
        temp_pm[id][pm_cnt[id]].i = tempi;
        //+++

        //+++
        pi_cnt[id]++;
        pm_cnt[id]+=move_p_cnt;
        //+++
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

  } while(0);

  for( face=0; face<6; face++ )
    if( shared[face] ) mp_end_send(mp,f2b[face]);

  //++++++++++++++++++++++++++++++++++
  //TODO - append p and pm to device's list here
  LIST_FOR_EACH( sp, sp_list ) {
    int id = sp->id;
    vpic_gpu::append_p_and_pm(temp_p[id], temp_pm[id], pi_cnt[id], pm_cnt[id], sp);
  }
  //++++++++++++++++++++++++++++++++++

  //++++++++++++++++++
  //free temp storage
    for(int i=0; i<MAX_SP; i++){
      if(pi_cnt[i] == 0)
        continue;
      FREE_ALIGNED( temp_p[i] );
      FREE_ALIGNED( temp_pm[i] );

    }
  //++++++++++++++++++
}
