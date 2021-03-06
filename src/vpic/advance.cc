/* 
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Heavily revised and extended from earlier V4PIC versions
 *
 */

#include "vpic.h"
#include "../cuda_src/gpu.cuh"
#define FAK field_array->kernel

int vpic_simulation::advance(void) {
  species_t *sp;
  double err;

#ifdef USE_GPU
  vpic_gpu::mpiSetDevice(world_rank);
  if(step() == 0){
    vpic_gpu::cudaInitSpeciesStream(species_list, accumulator_array, interpolator_array);
    LIST_FOR_EACH(sp , species_list){
        MY_MESSAGE(("Particle [%s]:\tsp->np %d,\tg->nv: %d", sp->name, sp->np, sp->g->nv));
    }
  }
#endif
  // Determine if we are done ... see note below why this is done here

  if( num_step>0 && step()>=num_step ) return 0;

  // Sort the particles for performance if desired.
#ifdef USE_GPU
  LIST_FOR_EACH( sp, species_list )
    if( (sp->sort_interval>0) && ((step() % sp->sort_interval)==0) ) {
      if( rank()==0 ) MESSAGE(( "Performance sorting \"%s\"", sp->name ));
      TIC sort_p( sp ); TOC( sort_p, 1 );
    } 
#endif

  // At this point, fields are at E_0 and B_0 and the particle positions
  // are at r_0 and u_{-1/2}.  Further the mover lists for the particles should
  // empty and all particles should be inside the local computational domain.
  // Advance the particle lists.
#ifndef USE_GPU
  if( species_list )
    TIC clear_accumulator_array( accumulator_array ); TOC( clear_accumulators, 1 );
#endif
  // Note: Particles should not have moved since the last performance sort
  // when calling collision operators.
  // FIXME: Technically, this placement of the collision operators only
  // yields a first order accurate Trotter factorization (not a second
  // order accurate factorization).

  if( collision_op_list )
    TIC apply_collision_op_list( collision_op_list ); TOC( collision_model, 1 );
  TIC user_particle_collisions(); TOC( user_particle_collisions, 1 );

#ifdef USE_GPU
  vpic_gpu::advance_p_gpu_launcher(species_list ,accumulator_array, interpolator_array);
#else
  LIST_FOR_EACH( sp, species_list )
    TIC advance_p( sp, accumulator_array, interpolator_array ); TOC( advance_p, 1 );
#endif

  // Because the partial position push when injecting aged particles might
  // place those particles onto the guard list (boundary interaction) and
  // because advance_p requires an empty guard list, particle injection must
  // be done after advance_p and before guard list processing. Note:
  // user_particle_injection should be a stub if species_list is empty.

  if( emitter_list )
    TIC apply_emitter_list( emitter_list ); TOC( emission_model, 1 );
  TIC user_particle_injection(); TOC( user_particle_injection, 1 );

  // This should be after the emission and injection to allow for the
  // possibility of thread parallelizing these operations
#ifndef USE_GPU
  if( species_list )
    TIC reduce_accumulator_array( accumulator_array ); TOC( reduce_accumulators, 1 );
#endif
  // At this point, most particle positions are at r_1 and u_{1/2}. Particles
  // that had boundary interactions are now on the guard list. Process the
  // guard lists. Particles that absorbed are added to rhob (using a corrected
  // local accumulation).
#ifdef USE_GPU
  TIC
    for( int round=0; round<num_comm_round; round++ ){
      boundary_p_gpu( particle_bc_list, species_list,
                      field_array, accumulator_array );
    }
  TOC( boundary_p, num_comm_round );
#else
  TIC
    for( int round=0; round<num_comm_round; round++ )
      boundary_p( particle_bc_list, species_list,
                  field_array, accumulator_array );
  TOC( boundary_p, num_comm_round );
#endif

  LIST_FOR_EACH( sp, species_list ) {
    if( sp->nm && verbose ){
      WARNING(( "Removing %i particles associated with unprocessed %s movers (increase num_comm_round)",
                sp->nm, sp->name ));
    }

    // Drop the particles that have unprocessed movers due to a user defined
    // boundary condition. Particles of this type with unprocessed movers are
    // in the list of particles and move_p has set the voxel in the particle to
    // 8*voxel + face. This is an incorrect voxel index and in many cases can
    // in fact go out of bounds of the voxel indexing space. Removal is in
    // reverse order for back filling. Particle charge is accumulated to the
    // mesh before removing the particle.
    int nm = sp->nm;
    // particle_mover_t * RESTRICT ALIGNED(16)  pm = sp->pm + sp->nm - 1;
    // particle_t * RESTRICT ALIGNED(128) p0 = sp->p;

    std::vector<particle_t> preload_p(sp->nm);
    std::vector<particle_mover_t> preload_pm(sp->nm);
    vpic_gpu::boundary_p_get_p_pm(&preload_p[0], &preload_pm[0], sp);

    for (int i = 0;i < sp->nm; i++) {
      preload_p[i].i >>= 3; // shift particle voxel down
      // accumulate the particle's charge to the mesh
      accumulate_rhob( field_array->f, &preload_p[i], sp->g, sp->q );
      sp->np --; // decrement the number of particles
    }
    sp->nm = 0;
  }


#ifdef USE_GPU
  vpic_gpu::energy_p_gpu_stage_1( species_list, interpolator_array );
#endif
  // At this point, all particle positions are at r_1 and u_{1/2}, the
  // guard lists are empty and the accumulators on each processor are current.
  // Convert the accumulators into currents.

  TIC FAK->clear_jf( field_array ); TOC( clear_jf, 1 );
  if( species_list )
    TIC unload_accumulator_array( field_array, accumulator_array ); TOC( unload_accumulator, 1 );
  TIC FAK->synchronize_jf( field_array ); TOC( synchronize_jf, 1 );

  // At this point, the particle currents are known at jf_{1/2}.
  // Let the user add their own current contributions. It is the users
  // responsibility to insure injected currents are consistent across domains.
  // It is also the users responsibility to update rhob according to
  // rhob_1 = rhob_0 + div juser_{1/2} (corrected local accumulation) if
  // the user wants electric field divergence cleaning to work.

  TIC user_current_injection(); TOC( user_current_injection, 1 );

  // Half advance the magnetic field from B_0 to B_{1/2}

  TIC FAK->advance_b( field_array, 0.5 ); TOC( advance_b, 1 );

  // Advance the electric field from E_0 to E_1

  TIC FAK->advance_e( field_array, 1.0 ); TOC( advance_e, 1 );

  // Let the user add their own contributions to the electric field. It is the
  // users responsibility to insure injected electric fields are consistent
  // across domains.

  TIC user_field_injection(); TOC( user_field_injection, 1 );

  // Half advance the magnetic field from B_{1/2} to B_1

  TIC FAK->advance_b( field_array, 0.5 ); TOC( advance_b, 1 );

  // Divergence clean e

  if( (clean_div_e_interval>0) && ((step() % clean_div_e_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Divergence cleaning electric field" ));

    TIC FAK->clear_rhof( field_array ); TOC( clear_rhof,1 );
    
    if( species_list ) 
      TIC LIST_FOR_EACH( sp, species_list ) {
#ifdef USE_GPU
       vpic_gpu::accumulate_rho_p_gpu_launcher( field_array, sp );
#else
        accumulate_rho_p( field_array, sp );
#endif
       } TOC( accumulate_rho_p, species_list->id );

    TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );

    for( int round=0; round<num_div_e_round; round++ ) {
      TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
      if( round==0 || round==num_div_e_round-1 ) {
        TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
        if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
      }
      TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );
    }
  }

  // Divergence clean b

  if( (clean_div_b_interval>0) && ((step() % clean_div_b_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Divergence cleaning magnetic field" ));

    for( int round=0; round<num_div_b_round; round++ ) {
      TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
      if( round==0 || round==num_div_b_round-1 ) {
        TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
        if( rank()==0 ) MESSAGE(( "%s rms error = %e (charge/volume)", round==0 ? "Initial" : "Cleaned", err ));
      }
      TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );
    }
  }

  // Synchronize the shared faces

  if( (sync_shared_interval>0) && ((step() % sync_shared_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Synchronizing shared tang e, norm b, rho_b" ));
    TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
    if( rank()==0 ) MESSAGE(( "Domain desynchronization error = %e (arb units)", err ));
  }

  // Fields are updated ... load the interpolator for next time step and
  // particle diagnostics in user_diagnostics if there are any particle
  // species to worry about

  if( species_list ) TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );

  step()++;

  // Print out status

  if( (status_interval>0) && ((step() % status_interval)==0) ) {
    if( rank()==0 ) MESSAGE(( "Completed step %i of %i", step(), num_step ));
    update_profile( rank()==0 );
  }

  // Let the user compute diagnostics

  TIC user_diagnostics(); TOC( user_diagnostics, 1 );

  // "return step()!=num_step" is more intuitive. But if a checkpt
  // saved in the call to user_diagnostics() above, is done on the final step
  // (silly but it might happen), the test will be skipped on the restore. We
  // return true here so that the first call to advance after a restore
  // will act properly for this edge case.

  //MESSAGE(("%d\n", need_dump));
  if(dont_dump == 0) dump_energies("energy.txt", 1);
  return 1;
}
