#include "vpic.h"
#include "../cuda_src/gpu.cuh"
#define FAK field_array->kernel

void vpic_simulation::copy_p_back(void) {
  species_t * sp;
  LIST_FOR_EACH( sp, species_list ) {
    printf("copy %s back\n", sp->name);
    /*
    particle_t * new_p;
    MALLOC_ALIGNED( new_p, sp->max_np, 128 );
    FREE_ALIGNED( sp->p );
    sp->p = new_p;*/
    vpic_gpu::copy_p_back(sp);
  }
}