#ifndef __NTHU_GPU_H__
#define __NTHU_GPU_H__

#define IN_spa
#include <map>

#include "src/species_advance/standard/pipeline/spa_private.h"

typedef void *device_pointer;
typedef void *host_pointer;

namespace vpic_gpu {
class gpu_memory_allocator {
 private:
  std::map<host_pointer, device_pointer> host_device_map;
  std::map<device_pointer, size_t> device_array_size;

 public:
  gpu_memory_allocator() {}
  device_pointer map_to_device(host_pointer, size_t);  // copy if needed
  device_pointer copy_to_device(host_pointer, size_t);
  void realloc(host_pointer, size_t, size_t);
  void copy_to_host(host_pointer, size_t);
};

extern gpu_memory_allocator gm;

void mpiSetDevice(int rank);
void cudaInitSpeciesStream(species_t *sp_list);

void advance_p_gpu_launcher(advance_p_pipeline_args_t *, species_t *);
void energy_p_gpu_stage_1(species_t *sp_list, interpolator_array_t *ia);
double energy_p_gpu_stage_2(species_t *sp, interpolator_array_t *ia);
void sort_p_gpu_launcher(species_t *);
void accumulate_rho_p_gpu_launcher(field_array_t *, const species_t *);

void boundary_p_get_p_pm(particle_t *p0, particle_mover_t *pm, species_t *sp);
void append_p_and_pm(particle_t *temp_p, particle_mover_t *temp_pm, int pi_cnt, int pm_cnt, species_t *sp);

template <typename T>
void resize_on_device(T *the, size_t original_cnt, size_t new_cnt) {
  gm.realloc(the, sizeof(T) * original_cnt, sizeof(T) * new_cnt);
}

};  // namespace vpic_gpu

#endif