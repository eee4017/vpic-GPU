#ifndef __NTHU_GPU_H__
#define __NTHU_GPU_H__

#define IN_spa
#include "src/species_advance/standard/pipeline/spa_private.h"
#include <map>

typedef void *device_pointer;
typedef void *host_pointer;

namespace vpic_gpu{
	class gpu_memory_allocator{
	private:
		std::map<host_pointer, device_pointer>host_device_map;
	public:
		gpu_memory_allocator(){}
		device_pointer map_to_device(host_pointer, size_t); // copy if needed
		device_pointer copy_to_device(host_pointer, size_t);
		void realloc(host_pointer, size_t , size_t);
		void copy_to_host(host_pointer, size_t);
	};

	extern gpu_memory_allocator gm;

    void advance_p_gpu_launcher(advance_p_pipeline_args_t *, species_t *);
    void energy_p_gpu_launcher(energy_p_pipeline_args_t *,const species_t *);
	void sort_p_gpu_launcher(species_t *);
    void accumulate_rho_p_gpu_launcher(field_array_t* , const species_t*);

	void boundary_p_get_p_pm(particle_t *p0, particle_mover_t *pm, species_t * sp);
	void append_p_and_pm(particle_t *temp_p, particle_mover_t *temp_pm, int pi_cnt, int pm_cnt, species_t *sp);
	
    template <typename T>
    void resize_on_device(T* the, size_t original_cnt, size_t new_cnt){
      gm.realloc(the, sizeof(T) * original_cnt, sizeof(T) * new_cnt);
    }
	
};


#endif