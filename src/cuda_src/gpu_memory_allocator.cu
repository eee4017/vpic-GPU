#include "gpu.cuh"
#include "gpu_util.cuh"
#include <map>
#include <cassert>
using namespace vpic_gpu;
using namespace std;

gpu_memory_allocator gm;

device_pointer gpu_memory_allocator::map_to_device(host_pointer ptr, size_t size = 0){
    auto it = host_device_map.find(ptr);
    if(it != host_device_map.end()) {
        return it->second;
    }

    assert(size != 0);
    return copy_to_device(ptr, size);
}
device_pointer gpu_memory_allocator::copy_to_device(host_pointer ptr, size_t size){
    device_pointer dev_ptr;

    auto it = host_device_map.find(ptr);
    if(it == host_device_map.end()){
        gpuErrchk( cudaMalloc(&dev_ptr, size) );
        host_device_map.insert(make_pair(ptr, dev_ptr));
    }else {
        dev_ptr = it->second;
    }
    
    gpuErrchk( cudaMemcpy(dev_ptr, ptr, size, cudaMemcpyHostToDevice) );
    return dev_ptr;
}
void gpu_memory_allocator::realloc(host_pointer ptr, size_t original_size, size_t new_size){
    auto it = host_device_map.find(ptr);
    assert(it != host_device_map.end());

    device_pointer original_array = it->second;
    device_pointer new_array;
    gpuErrchk( cudaMalloc(&new_array, new_size) );
    gpuErrchk( cudaMemcpy(new_array, original_array, original_size, cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaFree(original_array) );
}
void gpu_memory_allocator::copy_to_host(host_pointer ptr, size_t size){
    auto it = host_device_map.find(ptr);
    assert(it != host_device_map.end());

    device_pointer dev_ptr = it->second;
    gpuErrchk( cudaMemcpy(ptr, dev_ptr, size, cudaMemcpyDeviceToHost) );
}
