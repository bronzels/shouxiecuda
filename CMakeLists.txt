cmake_minimum_required(VERSION 3.24)
project(shouxiecuda)
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86 87)

#file(COPY model1.onnx DESTINATION .)

# CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
#set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -std=c++14 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true -gencode arch=compute_86,code=sm_86 -std=c++14")
#The CUDA_NVCC_FLAGS variable is a list not a string.
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86 -rdc=true -std=c++14)
list(APPEND CUDA_NVCC_FLAGS -G -g --extended-lambda -gencode arch=compute_86,code=sm_86 -rdc=true)
set(CUDA_SEPARABLE_COMPILATION ON)
#add_definitions(-DUSE_GPU)
#set(LDFLAGS "$LDFLAGS -lcudadevrt")
file(GLOB COMMON common.h common.cpp)
#link_directories(${CUDA_LIBRARIES})
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")

include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/cuda-samples/Common /usr/include)

link_directories(/usr/local/cuda/lib64 /usr/lib)
#link_libraries(cudadevrt cuda ${CUDA_LIBRARIES})
link_libraries(cudadevrt ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} openblas pthread)

#include_directories(/root/trt/include)
#link_directories(/root/trt/lib)
#link_libraries(nvinfer nvonnxparser ${CUDA_LIBRARIES})

#set_source_files_properties(common.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

cuda_add_executable(master_hello_cuda master_hello_cuda.cu)
cuda_add_executable(master_unique_index master_unique_index.cu)
cuda_add_executable(master_memory_transfer master_memory_transfer.cu)
cuda_add_executable(master_sum_array master_sum_array.cu)
cuda_add_executable(master_device_query master_device_query.cu)
cuda_add_executable(master_warp_id_and_thread_id master_warp_id_and_thread_id.cu)
cuda_add_executable(master_warp_divergency master_warp_divergency.cu)
cuda_add_executable(master_occupancy_test master_occupancy_test.cu)
#cuda_add_executable(master_sum_array_profile ${PROJECT_SOURCE_DIR}/common.cpp master_sum_array_profile.cu cuda_common.cu)
cuda_add_executable(master_sum_array_profile common.cpp master_sum_array_profile.cu cuda_common.cu)
cuda_add_executable(master_reduction common.cpp master_reduction.cu)
cuda_add_executable(master_dynamic_parallelism master_dynamic_parallelism.cu)
cuda_add_executable(master_reduction_default_parallelism common.cpp master_reduction_default_parallelism.cu)
cuda_add_executable(master_register_usage master_register_usage.cu)
cuda_add_executable(master_soa_aos master_soa_aos.cu)
cuda_add_executable(master_matrix_transpose common.cpp master_matrix_transpose.cu)
cuda_add_executable(master_intro_smem common.cpp master_intro_smem.cu)
cuda_add_executable(master_row_column_access_smem common.cpp master_row_column_access_smem.cu)
cuda_add_executable(master_constant_stencil common.cpp master_constant_stencil.cu)
cuda_add_executable(master_shfl common.cpp master_shfl.cu)
cuda_add_executable(master_simple_cuda_stream common.cpp master_simple_cuda_stream.cu)
cuda_add_executable(master_stream_blocking_ornot master_stream_blocking_ornot.cu)
cuda_add_executable(master_elapsed_time_events master_elapsed_time_events.cu)
cuda_add_executable(master_numeric_accuracy master_numeric_accuracy.cu)
cuda_add_executable(master_atomic master_atomic.cu)
cuda_add_executable(master_scan common.cpp master_scan.cu)

cuda_add_executable(misc_matmul common.cpp misc_matmul.cu)
cuda_add_executable(misc_thrust_sort_max common.cpp misc_thrust_sort_max.cu)
cuda_add_executable(misc_thrust_foreach common.cpp dotmethod.cpp misc_thrust_foreach.cu)
cuda_add_executable(misc_thrust_transforms common.cpp misc_thrust_transforms.cu)

cuda_add_executable(tiandiao_vecadd tiandiao_vecadd.cu)
cuda_add_executable(tiandiao_stencil tiandiao_stencil.cu)

cuda_add_executable(nsight_stall_reason_longsb nsight_stall_reason_longsb.cu)
cuda_add_executable(nsight_stall_reason_shortsb nsight_stall_reason_shortsb.cu)
cuda_add_executable(nsight_matrix_transpose nsight_matrix_transpose.cu)

endif()

add_executable(test_clock test_clock.c)

#set_target_properties(hello_cuda PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)