cmake_minimum_required(VERSION 3.24)
project(shouxiecuda_opencl)
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_STANDARD 14)

#file(COPY model1.onnx DESTINATION .)
file(COPY misc_opencl_matmul.cl DESTINATION .)

find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message("OpenCL_INCLUDE_DIRS = ${OpenCL_INCLUDE_DIRS}")
    message("OpenCL_LIBRARIES = ${OpenCL_LIBRARIES}")
    message("OpenCL_VERSION_STRING = ${OpenCL_VERSION_STRING}")
    message("OpenCL_VERSION_MAJOR = ${OpenCL_VERSION_MAJOR}")
    message("OpenCL_VERSION_MINOR = ${OpenCL_VERSION_MINOR}")

    include_directories(/usr/include ${OpenCL_INCLUDE_DIRS} /data0/shouxiecuda/common)

    link_directories(/usr/lib ${OpenCL_LIBRARIES})
    link_libraries(openblas pthread OpenCL)

    set(SIMDFLAGS "-mavx2 -mavx512f")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${SIMDFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${SIMDFLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    add_executable(misc_opencl_device cl_common.c /data0/shouxiecuda/common/common.cpp misc_opencl_device.cpp)
    add_executable(misc_opencl_sum_array_c cl_common.c /data0/shouxiecuda/common/common.c misc_opencl_sum_array.c)
    add_executable(misc_opencl_sum_array_cpp cl_common.c /data0/shouxiecuda/common/common.cpp misc_opencl_sum_array.cpp)
    add_executable(misc_opencl_matmul cl_common.c /data0/shouxiecuda/common/common.cpp misc_opencl_matmul.cpp)

endif()


