cmake_minimum_required(VERSION 3.25)
#！！！OpenACC_FOUND New in version 3.25.
#！！！export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/comm_libs/mpi/bin:/usr/local/NVIDIA-Nsight-Compute:/opt/nvidia/nsight-systems/systems/bin:/usr/local/bin:/opt/rh/devtoolset-10/root/usr/bin:/opt/rh/devtoolset-7/root/usr/bin:/data0/envs/deepspeed/bin:/usr/local/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/usr/local/cuda/bin:/data0/ant/bin:/usr/local/swig/bin:/root/bin
project(shouxiecuda_openacc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)

#if (NOT CMAKE_BUILD_TYPE)
#    set(CMAKE_BUILD_TYPE Release)
#endif (NOT CMAKE_BUILD_TYPE)
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

#file(COPY model1.onnx DESTINATION .)
#/opt/nvidia/hpc_sdk
#find_package(OpenACC REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenACC REQUIRED)
if(CUDA_FOUND AND OpenMP_FOUND AND OpenACC_FOUND)
    message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
    message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
    message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")

    list(APPEND CUDA_NVCC_FLAGS --extended-lambda -gencode arch=compute_86,code=sm_86 -rdc=true)
    set(CUDA_SEPARABLE_COMPILATION ON)
    #include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/cuda-samples/Common /usr/include common)
    set(HPC_SDK_HOME "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/include")
    set(HPC_SDK_INCLUDE_STDEXEC "${HPC_SDK_HOME}-stdexec")
    set(HPC_SDK_INCLUDE_STDPAR "${HPC_SDK_HOME}-stdpar")
    set(HPC_SDK_INCLUDE_ACC "${HPC_SDK_HOME}_acc")
    set(HPC_SDK_INCLUDE_MAN "${HPC_SDK_HOME}_man")
    #include-stdexec  include-stdpar  include_acc  include_man
    set(LIB_DIR_COMMON "/data0/shouxiecuda/common")
    include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/cuda-samples/Common /usr/include common ${HPC_SDK_INCLUDE_STDEXEC} ${HPC_SDK_INCLUDE_STDPAR} ${HPC_SDK_INCLUDE_ACC} ${HPC_SDK_INCLUDE_MAN} ${LIB_DIR_COMMON})

    link_directories(/usr/local/cuda/lib64 /usr/lib)
    #${OpenACC_CXX_OPTIONS}
    #升级到cmake 3.27提示Policy CMP0146 is not set: The FindCUDA module is removed
    #${CUDA_cublas_LIBRARY}不再被赋值
    link_libraries(cudadevrt ${CUDA_LIBRARIES} cublas openblas)

    message("OpenMP_C_FLAGS = ${OpenMP_C_FLAGS}")
    message("OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")

    message("OpenACC_FOUND = ${OpenACC_FOUND}")
    message("OpenACC_C_FOUND = ${OpenACC_C_FOUND}")
    message("OpenACC_CXX_FOUND = ${OpenACC_CXX_FOUND}")

    message("OpenACC_C_FLAGS = ${OpenACC_C_FLAGS}")
    message("OpenACC_CXX_FLAGS = ${OpenACC_CXX_FLAGS}")
    message("OpenACC_C_OPTIONS = ${OpenACC_C_OPTIONS}")
    message("OpenACC_CXX_OPTIONS = ${OpenACC_CXX_OPTIONS}")

    if (CMAKE_OPENACC_VERBOSE)
        set(OpenACC_C_FLAGS "${OpenACC_C_FLAGS} ${OPENACC_C_VERBOSE}")
    endif (CMAKE_OPENACC_VERBOSE)

    set(SIMDFLAGS "-mavx2 -mavx512f")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${SIMDFLAGS} ${OpenACC_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${SIMDFLAGS} ${OpenACC_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=large -fno-tree-vectorize")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=large -fno-tree-vectorize")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    endif()

    message("CMAKE_C_FLAGS = ${CMAKE_C_FLAGS}")
    message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

    add_executable(epc_mass_sum ${LIB_DIR_COMMON}/common.cpp epc_mass_sum.c epc_mass_main.c)
    set_source_files_properties(epc_mass_sum.c PROPERTIES COMPILE_FLAGS "${OpenACC_C_FLAGS}")
    set_target_properties(epc_mass_sum PROPERTIES LINK_FLAGS "${OpenACC_C_OPTIONS}")

    add_custom_target(distclean COMMAND rm -rf cmake-build-* CMakeCache.txt CMakeFiles
            Makefile cmake_install.cmake *.dSYM ipo_out.optrpt *.o)

endif()


