cmake_minimum_required ( VERSION 3.15 )

project ( epc_mass_cpp )

#set (CMAKE_CXX_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/bin/pgc++)
#set (CMAKE_CXX_LINK_EXECUTABLE /opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/bin/pgc++)
#add_compile_options(-std=c++11 -Wall -Werror)
set (CMAKE_C_COMPILER /opt/rh/devtoolset-11/root/usr/bin/gcc)
set (CMAKE_CXX_COMPILER /opt/rh/devtoolset-11/root/usr/bin/g++)

set ( PRJ_HEADER_LIST )
set ( PRJ_SRC_LIST )
set ( PRJ_INCLUDE_DIRS )
set ( PRJ_LINK_DIRS )
set ( PRJ_LIBRARIES )
set ( PRJ_COMPILE_OPTIONS )
set ( PRJ_COMPILE_FEATURES )

find_package ( OpenACC )

#set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS}" )

file ( GLOB root_header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.h" )
file ( GLOB root_src_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )

list ( APPEND PRJ_HEADER_LIST ${root_header_files} )
list ( APPEND PRJ_SRC_LIST ${root_src_files} )
#list ( APPEND PRJ_COMPILE_OPTIONS -ta=multicore,tesla )
#list ( APPEND PRJ_COMPILE_OPTIONS  -fast -Minfo -acc )
#list ( APPEND PRJ_COMPILE_FEATURES cxx_std_17 )

add_executable ( ${PROJECT_NAME} ${PRJ_SRC_LIST} ${PRJ_HEADER_LIST} )

target_include_directories ( ${PROJECT_NAME}
        PRIVATE
        ${PRJ_INCLUDE_DIRS}
        )

target_link_directories ( ${PROJECT_NAME}
        PRIVATE
        ${PRJ_LINK_DIRS}
        )

target_compile_options ( ${PROJECT_NAME}
        PRIVATE
        ${PRJ_COMPILE_OPTIONS}
        )

target_compile_features ( ${PROJECT_NAME}
        PUBLIC
        ${PRJ_COMPILE_FEATURES}
        )

target_link_libraries ( ${PROJECT_NAME}
        PRIVATE
        ${PRJ_LIBRARIES}
        )