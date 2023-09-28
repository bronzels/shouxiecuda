cmake_minimum_required(VERSION 3.23)
project(shouxiecuda_openacc)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

ADD_SUBDIRECTORY(./c)
ADD_SUBDIRECTORY(./cplus)
