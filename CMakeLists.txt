cmake_minimum_required(VERSION 3.10)
project(matmul LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -fno-omit-frame-pointer")  # -g keeps symbols for perf
find_package(Threads REQUIRED)

add_executable(matmul main.cpp)
target_link_libraries(matmul Threads::Threads)

add_executable(false_sharing false_sharing.cpp)
target_link_libraries(false_sharing Threads::Threads)

# gprof build: cmake -DENABLE_GPROF=ON ..
option(ENABLE_GPROF "Build with -pg" OFF)
if(ENABLE_GPROF)
  target_compile_options(matmul PRIVATE -pg)
  target_link_options(matmul PRIVATE -pg)
endif()

add_executable(cache_sim cache_sim.cpp)
