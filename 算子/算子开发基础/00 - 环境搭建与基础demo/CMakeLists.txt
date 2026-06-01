cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU arch")
find_package(ASC REQUIRED)
project(add_project LANGUAGES ASC CXX)
add_executable(add_demo add.asc)
target_compile_options(add_demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
