cmake_minimum_required(VERSION 2.8)
project(deformation)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++11 -g")

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

add_library(impl impl.cpp)

add_executable(demo main.cpp)

target_link_libraries(demo impl ${OpenCV_LIBS})
