cmake_minimum_required(VERSION 3.20)
project(dqnn_feeder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# opnic's CMake config pulls in the CUDA target; same arch as opnic_dgx_opx (GH200)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

find_package(OpenSSL REQUIRED)
find_package(opnic CONFIG REQUIRED)

add_executable(dqnn_feeder dqnn_feeder.cpp)
target_link_libraries(dqnn_feeder PRIVATE qm::opnic qm::opnic-cuda)
