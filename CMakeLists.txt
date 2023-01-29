cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(net_server)

set(CMAKE_PREFIX_PATH "libs/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 ${TORCH_CXX_FLAGS}")

include(${CMAKE_SOURCE_DIR}/PocoConfig.cmake.in)

add_executable(net_server
    src/server.cpp)

target_link_libraries(net_server
    PocoUtil
    PocoNet
    PocoXML
    PocoFoundation
    PocoJSON
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)