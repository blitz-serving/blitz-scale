cmake_minimum_required(VERSION 3.13)
project(service LANGUAGES CXX)

set(SERVICE service.cc service.h)
add_library(service SHARED ${SERVICE})
add_dependencies(service generatev2)
set(PROTO_DIR ${CMAKE_BINARY_DIR}/proto)
target_include_directories(service PUBLIC ${CMAKE_SOURCE_DIR}/src/blitz ${PROTO_DIR} ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(service PUBLIC nccl_utils generatev2 ${NCCL_LIBRARIES} blitz_tccl blitz_model_loader)

