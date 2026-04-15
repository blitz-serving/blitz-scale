include_directories(${CMAKE_SOURCE_DIR}/src/util)

add_library(blitz_tccl STATIC ./blitz_tccl.cpp)
target_link_libraries(blitz_tccl PUBLIC pickle MPI::MPI_CXX numa)
target_include_directories(blitz_tccl PUBLIC ${CMAKE_SOURCE_DIR}/src/include)
set_target_properties(blitz_tccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(blitz_model_loader ./model_loader.cc)
set_target_properties(blitz_model_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(BLITZ_COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/blitz
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/proto)

add_library(blitz_param_transfer ./param_transfer.cc)
target_include_directories(blitz_param_transfer PUBLIC ${BLITZ_COMMON_INCLUDES})
target_link_libraries(blitz_param_transfer PUBLIC blitz_tccl)
set_target_properties(blitz_param_transfer PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(blitz_forward_executor ./forward_executor.cc)
target_include_directories(blitz_forward_executor PUBLIC ${BLITZ_COMMON_INCLUDES})
target_link_libraries(blitz_forward_executor PUBLIC blitz_tccl)
set_target_properties(blitz_forward_executor PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add_subdirectory(tests)