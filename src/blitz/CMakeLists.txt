include_directories(${CMAKE_SOURCE_DIR}/src/util)

add_library(blitz_tccl STATIC ./blitz_tccl.cpp)
target_link_libraries(blitz_tccl PUBLIC pickle MPI::MPI_CXX numa)
target_include_directories(blitz_tccl PUBLIC ${CMAKE_SOURCE_DIR}/src/include)
set_target_properties(blitz_tccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(blitz_model_loader ./model_loader.cc)
set_target_properties(blitz_model_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add_subdirectory(tests)