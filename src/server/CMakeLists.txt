set(BLITZ_DIR ${CMAKE_SOURCE_DIR}/src/blitz/include)
set(ARGPARSE_DIR ${CMAKE_SOURCE_DIR}/third_party/argparse)
set(JSON_DIR ${CMAKE_SOURCE_DIR}/third_party/json)
set(FMT_DIR ${CMAKE_SOURCE_DIR}/third_party/fmt)
set(PROTO_DIR ${CMAKE_BINARY_DIR}/proto)

set(INCLUDES ${BLITZ_DIR}
    ${ARGPARSE_DIR}/include
    ${PROTO_DIR}
    ${JSON_DIR}/include
    ${FMT_DIR}/include
    ${CMAKE_SOURCE_DIR}/include)

add_executable(run_server_disaggregative disaggregate.cc)
add_dependencies(run_server_disaggregative generatev2)
target_link_libraries(run_server_disaggregative
    service
    argparse
    flashinfer_kernel
    tokenizer
    ${MPI_LIBRARIES}
    nccl_utils
    blitz_tccl
    CUDA::cublas)
target_include_directories(run_server_disaggregative PUBLIC ${INCLUDES})