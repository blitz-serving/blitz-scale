# define self_impl tools flashinfer_kernel
#todo target include directory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/flashinfer)

file(GLOB flashinfer_self_impl ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
add_library(flashinfer_kernel STATIC ${flashinfer_self_impl})
set_property(TARGET flashinfer_kernel PROPERTY LINKER_LANGUAGE CUDA)
set_property(TARGET flashinfer_kernel PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(flashinfer_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/flashinfer/include/)
target_include_directories(flashinfer_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/flashinfer/src/)
target_link_libraries(flashinfer_kernel PRIVATE prefill_kernels decode_kernels)

