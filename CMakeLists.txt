cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(BlitzTransformer LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH "/root/.local")
else()
  set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Threads)

# add_subdirectory(third_party/googletest)
add_subdirectory(third_party/json)
add_subdirectory(third_party/argparse)
set(CMAKE_PIC_DEFAULT ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(third_party/fmt)
set(CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_PIC_DEFAULT})
link_libraries(fmt::fmt)

include_directories(${CMAKE_SOURCE_DIR}/include)

if(DEFINED ENV{CONDA_PREFIX})
  # use conda environment
  link_directories($ENV{CONDA_PREFIX}/lib)
  include_directories($ENV{CONDA_PREFIX}/include)

  # set atop
  # list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

if(DEFINED ENV{CUDA_HOME})
  set(CUDAToolkit_ROOT $ENV{CUDA_HOME})
else()
  set(CUDAToolkit_ROOT /usr/local/cuda)
endif()

find_package(CUDAToolkit 12.1 REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 80)

# gcc >= 8 is required, we do not support other compilers
if((NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0))
  message(FATAL_ERROR "GCC 8.0 or higher is required")
endif()

# Add filesystem library for gcc < 9
link_libraries("$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>")

# Set up C++ standard
set(CXX_STD "20" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Switch between release mode and debug mode
# The user can use `-DBUILD_MODE=DEBUG` or `-DBUILD_MODE=RELEASE` to
# choose the build mode.
# If no option is provided, default to debug mode
if(BUILD_MODE)
  string(TOUPPER ${BUILD_MODE} BUILD_MODE)

  if(BUILD_MODE STREQUAL "DEBUG")
    set(DEBUG ON)
    set(CUDA_DEBUG OFF)
  elseif(BUILD_MODE STREQUAL "TRACE")
    set(DEBUG ON)
    set(CUDA_DEBUG ON)
  elseif(BUILD_MODE STREQUAL "RELWITHDEBUG")
    set(RELWITHDEBUG ON)
    set(CUDA_DEBUG OFF)
  elseif(BUILD_MODE STREQUAL "RELEASE")
    set(RELEASE ON)
  else()
    message(FATAL_ERROR "Unknown build mode: ${BUILD_MODE}")
  endif()
else()
  message("No build type selected, defaulting to RELEASE mode")
  message("Use -DBUILD_MODE=DEBUG or -DBUILD_MODE=RELEASE to specify build type")
  set(RELEASE ON)
endif()

# TODO
# Set up C++ flag and CUDA flag
if(RELEASE)
  message("Building in release mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --prec-div=false")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -DRELEASE")
elseif(RELWITHDEBUG)
  message("Building in release with debug info mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --prec-div=false")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function")
elseif(DEBUG AND(NOT CUDA_DEBUG))
  message("Building in Host debug mode, disabling CUDA force synchronize")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --prec-div=false")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wno-unused-parameter -Wno-unused-function")
else()
  message("Building in FULL debug mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -DDEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -DDEBUG")
endif()

# TODO what for?
# Set up COMMON_HEADER_DIRS and COMMON_LIB_DIRS
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
)
set(COMMON_LIB_DIRS "")

list(APPEND COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR}/third_party/fmt/include)

# Set up MPI and NCCL for multi-GPU communication
message("Building with MPI and NCCL")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(MKL_MPI "openmpi")
find_package(NCCL REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building

# Add MPI and NCCL into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
list(APPEND COMMON_HEADER_DIRS ${MPI_INCLUDE_PATH} ${NCCL_INCLUDE_DIR})
list(APPEND COMMON_LIB_DIRS ${MPI_LIBRARIES} ${NCCL_LIBRARIES})

set(COMMON_LIBS CUDA::cudart)

# Add Python into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "import sysconfig;
print(sysconfig.get_paths()['include']);"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PY_INCLUDE_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Python config Error.")
endif()

list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

if(NOT DEFINED Torch_DIR)
# Add LibTorch into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
execute_process(COMMAND ${PYTHON_PATH} "-c" "import os; import torch;
print(os.path.dirname(torch.__file__), end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_DIR
  ERROR_QUIET
)
message("${TORCH_DIR}")

# string(REGEX MATCH "/usr.*/torch" TORCH_DIR "${TORCH_DIR}")
if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Torch config Error.")
endif()

message("Torch Directory: ${TORCH_DIR}")
else()
message("Torch Directory: ${Torch_DIR}")
set(TORCH_DIR ${Torch_DIR})
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
set(CAFFE2_USE_CUDNN 1)
find_package(Torch REQUIRED)
list(APPEND COMMON_HEADER_DIRS "${TORCH_INCLUDE_DIRS}")
list(APPEND COMMON_LIBS "${TORCH_LIBRARIES}")

# Let COMMON_HEADER_DIRS & COMMON_LIB_DIRS take effect
include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})
link_libraries(${COMMON_LIBS})

# Should turn off CXX11 ABI if pytorch is built with CXX11 ABI off
execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE USE_CXX11_ABI)
message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")

if(USE_CXX11_ABI)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# Let all executable targets go to bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_BINARY_DIR}/proto)

set(BUILD_SHARED_DEFAULT ${BUILD_SHARED_LIBS})
set(CMAKE_PIC_DEFAULT ${CMAKE_POSITION_INDEPENDENT_CODE})
set(BUILD_SHARED_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_BENCHMARKS OFF)
set(USE_CUDA OFF)
add_subdirectory(third_party/pickle)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_DEFAULT})
set(CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_PIC_DEFAULT})

# Add subdirectories
add_subdirectory(tokenizer)
add_subdirectory(proto)
add_subdirectory(src)
