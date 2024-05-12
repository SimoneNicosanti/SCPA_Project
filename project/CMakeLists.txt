cmake_minimum_required(VERSION 3.0)
project(SCPA_Project)

# Set a custom build directory
set(CUSTOM_BUILD_DIR ${CMAKE_SOURCE_DIR}/out)
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)


## Target for Matrix utility
add_library(Matrix
    code/Matrix/Matrix.h
    code/Matrix/Matrix.c
)
target_include_directories(Matrix PUBLIC code/Matrix)

## Target for Utils utility
add_library(Utils
    code/Utils/PrintUtils.c
    code/Utils/PrintUtils.h
)
target_include_directories(Utils PUBLIC code/Utils)

## Target for MPI utility
add_library(MpiProduct
    code/MPI/MpiProduct.c
    code/MPI/MpiProduct.h 
    code/MPI/SendRecvUtils.c 
    code/MPI/SendRecvUtils.h
)
target_link_libraries(MpiProduct 
    PRIVATE Matrix
)
target_include_directories(MpiProduct PUBLIC code/MPI)

## Target for OpenMP utility
find_package(OpenMP REQUIRED)
add_library(OpenMp
    code/OpenMp/OpenMpProduct.c
    code/OpenMp/OpenMpProduct.h
)
target_link_libraries(OpenMp
    PRIVATE OpenMP::OpenMP_CXX
)
target_compile_options(OpenMp  
    PRIVATE -fopenmp
)
target_include_directories(OpenMp PUBLIC code/OpenMp)

## Target for MPI/OpenMP utility
add_library(OpenMpiProduct
    code/MPI/MpiProduct.c
    code/MPI/MpiProduct.h 
    code/MPI/SendRecvUtils.c 
    code/MPI/SendRecvUtils.h
)
target_link_libraries(OpenMpiProduct 
    PRIVATE OpenMp
)
target_compile_options(OpenMpiProduct 
    PRIVATE -DOPEN_MP
)
target_include_directories(OpenMpiProduct PUBLIC code/MPI)


## Target for Test utility
add_library(Test
    code/Test/ResultWriter.c
    code/Test/ResultWriter.h
)
target_include_directories(Test PUBLIC code/Test)


## MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_executable(MpiTest.out 
    code/Test/MpiTest.c
)
target_link_libraries(MpiTest.out 
    PRIVATE Matrix
    PRIVATE MpiProduct
    PRIVATE Test
    PRIVATE Utils
    PRIVATE ${MPI_CXX_LIBRARIES}
)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_executable(MpiSingleTest.out 
    code/Test/MpiSingleTest.c
)
target_link_libraries(MpiSingleTest.out 
    PRIVATE Matrix
    PRIVATE MpiProduct
    PRIVATE Test
    PRIVATE Utils
    PRIVATE ${MPI_CXX_LIBRARIES}
)

add_executable(OpenMpiSingleTest.out 
    code/Test/MpiSingleTest.c
)
target_link_libraries(OpenMpiSingleTest.out 
    PRIVATE Matrix
    PRIVATE OpenMpiProduct
    PRIVATE Test
    PRIVATE Utils
    PRIVATE ${MPI_CXX_LIBRARIES}
)
target_compile_options(OpenMpiSingleTest.out  
PRIVATE -fopenmp
)

# ## CUDA
# add_executable(cudaExecutable code/CUDA/main.c)
# target_link_libraries(cudaExecutable 
#     PRIVATE Matrix
#     PRIVATE Test
# )

# Specify the output directory for the executable
set_target_properties(MpiTest.out  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUSTOM_BUILD_DIR}
)

set_target_properties(MpiSingleTest.out  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUSTOM_BUILD_DIR}
)

set_target_properties(OpenMpiSingleTest.out  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUSTOM_BUILD_DIR}
)