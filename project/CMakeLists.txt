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

## Targer for Sequential utility
add_library(Sequential
    code/Sequential/Sequential.h
    code/Sequential/Sequential.c
)
target_include_directories(Sequential PUBLIC code/Sequential)

## Target for MPI utility
add_library(MpiProduct
    code/MPI/MpiProduct.c
    code/MPI/MpiProduct.h 
    code/MPI/SendRecvUtils.c 
    code/MPI/SendRecvUtils.h
)
target_include_directories(MpiProduct PUBLIC code/MPI)

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
    PRIVATE Sequential
    PRIVATE Test
    PRIVATE Utils
    PRIVATE ${MPI_CXX_LIBRARIES}
)

# ## OpenMp
# add_executable(openMpExecutable code/OpenMP/main.c)
# target_link_libraries(openMpExecutable 
#     PRIVATE Matrix
#     PRIVATE Test
# )

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