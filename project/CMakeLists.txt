cmake_minimum_required(VERSION 3.0)
project(SCPA_Project)

# Set a custom build directory
set(CUSTOM_BUILD_DIR ../build)


## Target for Matrix utility
add_library(Matrix
    code/Matrix/Matrix.h
    code/Matrix/Matrix.c
)
target_include_directories(Matrix PUBLIC code/Matrix)

## Target for Test utility
add_library(Test
    code/Test/Test.h
    code/Test/Test.c
    code/Test/TestResultWriter.c
)
target_include_directories(Test PUBLIC code/Test)


## MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(SOURCE_FILES 
    code/MPI/main.c
    code/MPI/MpiProduct.c
    code/MPI/MpiProduct.h
)
add_executable(mpiExecutable.out 
    ${SOURCE_FILES}
)
target_link_libraries(mpiExecutable.out 
    PRIVATE Matrix
    PRIVATE Test
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
set_target_properties(mpiExecutable.out PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUSTOM_BUILD_DIR}
)