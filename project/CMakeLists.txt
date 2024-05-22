cmake_minimum_required(VERSION 3.0)
project(SCPA_Project)

# Set a custom build directory
set(CUSTOM_BUILD_DIR ${CMAKE_SOURCE_DIR}/out)

# Set compilation flags
add_compile_options(-O3) # Optimization Flag
add_compile_options(-march=native) # Optimization Flag
# add_compile_options(-ffast-math) # Optimization Flag
# add_compile_options(-funroll-loops) # Optimization Flag

# add_compile_options(-Wall)
# add_compile_options(-Wextra)

# Find MPI and OpenMP Packages
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


## Target for Matrix utility
file(GLOB Matrix_Src code/Matrix/*)
add_library(Matrix
    ${Matrix_Src}
)
target_include_directories(Matrix 
    PUBLIC code/Matrix
)

## Target for Utils utility
file(GLOB Utils_Src code/Utils/*)
add_library(Utils
    ${Utils_Src}
)
target_include_directories(Utils 
    PUBLIC code/Utils
)


## Target for MPI utility
file(GLOB MpiProduct_Src code/MPI/*)
add_library(MpiProduct
    ${MpiProduct_Src}
)
target_include_directories(MpiProduct 
    PUBLIC code/MPI
)
target_link_libraries(MpiProduct 
    PRIVATE Matrix
    PRIVATE Utils
    PRIVATE ${MPI_C_LIBRARIES}
)


## Target for Test utility
file(GLOB Test_Src code/Test/ResultWriter.*)
add_library(Test
    ${Test_Src}
)
target_include_directories(Test 
    PUBLIC code/Test
)


## MPI executables
add_executable(MpiTest.out 
    code/Test/MpiTest.c
)
target_link_libraries(MpiTest.out 
    PRIVATE Matrix
    PRIVATE MpiProduct
    PRIVATE Test
    PRIVATE Utils
    PRIVATE ${MPI_C_LIBRARIES}
)

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