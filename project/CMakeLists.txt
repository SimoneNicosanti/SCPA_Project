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
find_package(OpenMP REQUIRED)
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


## Target for OpenMP utility
file(GLOB OpenMp_Src code/OpenMp/*)
add_library(OpenMp
    ${OpenMp_Src}
)
target_link_libraries(OpenMp
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC Matrix
)
target_compile_options(OpenMp  
    PRIVATE -fopenmp
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


## Target for MPI/OpenMP utility
add_library(OpenMpiProduct
    ${MpiProduct_Src}
)
target_include_directories(OpenMpiProduct 
    PUBLIC code/MPI
    PUBLIC code/OpenMp
)
target_link_libraries(OpenMpiProduct 
    PUBLIC OpenMp
    PUBLIC Matrix
    PUBLIC Utils
    PRIVATE ${MPI_C_LIBRARIES}
)
target_compile_options(OpenMpiProduct 
    PRIVATE -DOPEN_MP
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

# MPI/OpenMP Executables
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

# MPI/OpenMp Tests
add_executable(OpenMpiTest.out 
    code/Test/MpiSingleTest.c
)
target_link_libraries(OpenMpiTest.out 
    PRIVATE Matrix
    PRIVATE OpenMpiProduct
    PRIVATE Test
    PRIVATE Utils
    PRIVATE ${MPI_CXX_LIBRARIES}
)
target_compile_options(OpenMpiTest.out  
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

set_target_properties(OpenMpiTest.out   PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUSTOM_BUILD_DIR}
)