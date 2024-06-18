# Loading Cuda module
module load cuda

# Creating result directory structure
mkdir ./Results
mkdir ./Results/CUDA
mkdir ./Results/CUDA/Tests

# Clearing previous tests
rm ./Results/CUDA/Tests/*

# Compiling
./CMakeCaller.sh

# Changing directory to out
cd ./out

# Cuda Tests
# ./CudaTest.out -v 0
#./CudaTest.out -v 1
./CudaTest.out -v 2
# ./CudaTest.out -v 3
# ./CudaTest.out -v 4