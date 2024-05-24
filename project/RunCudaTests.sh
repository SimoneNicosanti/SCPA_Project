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
# TODO Add CUDA Tests here