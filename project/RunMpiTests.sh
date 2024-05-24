# Loading MPI module
module load mpi

# Creating result directory structure
mkdir ./Results
mkdir ./Results/MPI
mkdir ./Results/MPI/Tests

# Clearing previous tests
rm ./Results/MPI/Tests/*

# Compiling
./CMakeCaller.sh

# Changing directory to out
cd ./out

# MPI Tests
mpiexec -np 4 ./MpiTest.out
mpiexec -np 8 ./MpiTest.out
mpiexec -np 12 ./MpiTest.out
mpiexec -np 16 ./MpiTest.out
mpiexec -np 20 ./MpiTest.out