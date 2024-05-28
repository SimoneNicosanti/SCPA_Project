module load mpi
module load cuda

# Removing previous output files (including error files)
rm ./out/*.out*

# Changing to build directory in order to build from there (generated files will be put there)
cd build

# Compiling in the previous directory (that is the main directory of the project)
cmake ..

# Building using the cache in build directory
cmake --build .