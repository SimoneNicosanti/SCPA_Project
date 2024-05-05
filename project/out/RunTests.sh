
if [ $1 = 1 ]; then
# MPI Tests
    mpiexec -np 4 ./MpiTest.out
    mpiexec -np 8 ./MpiTest.out
    mpiexec -np 12 ./MpiTest.out
    mpiexec -np 16 ./MpiTest.out
    mpiexec -np 20 ./MpiTest.out
elif ['$1' -eq "2"]
then
    echo "CUDA TESTS"
else
    echo "INVALID PARAM"
fi