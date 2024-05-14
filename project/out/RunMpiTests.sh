./CMakeCaller.sh
rm ../Results/*/*/*

# MPI Tests
mpiexec -np 4 ./MpiTest.out
mpiexec -np 8 ./MpiTest.out
mpiexec -np 12 ./MpiTest.out
mpiexec -np 16 ./MpiTest.out
mpiexec -np 20 ./MpiTest.out