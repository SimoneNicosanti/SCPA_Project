# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/simone/Scrivania/University/SCPA/SCPA_Project/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/simone/Scrivania/University/SCPA/SCPA_Project/project/CMake

# Include any dependencies generated for this target.
include CMakeFiles/mpiExecutable.out.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mpiExecutable.out.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mpiExecutable.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpiExecutable.out.dir/flags.make

CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o: CMakeFiles/mpiExecutable.out.dir/flags.make
CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o: ../code/MPI/main.c
CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o: CMakeFiles/mpiExecutable.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simone/Scrivania/University/SCPA/SCPA_Project/project/CMake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o -MF CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o.d -o CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o -c /home/simone/Scrivania/University/SCPA/SCPA_Project/project/code/MPI/main.c

CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/simone/Scrivania/University/SCPA/SCPA_Project/project/code/MPI/main.c > CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.i

CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/simone/Scrivania/University/SCPA/SCPA_Project/project/code/MPI/main.c -o CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.s

CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o: CMakeFiles/mpiExecutable.out.dir/flags.make
CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o: ../code/MPI/mpi.c
CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o: CMakeFiles/mpiExecutable.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/simone/Scrivania/University/SCPA/SCPA_Project/project/CMake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o -MF CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o.d -o CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o -c /home/simone/Scrivania/University/SCPA/SCPA_Project/project/code/MPI/mpi.c

CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/simone/Scrivania/University/SCPA/SCPA_Project/project/code/MPI/mpi.c > CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.i

CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/simone/Scrivania/University/SCPA/SCPA_Project/project/code/MPI/mpi.c -o CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.s

# Object files for target mpiExecutable.out
mpiExecutable_out_OBJECTS = \
"CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o" \
"CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o"

# External object files for target mpiExecutable.out
mpiExecutable_out_EXTERNAL_OBJECTS =

../build/mpiExecutable.out: CMakeFiles/mpiExecutable.out.dir/code/MPI/main.c.o
../build/mpiExecutable.out: CMakeFiles/mpiExecutable.out.dir/code/MPI/mpi.c.o
../build/mpiExecutable.out: CMakeFiles/mpiExecutable.out.dir/build.make
../build/mpiExecutable.out: libMatrix.a
../build/mpiExecutable.out: libTest.a
../build/mpiExecutable.out: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
../build/mpiExecutable.out: /usr/lib/x86_64-linux-gnu/libmpich.so
../build/mpiExecutable.out: CMakeFiles/mpiExecutable.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/simone/Scrivania/University/SCPA/SCPA_Project/project/CMake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable ../build/mpiExecutable.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpiExecutable.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpiExecutable.out.dir/build: ../build/mpiExecutable.out
.PHONY : CMakeFiles/mpiExecutable.out.dir/build

CMakeFiles/mpiExecutable.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpiExecutable.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpiExecutable.out.dir/clean

CMakeFiles/mpiExecutable.out.dir/depend:
	cd /home/simone/Scrivania/University/SCPA/SCPA_Project/project/CMake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/simone/Scrivania/University/SCPA/SCPA_Project/project /home/simone/Scrivania/University/SCPA/SCPA_Project/project /home/simone/Scrivania/University/SCPA/SCPA_Project/project/CMake /home/simone/Scrivania/University/SCPA/SCPA_Project/project/CMake /home/simone/Scrivania/University/SCPA/SCPA_Project/project/CMake/CMakeFiles/mpiExecutable.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpiExecutable.out.dir/depend

