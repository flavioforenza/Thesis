# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/flavio/jetson-inference

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/flavio/jetson-inference/build

# Include any dependencies generated for this target.
include python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/depend.make

# Include the progress variables for this target.
include python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/progress.make

# Include the compile flags for this target's objects.
include python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/flags.make

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/flags.make
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o: ../python/bindings/PyDepthNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/flavio/jetson-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o -c /home/flavio/jetson-inference/python/bindings/PyDepthNet.cpp

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.i"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/flavio/jetson-inference/python/bindings/PyDepthNet.cpp > CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.i

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.s"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/flavio/jetson-inference/python/bindings/PyDepthNet.cpp -o CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.s

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o.requires:

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o.requires

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o.provides: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o.requires
	$(MAKE) -f python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build.make python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o.provides.build
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o.provides

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o.provides.build: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o


python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/flags.make
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o: ../python/bindings/PyDetectNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/flavio/jetson-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o -c /home/flavio/jetson-inference/python/bindings/PyDetectNet.cpp

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.i"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/flavio/jetson-inference/python/bindings/PyDetectNet.cpp > CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.i

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.s"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/flavio/jetson-inference/python/bindings/PyDetectNet.cpp -o CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.s

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o.requires:

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o.requires

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o.provides: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o.requires
	$(MAKE) -f python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build.make python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o.provides.build
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o.provides

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o.provides.build: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o


python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/flags.make
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o: ../python/bindings/PyImageNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/flavio/jetson-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o -c /home/flavio/jetson-inference/python/bindings/PyImageNet.cpp

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.i"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/flavio/jetson-inference/python/bindings/PyImageNet.cpp > CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.i

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.s"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/flavio/jetson-inference/python/bindings/PyImageNet.cpp -o CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.s

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o.requires:

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o.requires

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o.provides: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o.requires
	$(MAKE) -f python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build.make python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o.provides.build
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o.provides

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o.provides.build: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o


python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/flags.make
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o: ../python/bindings/PyInference.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/flavio/jetson-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o -c /home/flavio/jetson-inference/python/bindings/PyInference.cpp

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.i"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/flavio/jetson-inference/python/bindings/PyInference.cpp > CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.i

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.s"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/flavio/jetson-inference/python/bindings/PyInference.cpp -o CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.s

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o.requires:

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o.requires

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o.provides: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o.requires
	$(MAKE) -f python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build.make python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o.provides.build
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o.provides

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o.provides.build: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o


python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/flags.make
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o: ../python/bindings/PyPoseNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/flavio/jetson-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o -c /home/flavio/jetson-inference/python/bindings/PyPoseNet.cpp

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.i"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/flavio/jetson-inference/python/bindings/PyPoseNet.cpp > CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.i

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.s"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/flavio/jetson-inference/python/bindings/PyPoseNet.cpp -o CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.s

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o.requires:

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o.requires

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o.provides: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o.requires
	$(MAKE) -f python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build.make python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o.provides.build
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o.provides

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o.provides.build: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o


python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/flags.make
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o: ../python/bindings/PySegNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/flavio/jetson-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o -c /home/flavio/jetson-inference/python/bindings/PySegNet.cpp

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.i"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/flavio/jetson-inference/python/bindings/PySegNet.cpp > CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.i

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.s"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/flavio/jetson-inference/python/bindings/PySegNet.cpp -o CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.s

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o.requires:

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o.requires

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o.provides: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o.requires
	$(MAKE) -f python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build.make python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o.provides.build
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o.provides

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o.provides.build: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o


python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/flags.make
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o: ../python/bindings/PyTensorNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/flavio/jetson-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o -c /home/flavio/jetson-inference/python/bindings/PyTensorNet.cpp

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.i"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/flavio/jetson-inference/python/bindings/PyTensorNet.cpp > CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.i

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.s"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/flavio/jetson-inference/python/bindings/PyTensorNet.cpp -o CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.s

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o.requires:

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o.requires

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o.provides: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o.requires
	$(MAKE) -f python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build.make python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o.provides.build
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o.provides

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o.provides.build: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o


# Object files for target jetson-inference-python-27
jetson__inference__python__27_OBJECTS = \
"CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o" \
"CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o" \
"CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o" \
"CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o" \
"CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o" \
"CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o" \
"CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o"

# External object files for target jetson-inference-python-27
jetson__inference__python__27_EXTERNAL_OBJECTS =

aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o
aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o
aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o
aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o
aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o
aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o
aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o
aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build.make
aarch64/lib/python/2.7/jetson_inference_python.so: /usr/local/cuda/lib64/libcudart_static.a
aarch64/lib/python/2.7/jetson_inference_python.so: /usr/lib/aarch64-linux-gnu/librt.so
aarch64/lib/python/2.7/jetson_inference_python.so: aarch64/lib/libjetson-inference.so
aarch64/lib/python/2.7/jetson_inference_python.so: /usr/lib/aarch64-linux-gnu/libpython2.7.so
aarch64/lib/python/2.7/jetson_inference_python.so: aarch64/lib/python/2.7/jetson_utils_python.so
aarch64/lib/python/2.7/jetson_inference_python.so: /usr/lib/aarch64-linux-gnu/libpython2.7.so
aarch64/lib/python/2.7/jetson_inference_python.so: aarch64/lib/libjetson-utils.so
aarch64/lib/python/2.7/jetson_inference_python.so: /usr/local/cuda/lib64/libcudart_static.a
aarch64/lib/python/2.7/jetson_inference_python.so: /usr/lib/aarch64-linux-gnu/librt.so
aarch64/lib/python/2.7/jetson_inference_python.so: /usr/local/cuda/lib64/libnppicc.so
aarch64/lib/python/2.7/jetson_inference_python.so: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/flavio/jetson-inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared library ../../aarch64/lib/python/2.7/jetson_inference_python.so"
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jetson-inference-python-27.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build: aarch64/lib/python/2.7/jetson_inference_python.so

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/build

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/requires: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDepthNet.cpp.o.requires
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/requires: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyDetectNet.cpp.o.requires
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/requires: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyImageNet.cpp.o.requires
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/requires: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyInference.cpp.o.requires
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/requires: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyPoseNet.cpp.o.requires
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/requires: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PySegNet.cpp.o.requires
python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/requires: python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/PyTensorNet.cpp.o.requires

.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/requires

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/clean:
	cd /home/flavio/jetson-inference/build/python/bindings_python_2.7 && $(CMAKE_COMMAND) -P CMakeFiles/jetson-inference-python-27.dir/cmake_clean.cmake
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/clean

python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/depend:
	cd /home/flavio/jetson-inference/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/flavio/jetson-inference /home/flavio/jetson-inference/python/bindings /home/flavio/jetson-inference/build /home/flavio/jetson-inference/build/python/bindings_python_2.7 /home/flavio/jetson-inference/build/python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : python/bindings_python_2.7/CMakeFiles/jetson-inference-python-27.dir/depend

