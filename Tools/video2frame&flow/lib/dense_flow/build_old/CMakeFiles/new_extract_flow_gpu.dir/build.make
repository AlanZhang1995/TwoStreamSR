# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build

# Include any dependencies generated for this target.
include CMakeFiles/new_extract_flow_gpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/new_extract_flow_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/new_extract_flow_gpu.dir/flags.make

CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o: CMakeFiles/new_extract_flow_gpu.dir/flags.make
CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o: ../tools/new_extract_flow_gpu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o -c /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/tools/new_extract_flow_gpu.cpp

CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/tools/new_extract_flow_gpu.cpp > CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.i

CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/tools/new_extract_flow_gpu.cpp -o CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.s

CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o.requires:

.PHONY : CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o.requires

CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o.provides: CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o.requires
	$(MAKE) -f CMakeFiles/new_extract_flow_gpu.dir/build.make CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o.provides.build
.PHONY : CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o.provides

CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o.provides.build: CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o


# Object files for target new_extract_flow_gpu
new_extract_flow_gpu_OBJECTS = \
"CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o"

# External object files for target new_extract_flow_gpu
new_extract_flow_gpu_EXTERNAL_OBJECTS =

new_extract_flow_gpu: CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o
new_extract_flow_gpu: CMakeFiles/new_extract_flow_gpu.dir/build.make
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_ts.a
new_extract_flow_gpu: /usr/lib/x86_64-linux-gnu/libzip.so
new_extract_flow_gpu: libdenseflow.a
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_videostab.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_ts.a
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_superres.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_stitching.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_contrib.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_nonfree.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_ocl.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_gpu.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_photo.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_objdetect.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_legacy.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_video.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_ml.so.2.4.13
new_extract_flow_gpu: /usr/local/cuda-8.0/lib64/libcufft.so
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_calib3d.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_features2d.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_highgui.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_imgproc.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_flann.so.2.4.13
new_extract_flow_gpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_core.so.2.4.13
new_extract_flow_gpu: /usr/local/cuda-8.0/lib64/libcudart.so
new_extract_flow_gpu: /usr/local/cuda-8.0/lib64/libnppc.so
new_extract_flow_gpu: /usr/local/cuda-8.0/lib64/libnppi.so
new_extract_flow_gpu: /usr/local/cuda-8.0/lib64/libnpps.so
new_extract_flow_gpu: /usr/lib/x86_64-linux-gnu/libzip.so
new_extract_flow_gpu: CMakeFiles/new_extract_flow_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable new_extract_flow_gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/new_extract_flow_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/new_extract_flow_gpu.dir/build: new_extract_flow_gpu

.PHONY : CMakeFiles/new_extract_flow_gpu.dir/build

CMakeFiles/new_extract_flow_gpu.dir/requires: CMakeFiles/new_extract_flow_gpu.dir/tools/new_extract_flow_gpu.cpp.o.requires

.PHONY : CMakeFiles/new_extract_flow_gpu.dir/requires

CMakeFiles/new_extract_flow_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/new_extract_flow_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/new_extract_flow_gpu.dir/clean

CMakeFiles/new_extract_flow_gpu.dir/depend:
	cd /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build/CMakeFiles/new_extract_flow_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/new_extract_flow_gpu.dir/depend

