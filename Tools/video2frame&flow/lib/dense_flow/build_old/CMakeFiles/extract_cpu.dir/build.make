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
include CMakeFiles/extract_cpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/extract_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/extract_cpu.dir/flags.make

CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o: CMakeFiles/extract_cpu.dir/flags.make
CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o: ../tools/extract_flow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o -c /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/tools/extract_flow.cpp

CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/tools/extract_flow.cpp > CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.i

CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/tools/extract_flow.cpp -o CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.s

CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o.requires:

.PHONY : CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o.requires

CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o.provides: CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o.requires
	$(MAKE) -f CMakeFiles/extract_cpu.dir/build.make CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o.provides.build
.PHONY : CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o.provides

CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o.provides.build: CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o


# Object files for target extract_cpu
extract_cpu_OBJECTS = \
"CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o"

# External object files for target extract_cpu
extract_cpu_EXTERNAL_OBJECTS =

extract_cpu: CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o
extract_cpu: CMakeFiles/extract_cpu.dir/build.make
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_ts.a
extract_cpu: /usr/lib/x86_64-linux-gnu/libzip.so
extract_cpu: libdenseflow.a
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_videostab.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_ts.a
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_superres.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_stitching.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_contrib.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_nonfree.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_ocl.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_gpu.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_photo.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_objdetect.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_legacy.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_video.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_ml.so.2.4.13
extract_cpu: /usr/local/cuda-8.0/lib64/libcufft.so
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_calib3d.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_features2d.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_highgui.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_imgproc.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_flann.so.2.4.13
extract_cpu: /home/alan/files/temporal-segment-networks-master_png/3rd-party/opencv-2.4.13/build/lib/libopencv_core.so.2.4.13
extract_cpu: /usr/local/cuda-8.0/lib64/libcudart.so
extract_cpu: /usr/local/cuda-8.0/lib64/libnppc.so
extract_cpu: /usr/local/cuda-8.0/lib64/libnppi.so
extract_cpu: /usr/local/cuda-8.0/lib64/libnpps.so
extract_cpu: /usr/lib/x86_64-linux-gnu/libzip.so
extract_cpu: CMakeFiles/extract_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable extract_cpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/extract_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/extract_cpu.dir/build: extract_cpu

.PHONY : CMakeFiles/extract_cpu.dir/build

CMakeFiles/extract_cpu.dir/requires: CMakeFiles/extract_cpu.dir/tools/extract_flow.cpp.o.requires

.PHONY : CMakeFiles/extract_cpu.dir/requires

CMakeFiles/extract_cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/extract_cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/extract_cpu.dir/clean

CMakeFiles/extract_cpu.dir/depend:
	cd /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build /home/alan/files/temporal-segment-networks-master_png/lib/dense_flow/build/CMakeFiles/extract_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/extract_cpu.dir/depend

