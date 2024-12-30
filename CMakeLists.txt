cmake_minimum_required(VERSION 3.18)
project(VideoBlur LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find OpenCV (works via vcpkg toolchain or other methods)
find_package(OpenCV REQUIRED)

# Compile the CUDA source into a static library
add_library(GpuBlur STATIC
        src/GpuBlur.cu
        include/GpuBlur.hpp
)
set_target_properties(GpuBlur PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Build main executable
add_executable(video_blur
        src/main.cpp
        src/VideoProcessor.cpp
)

# Link against OpenCV and our CUDA library
target_link_libraries(video_blur
        GpuBlur
        ${OpenCV_LIBS}
)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Include OpenCV headers
target_include_directories(video_blur PRIVATE ${OpenCV_INCLUDE_DIRS} include)
target_include_directories(GpuBlur PRIVATE include)