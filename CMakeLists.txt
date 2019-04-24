cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  #set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(opencl-2.2)


add_executable(cl_prac
    main.cpp
    test_ops.cpp
    cpu_ops.cpp
    profiler.cpp
       ocl_executor.cpp
)
target_link_libraries(cl_prac PRIVATE
    "C:/Windows/System32/OpenCL.dll"
)
