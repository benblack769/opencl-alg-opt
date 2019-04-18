cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  #set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(opencl-2.2)


add_executable(cl_prac
    main.cpp
    test_ops.cpp
    cpu_ops.cpp
       ocl_executor.cpp
)
target_link_libraries(cl_prac PRIVATE
    "C:/Windows/System32/OpenCL.dll"
)
