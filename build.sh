OPENCL_DIR="/usr/lib64/libOpenCL.so.1"
#OPENCL_DIR="C:/Windows/System32/OpenCL.dll"

g++ -O3 -march=haswell -o mycl -I "opencl-2.2/"  main.cpp  ocl_executor.cpp test_ops.cpp cpu_ops.cpp profiler.cpp $OPENCL_DIR
