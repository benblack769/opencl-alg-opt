#include <chrono>
#include "profiler.h"

using namespace std;

VFloat rand_input(size_t size){
    vector<float> res(size);
    for(size_t i = 0; i < size; i++){
        res[i] = rand();
    }
    return res;
}
double time_func(std::function<void()> gpu_func, int run_times){
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<double> fsec;

    //warm up function
    gpu_func();

    //time function
    auto t0 = Time::now();
    for(size_t i = 0; i < run_times; i++){
        gpu_func();
    }
    auto t1 = Time::now();

    fsec fs = t1 - t0;
    return fs.count() / run_times;
}
