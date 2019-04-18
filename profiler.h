#include <functional>
#include <vector>
using VFloat = std::vector<float>;

VFloat rand_input(size_t size);
double time_func(std::function<void()> gpu_func, int run_times);
