#include <functional>
#include <vector>
using VFloat = std::vector<float>;

void test_matmul(std::function<void(VFloat&,VFloat&,VFloat&)> matmul,int isize, int jsize, int ksize);
void test_transpose(std::function<void(VFloat&,VFloat&)> transpose,int isize, int jsize);
