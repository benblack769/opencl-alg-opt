#include <iostream>
#include "ocl_executor.h"
#include "test_ops.h"

using namespace std;

void test_matmul_gpu_impl(){
    int isize = 8;
    int jsize = 16;
    int ksize = 32;
    OpenCLExecutor executor("mat_mul.cl","-D ISIZE=8 -D JSIZE=16 -D KSIZE=32");
    CLBuffer Abuf = executor.new_clbuffer(isize*ksize,sizeof(float));
    CLBuffer Bbuf = executor.new_clbuffer(jsize*ksize,sizeof(float));
    CLBuffer resbuf = executor.new_clbuffer(isize*jsize,sizeof(float));
    CLKernel matmul_kern = executor.new_clkernel(
                "matmul",
                CL_NDRange(isize,jsize),
                CL_NDRange(),
                {Abuf.k_arg(),Bbuf.k_arg(),resbuf.k_arg()});

    auto gpu_func = [&](VFloat & Adata, VFloat & Bdata, VFloat & resdata){
        Abuf.write_buffer(Adata);
        Bbuf.write_buffer(Bdata);
        matmul_kern.run();
        resbuf.read_buffer(resdata);
    };
    test_matmul(gpu_func,isize,jsize,ksize);
}
void test_test_impl(){
    int size = 10;
    OpenCLExecutor executor("test.cl");
    CLBuffer all_quant_buf = executor.new_clbuffer(size,sizeof(int));
    CLKernel update_quant_kern = executor.new_clkernel(
                "set_123",
                CL_NDRange(size),
                CL_NDRange(),
                {all_quant_buf.k_arg()});

    vector<int> quant_cpu_buf(size);

    all_quant_buf.write_buffer(quant_cpu_buf);

    update_quant_kern.run();

    all_quant_buf.read_buffer(quant_cpu_buf);
    cout << quant_cpu_buf[3] << endl;
}
int main(){
    //test_test_impl();
    test_matmul_gpu_impl();
}
