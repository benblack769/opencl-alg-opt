#include <iostream>
#include <utility>
#include "ocl_executor.h"
#include "test_ops.h"
#include "profiler.h"

using namespace std;

std::string format_defs(vector<string> unnamed_defs, vector<pair<string, string>> named_defs){
    string defstr = "";
    for(string s : unnamed_defs){
        defstr += " -D " + s;
        defstr += s;
    }
    for(pair<string, string> p : named_defs){
        defstr += " -D " + p.first + "=" + p.second;
    }
    return defstr;
}

void test_matmul_gpu_impl(){
    int isize = 512;
    int jsize = 512;
    int ksize = 512;
    string format_str = format_defs({},{
        make_pair("ISIZE",to_string(isize)),
        make_pair("JSIZE",to_string(jsize)),
        make_pair("KSIZE",to_string(ksize)),
    });
    cout << format_str;
    OpenCLExecutor executor("mat_mul.cl",format_str);
    CLBuffer Abuf = executor.new_clbuffer(isize*ksize,sizeof(float));
    CLBuffer Bbuf = executor.new_clbuffer(jsize*ksize,sizeof(float));
    CLBuffer resbuf = executor.new_clbuffer(isize*jsize,sizeof(float));
    CLKernel matmul_kern = executor.new_clkernel(
                "matmul",
                CL_NDRange(isize,jsize),
                CL_NDRange(),
                {Abuf.k_arg(),Bbuf.k_arg(),resbuf.k_arg()});

    CLKernel set_kern = executor.new_clkernel(
                "setarange",
                CL_NDRange(isize*ksize),
                CL_NDRange(),
                {Abuf.k_arg()});

    auto gpu_func = [&](VFloat & Adata, VFloat & Bdata, VFloat & resdata){
        Abuf.write_buffer(Adata);
        Bbuf.write_buffer(Bdata);
        matmul_kern.run();
        resbuf.read_buffer(resdata);
    };
    test_matmul(gpu_func,isize,jsize,ksize);
    int x = 0;
    auto mat_run_func = [&](){
        set_kern.run();
        matmul_kern.run();
        //x++;
        //if(x % 5 == 0){
            executor.wait_until_exec();
        //}
    };
    double time = time_func(mat_run_func,1000);
    cout << "average time: " << time << "\n";
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
