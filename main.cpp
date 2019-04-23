#include <iostream>
#include <utility>
#include "ocl_executor.h"
#include "test_ops.h"
#include "profiler.h"
#include "cpu_ops.h"

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
void test_transpose_gpu_impl(){
    int isize = 512;
    int jsize = 1024;
    int asize = 4;
    int bsize = 8;
    string format_str = format_defs({},{
        make_pair("ISIZE",to_string(isize)),
        make_pair("JSIZE",to_string(jsize)),
        make_pair("ASIZE",to_string(asize)),
        make_pair("BSIZE",to_string(bsize)),
    });
    cout << format_str;
    OpenCLExecutor executor("transpose.cl",format_str);
    CLBuffer Abuf = executor.new_clbuffer(isize*jsize,sizeof(float));
    CLBuffer resbuf = executor.new_clbuffer(isize*jsize,sizeof(float));
    CLKernel tranpose_kern = executor.new_clkernel(
                "transpose",
                CL_NDRange(isize,jsize),
                CL_NDRange(),//asize,bsize),
                {Abuf.k_arg(),resbuf.k_arg()});

    auto gpu_func = [&](VFloat & Adata, VFloat & resdata){
        Abuf.write_buffer(Adata);
        tranpose_kern.run();
        resbuf.read_buffer(resdata);
    };
    test_transpose(gpu_func,isize,jsize);
    auto transpose_run_func = [&](){
        tranpose_kern.run();
        executor.wait_until_exec();
    };
    double time = time_func(transpose_run_func,1000);
    cout << "average time: " << time << "\n";
}

void test_matmul_gpu_impl(){
    int isize = 32;
    int jsize = 32;
    int ksize = 32;
    int igroup = 4;
    int jgroup = 4;
    int kgroup = 8;
    string format_str = format_defs({},{
        make_pair("ISIZE",to_string(isize)),
        make_pair("JSIZE",to_string(jsize)),
        make_pair("KSIZE",to_string(ksize)),
        make_pair("IGROUP",to_string(igroup)),
        make_pair("JGROUP",to_string(jgroup)),
        make_pair("KGROUP",to_string(kgroup)),
    });
    cout << format_str;
    OpenCLExecutor executor("mat_mul.cl",format_str);
    CLBuffer Abuf = executor.new_clbuffer(isize*ksize,sizeof(float));
    CLBuffer Bbuf = executor.new_clbuffer(jsize*ksize,sizeof(float));
    CLBuffer resbuf = executor.new_clbuffer(isize*jsize,sizeof(float));
    CLKernel matmul_kern = executor.new_clkernel(
                "matmul",
                CL_NDRange(isize,jsize),
                CL_NDRange(igroup,jgroup),
                {Abuf.k_arg(),Bbuf.k_arg(),resbuf.k_arg()});

    auto gpu_func = [&](VFloat & Adata, VFloat & Bdata, VFloat & resdata){
        Abuf.write_buffer(Adata);
        Bbuf.write_buffer(Bdata);
        matmul_kern.run();
        resbuf.read_buffer(resdata);
    };
    test_matmul(gpu_func,isize,jsize,ksize);
    auto mat_run_func = [&](){
        matmul_kern.run();
        executor.wait_until_exec();
    };
    double time = time_func(mat_run_func,1000);
    cout << "average time: " << time << "\n";
}
void test_cpu_cubed(){
    int isize = 512;
    int jsize = 256;
    int ksize = 128;

    auto cubed_matmul = [&](VFloat & Adata, VFloat & Bdata, VFloat & resdata){
        cpu_ops::matmulcubed(Adata.data(),Bdata.data(),resdata.data(),isize,jsize,ksize);
    };
    test_matmul(cubed_matmul,isize,jsize,ksize);

    VFloat A = rand_input(isize*ksize);
    VFloat B = rand_input(jsize*ksize);
    VFloat res = rand_input(isize*jsize);
    auto mat_run_func = [&](){
        cpu_ops::matmulcubed(A.data(),B.data(),res.data(),isize,jsize,ksize);
    };
    double time = time_func(mat_run_func,10);
    cout << "average cubed time: " << time << "\n";
    auto mat_run_func2 = [&](){
        cpu_ops::matmul(A.data(),B.data(),res.data(),isize,jsize,ksize);
    };
    double time2 = time_func(mat_run_func2,10);
    cout << "average time: " << time2 << "\n";
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
    //test_cpu_cubed();
    //test_transpose_gpu_impl();
}
