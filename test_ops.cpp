#include "cpu_ops.h"
#include "test_ops.h"
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

vector<float> matinput(size_t size){
    vector<float> res(size);
    for(size_t i = 0; i < size; i++){
        res[i] = rand();
    }
    return res;
}
bool aprox_same(float x1, float x2){
    return x1 * 0.999 < x2 &&
            x1 > x2 * 0.999 &&
            x1 <= x2 + 1e-10 &&
            x1 + 1e-10 >= x2;
}
bool aprox_same(float * d1, float * d2, size_t size){
    for(size_t i = 0; i < size; i++){
        if(!aprox_same(d1[i],d2[i])){
            cout << d1[i] << "\t\t" << d2[i] << "\n\n\n";
            return false;
        }
    }
    return true;
}
void print_mat(vector<float> M, int rowsize){
    int colsize = M.size() / rowsize;
    cout << "\n";
    for(int i = 0; i < colsize; i++){
        for(int j = 0; j < rowsize; j++){
            cout << " " << M[i*rowsize+j];
        }
        cout << "\n";
    }
    cout << "\n";
}

void test_matmul(function<void(VFloat&,VFloat&,VFloat&)> matmul,int isize, int jsize, int ksize){
    vector<float> i1 = matinput(isize*ksize);
    vector<float> i2 = matinput(jsize*ksize);
    vector<float> res1(isize*jsize);
    vector<float> res2(isize*jsize);
    cpu_ops::matmul(i1.data(),i2.data(),res1.data(),isize,jsize,ksize);
    matmul(i1,i2,res2);
    if(!aprox_same(res1.data(),res2.data(),res1.size())){
        cout << "cpu impl:\n";
        print_mat(res1,jsize);
        cout << "test impl:\n";
        print_mat(res2,jsize);
    }
    else{
        cout << "matmul test passed\n";
    }
}
