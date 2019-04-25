#include "cpu_ops.h"
#include "test_ops.h"
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

vector<float> matinput(size_t size){
    vector<float> res(size);
    for(size_t i = 0; i < size; i++){
        res[i] = float(rand())/float(RAND_MAX) - 0.5f;
    }
    return res;
}
float abs(float x){
    return x < 0 ? -x : x;
}
bool aprox_same(float x1, float x2){
#define abs(x) ((x) < 0 ? -(x) : (x))
    return (abs(x1 * 0.98f) <= abs(x2) &&
                abs(x1) >= abs(x2 * 0.98f)) ||
            (x1 <= x2 + 1e-5f &&
                x1 + 1e-5f >= x2);
#undef abs
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
    cout << "matmul test running:" << endl;
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

void test_transpose(std::function<void(VFloat&,VFloat&)> transpose,int isize, int jsize){
    vector<float> A = matinput(isize*jsize);
    vector<float> res1(isize*jsize);
    vector<float> res2(isize*jsize);
    cout << "transpose test running:" << endl;
    cpu_ops::transpose(A.data(),res1.data(),isize,jsize);
    transpose(A,res2);
    if(!aprox_same(res1.data(),res2.data(),res1.size())){
        cout << "cpu impl:\n";
        print_mat(res1,jsize);
        cout << "test impl:\n";
        print_mat(res2,jsize);
    }
    else{
        cout << "transpose test passed\n";
    }
}
