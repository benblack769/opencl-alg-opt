#include "cpu_ops.h"
/*
    res = A * B
    A : (isize,ksize)
    B : (ksize,jsize)
    res: (isize,jsize)
*/

namespace cpu_ops {

void matmul(float * A, float * B, float * res, int isize, int jsize, int ksize){
    for(int i = 0; i < isize; i++){
        for(int j = 0; j < jsize; j++){
            float sum = 0;
            for(int k = 0; k < ksize; k++){
                 sum += A[i*ksize+k] * B[k*jsize+j];
            }
            res[i*jsize+j] = sum;
        }
    }
}

}
