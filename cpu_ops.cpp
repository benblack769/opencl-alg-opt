#include "cpu_ops.h"

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
void matmulcubed(float * A, float * B, float * res, int isize, int jsize, int ksize){
    constexpr int igroup = 4;
    constexpr int jgroup = 8;
    constexpr int kgroup = 16;

    for(int ib = 0; ib < isize; ib += igroup){
        for(int jb = 0; jb < jsize; jb += jgroup){

            float block_sum[igroup][jgroup] = {0};

            for(int kb = 0; kb < ksize; kb += kgroup){

                float A_block[igroup][kgroup];
                float B_block[jgroup][kgroup];

                for(int i = 0; i < igroup; i++){
                    for(int k = 0; k < kgroup; k++){
                        A_block[i][k] = A[(ib+i)*ksize + (kb + k)];
                    }
                }
                for(int k = 0; k < kgroup; k++){
                    for(int j = 0; j < jgroup; j++){
                        B_block[j][k] = B[(kb+k)*jsize + (jb + j)];
                    }
                }

                for(int i = 0; i < igroup; i++){
                    for(int j = 0; j < jgroup; j++){
                        float sum = 0;
                        for(int k = 0; k < kgroup; k++){
                            sum += A_block[i][k] * B_block[j][k];
                        }
                        block_sum[i][j] += sum;
                    }
                }
            }

            for(int i = 0; i < igroup; i++){
                for(int j = 0; j < jgroup; j++){
                    res[(ib+i)*jsize + (jb+j)] = block_sum[i][j];
                }
            }

        }
    }
}


void transpose(float * A, float * res, int isize, int jsize){
    for(int i = 0; i < isize; i++){
        for(int j = 0; j < jsize; j++){
            res[j*isize+i] = A[i*jsize+j];
        }
    }
}

}
