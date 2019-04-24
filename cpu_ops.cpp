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
    constexpr int igs = 4;
    constexpr int jgs = 4;
    constexpr int kgs = 2;

    constexpr int ithread = 4;
    constexpr int jthread = 4;
    constexpr int kthread = 8;

    constexpr int igroup = ithread * igs;
    constexpr int jgroup = jthread * jgs;
    constexpr int kgroup = kthread * kgs;

    for(int ib = 0; ib < isize; ib += igroup){
        for(int jb = 0; jb < jsize; jb += jgroup){

            float block_sum[igroup][jgroup] = {0};

            for(int kb = 0; kb < ksize; kb += kgroup){
                for(int kt = 0; kt < kthread; kt++){

                }

                float A_block[igroup][kgroup];
                float B_block[jgroup][kgroup];

                for(int ig = 0; ig < igs; ig++){
                    for(int kg = 0; kg < kgs; kg++){
                        for(int it = 0; it < ithread; it++){
                            for(int kt = 0; kt < kthread; kt++){
                                int i = it + ig * ithread;
                                int k = kt + kg * kthread;
                                A_block[i][k] = A[(ib+i)*ksize + (kb + k)];
                            }
                        }
                    }
                }
                for(int jg = 0; jg < jgs; jg++){
                    for(int kg = 0; kg < kgs; kg++){
                        for(int jt = 0; jt < jthread; jt++){
                            for(int kt = 0; kt < kthread; kt++){
                                int j = jt + jg * jthread;
                                int k = kt + kg * kthread;
                                B_block[j][k] = B[(kb+k)*jsize + (jb + j)];
                            }
                        }
                    }
                }


                for(int ig = 0; ig < igs; ig++){
                    for(int jg = 0; jg < jgs; jg++){
                        for(int it = 0; it < ithread; it++){
                            for(int jt = 0; jt < jthread; jt++){
                                int i = it + ig * ithread;
                                int j = jt + jg * jthread;

                                float sum = 0;
                                for(int k = 0; k < kgroup; k++){
                                    sum += A_block[i][k] * B_block[j][k];
                                }
                                block_sum[i][j] += sum;
                            }
                        }
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



void many_4x4(float * A, float * B, float * res, int num){
    constexpr int isize = 4;
    constexpr int jsize = 4;
    constexpr int ksize = 4;
    constexpr int matsize = 16;
    for(int x = 0; x < num; x++){
        for(int i = 0; i < isize; i++){
            for(int j = 0; j < jsize; j++){
                float sum = 0;
                for(int k = 0; k < ksize; k++){
                     sum += A[i*ksize+k + x*matsize] * B[k*jsize+j + x*matsize];
                }
                res[i*jsize+j + x*matsize] = sum;
            }
        }
    }
}


}
