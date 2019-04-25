#include "cpu_ops.h"
#include "x86vec.hpp"

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
void matmulcubed(float * A, float * B, float * res_mat, int ISIZE, int JSIZE, int KSIZE){
    constexpr int IGS = 4;
    constexpr int JGS = 4;
    constexpr int KGS = 1;
    constexpr int ITHREAD = 4;
    constexpr int JTHREAD = 4;
    constexpr int KTHREAD = 4;

    constexpr int IGROUP = IGS * ITHREAD;
    constexpr int JGROUP = JGS * JTHREAD;
    constexpr int KGROUP = KGS * KTHREAD;

    constexpr int VSIZE = 4;
    using float4 = fvec4;
    for(int ib = 0; ib < ISIZE; ib += IGROUP){
        for(int jb = 0; jb < JSIZE; jb += JGROUP){
            for(int it = 0; it < IGROUP; it += ITHREAD){
                for(int jt = 0; jt < JGROUP; jt += JTHREAD){
                    int kt = 0;
                    float4 res[ITHREAD];
                    for(int x = 0; x < ITHREAD; x++){
                        res[x] = float4();
                    }

                    for(int kg = 0; kg < KSIZE; kg += KGROUP){
                        float ABuf[ITHREAD][KTHREAD];
                        for(int io = 0; io < ITHREAD; io++){
                            int i = io + it + ib;
                            for(int ko = 0; ko < KTHREAD; ko += VSIZE){
                                int k = ko + kt + kg;
                                //float4 data(&A[i * KSIZE + k]);
                                float darr[VSIZE];
                                for(int x = 0; x < VSIZE; x++){
                                    darr[x] = A[(i*KSIZE + k) + x];
                                }
                                for(int x = 0; x < VSIZE; x++){
                                    ABuf[io][ko+x] = darr[x];
                                }
                            }
                        }

                        for(int ko = 0; ko < KTHREAD; ko++){
                            int k = ko + kt + kg;
                            int j = jb + jt;
                            float4 jvec((k * JSIZE + j) + B);
                            //float4 ivec((i * KSIZE + k) + A);
                            //float * iarr = (float *)(&ivec);
                            for(int jo = 0; jo < JTHREAD; jo++){
                                int io = jo;
                                //int i = io + it + ib;
                                float ival = ABuf[io][ko];//A[i*KSIZE+k];
                                float4 ivalvec(ival);
                                float4 mulval = jvec * ivalvec;
                                res[jo] += mulval;
                            }
                        }
                    }

                    for(int io = 0; io < ITHREAD; io++){
                        int i = io + it + ib;
                        int j = jb + jt;
                        int idx = (i*JSIZE + j);

                        res[io].store(res_mat + idx);
                    }
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
