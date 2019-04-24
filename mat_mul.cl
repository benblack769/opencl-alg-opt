#ifdef DIRECT
kernel void matmul(global float * A, global float * B, global float * res){
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0;
    for(int k = 0; k < KSIZE; k++){
        sum += A[i*KSIZE+k] * B[k*JSIZE+j];
    }
    res[i*JSIZE+j] = sum;
}
#else

#define IGROUP (ITHREAD * IGS)
#define JGROUP (JTHREAD * JGS)
#define KGROUP (KTHREAD * KGS)

kernel void matmul(global float * A, global float * B, global float * res){
    int ib = get_group_id(0) * IGROUP;
    int jb = get_group_id(1) * JGROUP;

    int ig = get_local_id(0);
    int jg = get_local_id(1);

    float sums[ITHREAD][JTHREAD] = {0};

    for(int kb = 0; kb < KSIZE; kb += KGROUP){
        local float A_block[IGROUP][KGROUP];
        local float B_block[JGROUP][KGROUP];

#if (KGROUP >= IGS)
        {
        int i = ig;
        int ISPLITS = KGROUP / IGS;
        for(int k = i * ISPLITS; k < (i+1) * ISPLITS; k++){
            for(int jt = 0; jt < JTHREAD; jt++){
                int j = jt + jg * JTHREAD;
                B_block[j][k] = B[(kb+k)*JSIZE + (jb + j)];
            }
        }
        }
#else
        if(ig < KGROUP){
            int k = ig;
            for(int jt = 0; jt < JTHREAD; jt++){
                int j = jt + jg * JTHREAD;
                B_block[j][k] = B[(kb+k)*JSIZE + (jb + j)];
            }
        }
#endif

#if (KGROUP >= JGS)
        {
        int j = jg;
        int JSPLITS = KGROUP / JGS;
        for(int k = j * JSPLITS; k < (j+1) * JSPLITS; k++){
            for(int it = 0; it < ITHREAD; it++){
                int i = it + ig * ITHREAD;
                A_block[i][k] =  A[(ib+i)*KSIZE + (kb + k)];
            }
        }
        }
#else
        if(jg < KGROUP){
            int k = jg;
            for(int it = 0; it < ITHREAD; it++){
                int i = it + ig * ITHREAD;
                A_block[i][k] =  A[(ib+i)*KSIZE + (kb + k)];
            }
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int it = 0; it < ITHREAD; it++){
            for(int jt = 0; jt < JTHREAD; jt++){
                int i = it + ig * ITHREAD;
                int j = jt + jg * JTHREAD;

                float sum = 0;
                for(int k = 0; k < KGROUP; k++){
                    sum += A_block[i][k] * B_block[j][k];
                }
                sums[it][jt] += sum;
            }
        }
    }
    for(int it = 0; it < ITHREAD; it++){
        for(int jt = 0; jt < JTHREAD; jt++){
            int i = it + ig * ITHREAD;
            int j = jt + jg * JTHREAD;

            res[(ib+i)*JSIZE + (jb+j)] = sums[it][jt];
        }
    }
}

#endif
