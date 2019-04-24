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
kernel void matmul(global float * A, global float * B, global float * res){
    int ib = get_group_id(0) * IGROUP;
    int jb = get_group_id(1) * JGROUP;

    int i = get_local_id(0);
    int j = get_local_id(1);

    float sum = 0;

    for(int kb = 0; kb < KSIZE; kb += KGROUP){
        local float A_block[IGROUP][KGROUP];
        local float B_block[JGROUP][KGROUP];

#if (KGROUP >= IGROUP)
        {
        int ISPLITS = KGROUP / IGROUP;
        for(int k = i * ISPLITS; k < (i+1) * ISPLITS; k++){
            B_block[j][k] = B[(kb+k)*JSIZE + (jb + j)];
        }
        }
#else
        if(i < KGROUP){
            int k = i;
            B_block[j][k] = B[(kb+k)*JSIZE + (jb + j)];
        }
#endif

#if (KGROUP >= JGROUP)
        {
        int JSPLITS = KGROUP / JGROUP;
        for(int k = j * JSPLITS; k < (j+1) * JSPLITS; k++){
            A_block[i][k] =  A[(ib+i)*KSIZE + (kb + k)];
        }
        }
#else
        if(j < KGROUP){
            int k = j;
            A_block[i][k] =  A[(ib+i)*KSIZE + (kb + k)];
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < KGROUP; k++){
            sum += A_block[i][k] * B_block[j][k];
        }
    }
    res[(ib+i)*JSIZE + (jb+j)] = sum;
}

#endif
