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

    int ISPLITS = KGROUP > ISIZE ? KGROUP / ISIZE : 1;
    int JSPLITS = KGROUP > JSIZE ? KGROUP / JSIZE : 1;

    float sum = 0;

    for(int kb = 0; kb < KSIZE; kb += KGROUP){
        local float A_block[IGROUP][KGROUP];
        local float B_block[JGROUP][KGROUP];

        if(j == 0){
            for(int k = 0; k < KGROUP; k++){
                A_block[i][k] =  A[(ib+i)*KSIZE + (kb + k)];
            }
        }
        if(i == 0){
            for(int k = 0; k < KGROUP; k++){
                B_block[j][k] = B[(kb+k)*JSIZE + (jb + j)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < KGROUP; k++){
            sum += A_block[i][k] * B_block[j][k];
        }
    }
    res[(ib+i)*JSIZE + (jb+j)] = sum;
}

#endif
