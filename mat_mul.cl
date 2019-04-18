
kernel void matmul(global float * A, global float * B, global float * res){
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0;
    for(int k = 0; k < KSIZE; k++){
        sum += A[i*KSIZE+k] * B[k*JSIZE+j];
    }
    res[i*JSIZE+j] = sum;
}
