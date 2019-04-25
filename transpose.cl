
kernel void transpose(global float * A, global float * res){
    int i = get_global_id(0);
    int j = get_global_id(1);

    res[ISIZE*j + i] = A[JSIZE*i + j];
}
