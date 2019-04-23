
kernel void transpose2(global float * A, global float * res){
    int i = get_global_id(0);
    int j = get_global_id(1);

    res[ISIZE*j + i] = A[JSIZE*i + j];
}

kernel void transpose(global float * A, global float * res){
    int i = get_global_id(0)*ASIZE;
    int j = get_global_id(1)*BSIZE;

    float buff[ASIZE][BSIZE];

    for (int y = 0; y < ASIZE; y++){
        for(int x = 0; x < BSIZE; x++){
            buff[y][x] = A[JSIZE*(i+y) + (j+x)];
        }
    }
    float buff2[BSIZE][ASIZE];
    for (int y = 0; y < ASIZE; y++){
        for(int x = 0; x < BSIZE; x++){
            buff2[x][y] = buff[y][x];
        }
    }
    for(int x = 0; x < BSIZE; x++){
        for (int y = 0; y < ASIZE; y++){
            res[ISIZE*(j + x) + (i + y)] = buff2[x][y];
        }
    }
}
