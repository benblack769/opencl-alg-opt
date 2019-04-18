
namespace cpu_ops {

/*
    res = A * B
    A : (isize,ksize)
    B : (ksize,jsize)
    res: (isize,jsize)
*/
void matmul(float * A, float * B, float * res, int isize, int jsize, int ksize);

}
