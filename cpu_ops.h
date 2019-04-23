
namespace cpu_ops {

/*
    res = A * B
    A : (isize,ksize)
    B : (ksize,jsize)
    res: (isize,jsize)
*/
void matmul(float * A, float * B, float * res, int isize, int jsize, int ksize);
void matmulcubed(float * A, float * B, float * res, int isize, int jsize, int ksize);

/*
    A : (isize,jsize)
    res: (jsize,isize)
*/
void transpose(float * A, float * res, int isize, int jsize);
}
