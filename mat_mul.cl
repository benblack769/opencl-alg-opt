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

#define IGROUP (IGS * ITHREAD)
#define JGROUP (JGS * JTHREAD)
#define KGROUP (KGS * KTHREAD)

#define VSIZE 4

#if (JGS <= KTHREAD)
#define JSPLITK (KTHREAD/JGS)
#else
#define JSPLITK 1
#endif

#if (IGS <= KTHREAD)
#define ISPLITK (KTHREAD/IGS)
#else
#define ISPLITK 1
#endif

kernel void matmul(global float * A, global float * B, global float * res_mat){
    int ib = get_group_id(0) * IGROUP;
    int jb = get_group_id(1) * JGROUP;

    int it = get_local_id(0) * ITHREAD;
    int jt = get_local_id(1) * JTHREAD;
    int kt = 0;//get_local_id(2) * KTHREAD;

    //local float block_sum[IGROUP][JGROUP];

    float4 res[VSIZE];
    for(int x = 0; x < VSIZE; x++){
        res[x] = 0;
    }

    for(int kg = 0; kg < KSIZE; kg += KGROUP){

#ifdef USELOCALMEM
        local float Abuf[KTHREAD*IGROUP];
        local float Bbuf[KTHREAD*JGROUP];

        int jn = get_local_id(1);
        if(jn < KTHREAD){
            for(int ko = jn*JSPLITK; ko < ((jn+1)*JSPLITK); ko++){
                for(int io = 0; io < ITHREAD; io++){
                    int i = io + it + ib;
                    int k = ko + kt + kg;
                    Abuf[(ko)*IGROUP + (io+it)] = A[i*KSIZE+k];
                }
            }
        }
        int in = get_local_id(0);
        if(in < KTHREAD){
            for(int ko = in*ISPLITK; ko < ((in+1)*ISPLITK); ko++){
                for(int jo = 0; jo < JTHREAD; jo++){
                    int k = ko + kt + kg;
                    int j = jo + jt + jb;
                    Bbuf[(ko)*JGROUP + (jo+jt)] = B[k*JSIZE+j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        for(int ko = 0; ko < KTHREAD; ko++){
            int k = ko + kt + kg;
            int j = jb + jt;
#ifdef USELOCALMEM
            float4 jvec = vload4((ko*JGROUP + jt)/VSIZE, Bbuf);
#else
            float4 jvec = vload4((k*JSIZE + j)/VSIZE, B);
#endif
            for(int jo = 0; jo < JTHREAD; jo++){
                int io = jo;
                int i = io + it + ib;
#ifdef USELOCALMEM
                float ival = Abuf[ko*IGROUP+it+io];//A[i*KSIZE+k];
#else
                float ival = A[i*KSIZE+k];
#endif
                res[jo] += jvec * ival;
            }
        }
    }
    for(int io = 0; io < ITHREAD; io++){
        int i = io + it + ib;
        int j = jb + jt;
        int idx = (i*JSIZE + j) / VSIZE;
        float4 cursum = res[io];

        vstore4(cursum,idx,res_mat);
    }
}

#endif
