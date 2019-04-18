
kernel void set_123(global int * data){
    data[get_global_id(0)] = 123;
}
