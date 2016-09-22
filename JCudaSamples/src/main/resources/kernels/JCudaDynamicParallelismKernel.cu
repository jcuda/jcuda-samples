/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
#include <stdio.h>

// A simple example of using dynamic parallelism. This kernel can
// be compiled into an object file by calling
//
//     nvcc -dc -arch=sm_52 JCudaDynamicParallelismKernel.cu -o JCudaDynamicParallelismKernel.o
// 
// The resulting object file can be linked into a CUBIN file with
// 
//     nvcc -dlink -arch=sm_52 -cubin JCudaDynamicParallelismKernel.o -o JCudaDynamicParallelismKernel.cubin
// 
// Alternatively, both steps can be taken at once, by calling
// 
//     nvcc -dlink -arch=sm_52 -cubin -c JCudaDynamicParallelismKernel.cu -o JCudaDynamicParallelismKernel.cubin
// 
// The architecture (here, sm_52) must match the architecture of
// the target device. 

extern "C"
__global__ void childKernel(unsigned int parentThreadIndex, float* data)
{
    printf("Parent thread index: %d, child thread index: %d\n", 
        parentThreadIndex, threadIdx.x);    
    data[threadIdx.x] = parentThreadIndex + 0.1f * threadIdx.x;
}

extern "C"
__global__ void parentKernel(unsigned int size, float *data)
{
    childKernel<<<1, 8>>>(threadIdx.x, data + threadIdx.x * 8);
    cudaDeviceSynchronize();
    __syncthreads();
}
