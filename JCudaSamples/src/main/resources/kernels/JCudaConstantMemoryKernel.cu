#define CONSTANT_MEMORY_SIZE 100
__constant__ float constantMemoryData[CONSTANT_MEMORY_SIZE];

extern "C"
__global__ void constantMemoryKernel(float* array, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && index < CONSTANT_MEMORY_SIZE) {
        array[index] = constantMemoryData[index];
    }
}
