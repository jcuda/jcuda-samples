#include <stdlib.h>

extern "C"
__global__ void allocatingKernel(void** devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice)
{
    int thread = threadIdx.x + blockDim.x * blockIdx.x;

    short* devicePointerAllocatedOnDevice = (short*) malloc(3 * sizeof(short));
    printf("In thread %d allocated %p\n", thread, devicePointerAllocatedOnDevice);
    for(int i=0; i < 3; i++)
    {
        devicePointerAllocatedOnDevice[i] = thread * 10 + i;
    }
    devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice[thread] = 
        devicePointerAllocatedOnDevice;
}

extern "C"
__global__ void copyingKernel(
    void** devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice, 
    void** devicePointerAllocatedOnHostToDevicePointersAllocatedOnHost)
{
    int thread = threadIdx.x + blockDim.x * blockIdx.x;

    short* devicePointerAllocatedOnDevice = (short*)devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice[thread];
    short* devicePointerAllocatedOnHost = (short*)devicePointerAllocatedOnHostToDevicePointersAllocatedOnHost[thread];
    
    printf("In thread %d copy from %p to %p\n", thread, devicePointerAllocatedOnDevice, devicePointerAllocatedOnHost);
    
    for(int i=0; i < 3; i++)
    {
        devicePointerAllocatedOnHost[i] = devicePointerAllocatedOnDevice[i];
    }
}

extern "C"
__global__ void freeingKernel(
    void** devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice)
{
    int thread = threadIdx.x + blockDim.x * blockIdx.x;

    short* devicePointerAllocatedOnDevice = (short*)devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice[thread];
    
    printf("In thread %d free %p\n", thread, devicePointerAllocatedOnDevice);
    
    free(devicePointerAllocatedOnDevice);
}

