/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.samples;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.samples.utils.JCudaSamplesUtils;

/**
 * A simple example showing how a kernel with dynamic parallelism 
 * can be loaded from a CUBIN file and launched.
 */
public class JCudaDynamicParallelism
{
    public static void main(String[] args)
    {
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize a context for the first device
        cuInit(0);
        CUcontext context = new CUcontext();
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Create the CUBIN file by calling the NVCC. 
        // See the prepareDefaultCubinFile method for the details about
        // the NVCC parameters that are used here. 
        String cubinFileName = JCudaSamplesUtils.prepareDefaultCubinFile(
            "src/main/resources/kernels/JCudaDynamicParallelismKernel.cu");

        // Load the CUBIN file 
        CUmodule module = new CUmodule();
        cuModuleLoad(module, cubinFileName);

        // Obtain a function pointer to the "parentKernel" function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "parentKernel");

        // Define the nesting structure. 
        // 
        // NOTE: The number of child threads MUST match the value that 
        // is used in the kernel, for the childKernel<<<1, 8>>> call!
        // 
        int numParentThreads = 8;
        int numChildThreads = 8;

        // Allocate the device data that will be filled by the kernel
        int numElements = numParentThreads * numChildThreads;
        CUdeviceptr deviceData = new CUdeviceptr();
        cuMemAlloc(deviceData, numElements * Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new int[] { numElements }),
            Pointer.to(deviceData)
        );

        // Call the kernel function.
        int blockSizeX = numParentThreads;
        int gridSizeX = (numElements + numElements - 1) / blockSizeX;
        cuLaunchKernel(function,
            gridSizeX,  1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Copy the device data to the host
        float hostData[] = new float[numElements];
        for(int i = 0; i < numElements; i++)
        {
            hostData[i] = i;
        }
        cuMemcpyDtoH(Pointer.to(hostData), 
            deviceData, numElements * Sizeof.FLOAT);

        // Compare the host data with the expected values
        float hostDataRef[] = new float[numElements];
        for(int i = 0; i < numParentThreads; i++)
        {
            for (int j=0; j < numChildThreads; j++)
            {
                hostDataRef[i * numChildThreads + j] = i + 0.1f * j;
            }
        }
        System.out.println("Result: "+Arrays.toString(hostData));
        boolean passed = Arrays.equals(hostData, hostDataRef);
        System.out.println(passed ? "PASSED" : "FAILED");

        // Clean up.
        cuMemFree(deviceData);
    }
}

