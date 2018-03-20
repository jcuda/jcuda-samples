/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2018 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.samples;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.*;

import java.io.IOException;
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
 * An example showing how to use constant memory in kernels. <br>
 */
public class JCudaConstantMemoryExample 
{
    public static void main(String[] args) throws IOException 
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Create the PTX file by calling the NVCC
        String ptxFileName = JCudaSamplesUtils.preparePtxFile(
            "src/main/resources/kernels/JCudaConstantMemoryKernel.cu");

        // Load the PTX file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain the pointer to the constant memory, and print some info
        CUdeviceptr constantMemoryPointer = new CUdeviceptr();
        long constantMemorySizeArray[] = { 0 };
        cuModuleGetGlobal(constantMemoryPointer, constantMemorySizeArray, 
            module, "constantMemoryData");
        int constantMemorySize = (int)constantMemorySizeArray[0];
        
        System.out.println("constantMemoryPointer: " + constantMemoryPointer);
        System.out.println("constantMemorySize: " + constantMemorySize);

        // Copy some host data to the constant memory
        int numElements = constantMemorySize / Sizeof.FLOAT;
        float hostData[] = new float[numElements];
        for (int i = 0; i < numElements; i++)
        {
            hostData[i] = i;
        }
        cuMemcpyHtoD(constantMemoryPointer, 
            Pointer.to(hostData), constantMemorySize);
        
        // Now use the constant memory in the kernel call:
        
        // Obtain a function pointer to the "constantMemoryKernel" function.
        CUfunction kernel = new CUfunction();
        cuModuleGetFunction(kernel, module, "constantMemoryKernel");

        // Allocate some device memory
        CUdeviceptr deviceData = new CUdeviceptr();
        cuMemAlloc(deviceData, constantMemorySize);
        
        // Set up the kernel parameters
        Pointer kernelParameters = Pointer.to(
            Pointer.to(deviceData),
            Pointer.to(new int[]{numElements})
        );
        
        // Launch the kernel
        int blockSizeX = numElements;
        int gridSizeX = 1;
        cuLaunchKernel(kernel,
            gridSizeX,  1, 1, 
            blockSizeX, 1, 1,
            0, null,         
            kernelParameters, null 
        );
        cuCtxSynchronize();
        
        // Copy the result back to the host, and verify that it is
        // the same that was copied to the constant memory
        float hostResult[] = new float[numElements];
        cuMemcpyDtoH(Pointer.to(hostResult), deviceData, constantMemorySize);
        
        boolean passed = Arrays.equals(hostData,  hostResult);
        System.out.println("Test " + (passed ? "PASSED" : "FAILED"));
    }
}
