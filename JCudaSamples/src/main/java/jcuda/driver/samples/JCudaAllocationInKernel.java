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
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

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
 * An example showing how to allocate memory in kernels. <br>
 * <br>
 * Kernels may allocate memory, using the standard <code>malloc</code> and
 * <code>free</code> functions. When used inside a kernel, these functions
 * will allocate device memory. This device memory can NOT be used in 
 * host functions (not even the ones that operate on device memory!).
 * The device memory that was allocated on the device is thus not compatible
 * with the device memory that was allocated on the host. 
 * See http://stackoverflow.com/a/13043240 for details.<br>
 * <br>
 * This example shows how to allocate, use and free memory in kernels. The 
 * usage pattern shown here does not necessarily make any sense, but it points 
 * out the difference between device memory allocated on the host, and device 
 * memory allocated on the device, using overly elaborate variable names.  
 */
public class JCudaAllocationInKernel 
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
            "src/main/resources/kernels/JCudaAllocationInKernelKernel.cu");

        // Load the PTX file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "allocatingKernel" function.
        CUfunction allocatingKernel = new CUfunction();
        cuModuleGetFunction(allocatingKernel, module, "allocatingKernel");

        // Obtain a function pointer to the "copyingKernel" function.
        CUfunction copyingKernel = new CUfunction();
        cuModuleGetFunction(copyingKernel, module, "copyingKernel");

        // Obtain a function pointer to the "freeingKernel" function.
        CUfunction freeingKernel = new CUfunction();
        cuModuleGetFunction(freeingKernel, module, "freeingKernel");

        int numThreads = 4;

        // NOTE: This must match the value in the kernels! 
        int numberOfShortsAllocatedInKernel = 3;

        // What will arrive in the allocating kernel: A device pointer that is 
        // allocated on the host. Each element of this "array" will afterwards 
        // contain a device pointer that was allocated on the device.
        CUdeviceptr devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice = 
            new CUdeviceptr();
        cuMemAlloc(devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice, 
            numThreads * Sizeof.POINTER);

        // The parameter for the allocating kernel: 
        // A pointer to a pointer that points to the device pointer that 
        // was allocated on the host, and points to the device pointers 
        // that will be allocated on the device. Yeah.
        Pointer allocatingKernelParameters = Pointer.to(
            Pointer.to(devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice)
            );

        // Launch the allocating kernel
        int blockSizeX = numThreads;
        int gridSizeX = 1;
        cuLaunchKernel(allocatingKernel,
            gridSizeX,  1, 1, 
            blockSizeX, 1, 1,
            0, null,         
            allocatingKernelParameters, null 
        );
        cuCtxSynchronize();

        // Create the (host) array of device pointers that are allocated on
        // the host
        CUdeviceptr devicePointersAllocatedOnHost[] = 
            new CUdeviceptr[numThreads];
        for (int i=0; i<numThreads; i++)
        {
            devicePointersAllocatedOnHost[i] = new CUdeviceptr();
            cuMemAlloc(devicePointersAllocatedOnHost[i], 
                numberOfShortsAllocatedInKernel * Sizeof.SHORT);
        }

        // Allocate a device pointer on the host, and fill it with the 
        // device pointers that are allocated on the host
        CUdeviceptr devicePointerAllocatedOnHostToDevicePointersAllocatedOnHost = 
            new CUdeviceptr();
        cuMemAlloc(devicePointerAllocatedOnHostToDevicePointersAllocatedOnHost, 
            numThreads * Sizeof.POINTER);
        cuMemcpyHtoD(devicePointerAllocatedOnHostToDevicePointersAllocatedOnHost, 
            Pointer.to(devicePointersAllocatedOnHost), 
            numThreads * Sizeof.POINTER);

        // The parameters for the copying kernel:
        // - A pointer to a pointer that points to the device pointer that 
        //   was allocated on the host, and points to the device pointers 
        //   that have been be allocated on the device.
        // - A pointer to a pointer that points to the device pointer that 
        //   was allocated on the host, and points to the device pointers 
        //   that have been be allocated on the host
        Pointer copyingKernelParameters = Pointer.to(
            Pointer.to(devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice),
            Pointer.to(devicePointerAllocatedOnHostToDevicePointersAllocatedOnHost)
            );

        // Launch the copying kernel
        cuLaunchKernel(copyingKernel,
            gridSizeX,  1, 1, 
            blockSizeX, 1, 1,
            0, null,         
            copyingKernelParameters, null 
        );
        cuCtxSynchronize();

        // Copy the contents of each pointer into a result array
        short resultArrays[][] = new short[numThreads][numberOfShortsAllocatedInKernel];
        for (int i=0; i<numThreads; i++)
        {
            cuMemcpyDtoH(Pointer.to(resultArrays[i]), 
                devicePointersAllocatedOnHost[i], 
                numberOfShortsAllocatedInKernel * Sizeof.SHORT);        
        }

        // The parameters for the freeing kernel: 
        // The same as for the allocating kernel
        Pointer freeingKernelParameters = Pointer.to(
            Pointer.to(devicePointerAllocatedOnHostToDevicePointersAllocatedOnDevice)
            );

        // Launch the freeing kernel
        cuLaunchKernel(freeingKernel,
            gridSizeX,  1, 1, 
            blockSizeX, 1, 1,
            0, null,         
            freeingKernelParameters, null 
        );
        cuCtxSynchronize();


        // Print the results
        for (int i = 0; i < numThreads; i++)
        {
            System.out.println("Result from thread " + i + " is " + 
                Arrays.toString(resultArrays[i]));
        }
    }


}
