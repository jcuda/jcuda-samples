package jcuda.driver.samples;
/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2011-2018 Marco Hutter - http://www.jcuda.org
 */
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;

import java.util.Locale;
import java.util.Random;

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
 * Example of a reduction. It is based on the NVIDIA 'reduction' sample, 
 * and uses an adapted version of one of the kernels presented in 
 * this sample (see src/main/resources/kernels/JCudaReductionKernel.cu) 
 */
public class JCudaReduction
{
    /**
     * The CUDA context created by this sample
     */
    private static CUcontext context;
    
    /**
     * The module which is loaded in form of a PTX file
     */
    private static CUmodule module;
    
    /**
     * The actual kernel function from the module
     */
    private static CUfunction function;
    
    /**
     * Temporary memory for the device output
     */
    private static CUdeviceptr deviceBuffer;
    
    /**
     * Entry point of this sample
     *
     * @param args Not used
     */
    public static void main(String args[])
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        init();
        boolean passed = true;
        for (int n = 100000; n <= 26500000; n *= 2)
        {
            float hostInput[] = createRandomArray(n);

            long timeNs0 = 0;
            long timeNs1 = 0;

            // Copy the input data to the device
            timeNs0 = System.nanoTime();
            CUdeviceptr deviceInput = new CUdeviceptr();
            cuMemAlloc(deviceInput, hostInput.length * Sizeof.FLOAT);
            cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), 
                hostInput.length * Sizeof.FLOAT);
            timeNs1 = System.nanoTime();
            long durationCopyNs = timeNs1 - timeNs0;

            // Execute the reduction with CUDA
            timeNs0 = System.nanoTime();
            float resultJCuda = reduce(deviceInput, hostInput.length);
            timeNs1 = System.nanoTime();
            long durationCompNs = timeNs1 - timeNs0;

            cuMemFree(deviceInput);

            // Execute the reduction with Java
            timeNs0 = System.nanoTime();
            float resultJava = reduceHost(hostInput);
            timeNs1 = System.nanoTime();
            long durationJavaNs = timeNs1 - timeNs0;

            System.out.println("Reduction of " + n + " elements");
            System.out.printf(Locale.ENGLISH,
                "  JCuda: %7.3f ms, result: %f " +
                "(copy: %7.3f ms, comp: %7.3f ms)\n",
                (durationCopyNs + durationCompNs) / 1e6, resultJCuda, 
                durationCopyNs / 1e6, durationCompNs / 1e6);
            System.out.printf(Locale.ENGLISH,
                "  Java : %7.3f ms, result: %f\n", 
                durationJavaNs / 1e6, resultJava);
            
            passed &= 
                Math.abs(resultJCuda - resultJava) < resultJava * 1e-5;
            
        }
        System.out.println("Test " + (passed ? "PASSED" : "FAILED"));

        shutdown();
    }    
    
    
    /**
     * Implementation of a Kahan summation reduction in plain Java
     * 
     * @param input The input 
     * @return The reduction result
     */
    private static float reduceHost(float data[])
    {
        float sum = data[0];
        float c = 0.0f;              
        for (int i = 1; i < data.length; i++)
        {
            float y = data[i] - c;  
            float t = sum + y;      
            c = (t - sum) - y;  
            sum = t;            
        }
        return sum;
    }
    
    
    /**
     * Initialize the context, module, function and other elements used 
     * in this sample
     */
    private static void init()
    {
        // Initialize the driver API and create a context for the first device
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Create the PTX file by calling the NVCC
        String ptxFileName = JCudaSamplesUtils.preparePtxFile(
            "src/main/resources/kernels/JCudaReductionKernel.cu");
        
        // Load the module from the PTX file
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "reduce" function.
        function = new CUfunction();
        cuModuleGetFunction(function, module, "reduce");
        
        // Allocate a chunk of temporary memory (must be at least
        // numberOfBlocks * Sizeof.FLOAT)
        deviceBuffer = new CUdeviceptr();
        cuMemAlloc(deviceBuffer, 1024 * Sizeof.FLOAT);
        
    }
    
    /**
     * Release all resources allocated by this class
     */
    private static void shutdown()
    {
        cuModuleUnload(module);
        cuMemFree(deviceBuffer);
        cuCtxDestroy(context);
    }
    
    /**
     * Performs a reduction on the given device memory with the given
     * number of elements.
     * 
     * @param deviceInput The device input memory
     * @param numElements The number of elements to reduce
     * @return The reduction result
     */
    private static float reduce(
        Pointer deviceInput, int numElements)
    {
        return reduce(deviceInput, numElements, 128, 64);
    }
    
    
    /**
     * Performs a reduction on the given device memory with the given
     * number of elements and the specified limits for threads and
     * blocks.
     * 
     * @param deviceInput The device input memory
     * @param numElements The number of elements to reduce
     * @param maxThreads The maximum number of threads
     * @param maxBlocks The maximum number of blocks
     * @return The reduction result
     */
    private static float reduce(
        Pointer deviceInput, int numElements, 
        int maxThreads, int maxBlocks)
    {
        // Determine the number of threads and blocks for the input 
        int numBlocks = getNumBlocks(numElements, maxBlocks, maxThreads);
        int numThreads = getNumThreads(numElements, maxBlocks, maxThreads);
        
        // Call the main reduction method
        float result = reduce(numElements, numThreads, numBlocks, 
            maxThreads, maxBlocks, deviceInput);
        return result;
    }
    

    
    /**
     * Performs a reduction on the given device memory.
     * 
     * @param n The number of elements for the reduction
     * @param numThreads The number of threads
     * @param numBlocks The number of blocks
     * @param maxThreads The maximum number of threads
     * @param maxBlocks The maximum number of blocks
     * @param deviceInput The input memory
     * @return The reduction result
     */
    private static float reduce(
        int  n, int  numThreads, int  numBlocks,
        int  maxThreads, int  maxBlocks, Pointer deviceInput)
    {
        // Perform a "tree like" reduction as in the NVIDIA sample
        reduce(n, numThreads, numBlocks, deviceInput, deviceBuffer);
        int s = numBlocks;
        while(s > 1) 
        {
            int threads = getNumThreads(s, maxBlocks, maxThreads);
            int blocks = getNumBlocks(s, maxBlocks, maxThreads);

            reduce(s, threads, blocks, deviceBuffer, deviceBuffer);
            s = (s + (threads * 2 - 1)) / (threads * 2);
        }
        
        float result[] = {0.0f};
        cuMemcpyDtoH(Pointer.to(result), deviceBuffer, Sizeof.FLOAT);     
        return result[0];
    }
    
    
    /**
     * Perform a reduction of the specified number of elements in the given 
     * device input memory, using the given number of threads and blocks, 
     * and write the results into the given output memory. 
     * 
     * @param size The size (number of elements) 
     * @param threads The number of threads
     * @param blocks The number of blocks
     * @param deviceInput The device input memory
     * @param deviceOutput The device output memory. Its size must at least 
     * be numBlocks*Sizeof.FLOAT
     */
    private static void reduce(int size, int threads, int blocks, 
        Pointer deviceInput, Pointer deviceOutput)
    {
        // Compute the shared memory size (as done in 
        // the NIVIDA sample)
        int sharedMemSize = threads * Sizeof.FLOAT;
        if (threads <= 32) 
        {
            sharedMemSize *= 2;
        }
        
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(deviceInput),
            Pointer.to(deviceOutput),
            Pointer.to(new int[]{size})
        );

        // Call the kernel function.
        cuLaunchKernel(function,
            blocks,  1, 1,         // Grid dimension
            threads, 1, 1,         // Block dimension
            sharedMemSize, null,   // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
    }
    
    
    /**
     * Compute the number of blocks that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of blocks
     */
    private static int getNumBlocks(int n, int maxBlocks, int maxThreads)
    {
        int blocks = 0;
        int threads = getNumThreads(n, maxBlocks, maxThreads);
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    /**
     * Compute the number of threads that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of threads
     */
    private static int getNumThreads(int n, int maxBlocks, int maxThreads)
    {
        int threads = 0;
        threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
        return threads;
    }
    
    /**
     * Returns the power of 2 that is equal to or greater than x
     * 
     * @param x The input
     * @return The next power of 2
     */
    private static int nextPow2(int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    
    /**
     * Create an array of the given size, with random data
     * 
     * @param size The array size
     * @return The array
     */
    private static float[] createRandomArray(int size)
    {
        Random random = new Random(0);
        float array[] = new float[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = random.nextFloat() * 0.01f;
        }
        return array;
    }
}


