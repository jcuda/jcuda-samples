/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2017 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.samples;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchHostFunc;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUhostFn;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

/**
 * An example showing how to call a host function via the driver API
 */
public class JCudaDriverHostFunction
{
    /**
     * Entry point
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        // Default initialization
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUcontext context = new CUcontext();
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Create a stream
        CUstream stream = new CUstream();
        cuStreamCreate(stream, 0);
        
        // Define a host function and launch it
        CUhostFn fn = new CUhostFn()
        {
            @Override
            public void call(Object userData)
            {
                System.out.println("Called with " + userData);
            }
        };
        cuLaunchHostFunc(stream, fn, "Example user object");
        
        // Wait for the stream to finish
        cuStreamSynchronize(stream);

        // Clean up
        cuCtxDestroy(context);
        
        System.out.println("Done");
    }
}

