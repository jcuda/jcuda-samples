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
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuStreamAddCallback;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.CUstreamCallback;
import jcuda.driver.JCudaDriver;

/**
 * A very basic example / test for the stream callback functionality in the
 * JCuda Driver API
 */
public class JCudaDriverBasicStreamCallback
{
    /**
     * Entry point of this program
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        JCudaDriver.setExceptionsEnabled(true);

        // Default initialization
        cuInit(0);
        CUcontext context = new CUcontext();
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // The stream on which the callbacks will be registered.
        // When this is "null", then it is the default stream.
        CUstream stream = null;

        boolean useDefaultStream = true;
        useDefaultStream = false;
        if (!useDefaultStream)
        {
            stream = new CUstream();
            cuStreamCreate(stream, 0);
        }
        System.out.println("Using stream " + stream);

        // Define the callback
        CUstreamCallback callback = new CUstreamCallback()
        {
            @Override
            public void call(CUstream hStream, int status, Object userData)
            {
                System.out.println("Callback called");
                System.out.println("    stream  : " + hStream);
                System.out.println("    status  : " + status);
                System.out.println("    userData: " + userData);
                System.out.println("    thread  : " + Thread.currentThread());
            }
        };

        // Create some dummy data on the host, and copy it to the
        // device asynchronously
        int n = 100000;
        float hostData[] = new float[n];
        CUdeviceptr deviceData = new CUdeviceptr();
        cuMemAlloc(deviceData, n * Sizeof.FLOAT);
        cuMemcpyHtoDAsync(deviceData, Pointer.to(hostData), 
            n * Sizeof.FLOAT, stream);

        // Add the callback to the stream that carries the copy operation
        Object userData = "Example user data";
        cuStreamAddCallback(stream, callback, userData, 0);

        // Wait until the stream is finished
        cuStreamSynchronize(stream);

        // Clean up
        cuMemFree(deviceData);
        cuCtxDestroy(context);

        System.out.println("Done");
    }

}
