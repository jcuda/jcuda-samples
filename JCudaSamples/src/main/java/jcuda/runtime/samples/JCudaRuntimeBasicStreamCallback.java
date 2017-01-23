/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2017 Marco Hutter - http://www.jcuda.org
 */
package jcuda.runtime.samples;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpyAsync;
import static jcuda.runtime.JCuda.cudaStreamAddCallback;
import static jcuda.runtime.JCuda.cudaStreamCreate;
import static jcuda.runtime.JCuda.cudaStreamSynchronize;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStreamCallback;
import jcuda.runtime.cudaStream_t;

/**
 * A very basic example / test for the stream callback functionality in the
 * JCuda Runtime API
 */
public class JCudaRuntimeBasicStreamCallback
{
    /**
     * Entry point of this program
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        JCuda.setExceptionsEnabled(true);

        // The stream on which the callbacks will be registered.
        // When this is "null", then it is the default stream.
        cudaStream_t stream = null;

        boolean useDefaultStream = true;
        useDefaultStream = false;
        if (!useDefaultStream)
        {
            stream = new cudaStream_t();
            cudaStreamCreate(stream);
        }
        System.out.println("Using stream " + stream);

        // Define the callback
        cudaStreamCallback callback = new cudaStreamCallback()
        {
            @Override
            public void call(cudaStream_t stream, int status, Object userData)
            {
                System.out.println("Callback called");
                System.out.println("    stream  : " + stream);
                System.out.println("    status  : " + status);
                System.out.println("    userData: " + userData);
                System.out.println("    thread  : " + Thread.currentThread());
            }
        };

        // Create some dummy data on the host, and copy it to the
        // device asynchronously
        int n = 100000;
        float hostData[] = new float[n];
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, n * Sizeof.FLOAT);
        cudaMemcpyAsync(deviceData, Pointer.to(hostData), 
            n * Sizeof.FLOAT, cudaMemcpyHostToDevice, stream);

        // Add the callback to the stream that carries the copy operation
        Object userData = "Example user data";
        cudaStreamAddCallback(stream, callback, userData, 0);

        // Wait until the stream is finished
        cudaStreamSynchronize(stream);

        // Clean up
        cudaFree(deviceData);

        System.out.println("Done");
    }

}
