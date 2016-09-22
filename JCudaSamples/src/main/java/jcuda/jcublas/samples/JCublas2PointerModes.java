/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcublas.samples;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasSdot;
import static jcuda.jcublas.JCublas2.cublasSetPointerMode;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_HOST;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;


/**
 * A sample demonstrating the different pointer modes for CUBLAS 2.
 * With CUBLAS 2, functions may receive pointers as arguments which are
 * either used as input parameters or will store results. These pointers
 * may either be pointers to host or to device memory. This sample shows
 * how to obtain the result of a 'dot' operation in host- or device 
 * memory.
 */
public class JCublas2PointerModes
{
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        // Enable exceptions and omit subsequent error checks
        JCublas2.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);

        // Create the input data: A vector containing the
        // value 1.0 exactly n times.
        int n = 1000000;
        float hostData[] = new float[n];
        Arrays.fill(hostData,  1.0f);

        // Allocate device memory, and copy the input data to the device
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, n * Sizeof.FLOAT);
        cudaMemcpy(deviceData, Pointer.to(hostData), n * Sizeof.FLOAT,
            cudaMemcpyHostToDevice);

        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);


        // Execute the 'dot' function in HOST pointer mode:
        // The result will be written to a pointer that
        // points to host memory.

        // Set the pointer mode to HOST
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

        // Prepare the pointer for the result in HOST memory
        float hostResult[] = { -1.0f };
        Pointer hostResultPointer = Pointer.to(hostResult);

        // Execute the 'dot' function
        long beforeHostCall = System.nanoTime();
        cublasSdot(handle, n, deviceData, 1, deviceData, 1, hostResultPointer);
        long afterHostCall = System.nanoTime();

        // Print the result and timing information
        double hostDuration = (afterHostCall - beforeHostCall) / 1e6;
        System.out.println("Host call duration: " + hostDuration + " ms");
        System.out.println("Result: " + hostResult[0]);


        // Execute the 'dot' function in DEVICE pointer mode:
        // The result will be written to a pointer that
        // points to device memory.

        // Set the pointer mode to DEVICE
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

        // Prepare the pointer for the result in DEVICE memory
        Pointer deviceResultPointer = new Pointer();
        cudaMalloc(deviceResultPointer, Sizeof.FLOAT);

        // Execute the 'dot' function
        long beforeDeviceCall = System.nanoTime();
        cublasSdot(handle, n, deviceData, 1, deviceData, 1,
            deviceResultPointer);
        long afterDeviceCall = System.nanoTime();

        // Synchronize in order to wait for the result to
        // be available (note that this is done implicitly
        // when cudaMemcpy is called)
        cudaDeviceSynchronize();
        long afterDeviceSync = System.nanoTime();

        // Copy the result from the device to the host
        float deviceResult[] = { -1.0f };
        cudaMemcpy(Pointer.to(deviceResult), deviceResultPointer, 
            Sizeof.FLOAT, cudaMemcpyDeviceToHost);

        // Print the result and timing information
        double deviceCallDuration = (afterDeviceCall - beforeDeviceCall) / 1e6;
        double deviceFullDuration = (afterDeviceSync - beforeDeviceCall) / 1e6;
        System.out .println(
            "Device call duration: " + deviceCallDuration + " ms");
        System.out.println(
            "Device full duration: " + deviceFullDuration + " ms");
        System.out.println("Result: " + deviceResult[0]);

        // Clean up
        cudaFree(deviceData);
        cublasDestroy(handle);
    }


}
