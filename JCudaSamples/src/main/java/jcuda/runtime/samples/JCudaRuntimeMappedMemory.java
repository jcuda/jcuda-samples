/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.runtime.samples;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasSscal;
import static jcuda.runtime.JCuda.cudaDeviceMapHost;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFreeHost;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;
import static jcuda.runtime.JCuda.cudaHostAlloc;
import static jcuda.runtime.JCuda.cudaHostAllocMapped;
import static jcuda.runtime.JCuda.cudaHostGetDevicePointer;
import static jcuda.runtime.JCuda.cudaSetDeviceFlags;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 * An example showing how to use mapped memory in JCuda. Host memory is
 * allocated and mapped to the device. There, it is modified with a 
 * runtime library function (CUBLAS, for example), which then 
 * effectively writes to host memory.
 */
public class JCudaRuntimeMappedMemory
{
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String args[])
    {
        // Enable exceptions to quickly be informed about errors in this test
        JCuda.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);

        // Check if the device supports mapped host memory
        cudaDeviceProp deviceProperties = new cudaDeviceProp();
        cudaGetDeviceProperties(deviceProperties, 0);
        if (deviceProperties.canMapHostMemory == 0)
        {
            System.err.println("This device can not map host memory");
            System.err.println(deviceProperties.toFormattedString());
            return;
        }

        // Set the flag indicating that mapped memory will be used
        cudaSetDeviceFlags(cudaDeviceMapHost);

        // Allocate mappable host memory
        int n = 5;
        Pointer hostPointer = new Pointer();
        cudaHostAlloc(hostPointer, n * Sizeof.FLOAT, cudaHostAllocMapped);

        // Create a device pointer mapping the host memory
        Pointer devicePointer = new Pointer();
        cudaHostGetDevicePointer(devicePointer, hostPointer, 0);

        // Obtain a ByteBuffer for accessing the data in the host
        // pointer. Modifications in this ByteBuffer will be
        // visible in the device memory.
        ByteBuffer byteBuffer = hostPointer.getByteBuffer(0, n * Sizeof.FLOAT);

        // Set the byte order of the ByteBuffer
        byteBuffer.order(ByteOrder.nativeOrder());

        // For convenience, view the ByteBuffer as a FloatBuffer
        // and fill it with some sample data
        FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
        System.out.print("Input : ");
        for (int i = 0; i < n; i++)
        {
            floatBuffer.put(i, (float) i);
            System.out.print(floatBuffer.get(i) + ", ");
        }
        System.out.println();

        // Apply a CUBLAS routine to the device pointer. This will
        // modify the host data, which was mapped to the device.
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);
        Pointer two = Pointer.to(new float[] { 2.0f });
        cublasSscal(handle, n, two, devicePointer, 1);
        cublasDestroy(handle);
        cudaDeviceSynchronize();

        // Print the contents of the host memory after the
        // modification via the mapped pointer.
        System.out.print("Output: ");
        for (int i = 0; i < n; i++)
        {
            System.out.print(floatBuffer.get(i) + ", ");
        }
        System.out.println();

        // Clean up
        cudaFreeHost(hostPointer);
    }
}