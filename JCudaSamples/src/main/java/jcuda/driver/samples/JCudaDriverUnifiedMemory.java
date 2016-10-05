/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.samples;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY;
import static jcuda.driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL;
import static jcuda.driver.CUmemAttach_flags.CU_MEM_ATTACH_HOST;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAllocManaged;
import static jcuda.driver.JCudaDriver.cuStreamAttachMemAsync;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasSdot;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.cublasHandle;

/**
 * An example showing how to use Unified / Managed memory with the 
 * JCuda Driver API
 */
public class JCudaDriverUnifiedMemory
{
    public static void main(String[] args)
    {
        JCudaDriver.setExceptionsEnabled(true);
        JCublas.setExceptionsEnabled(true);
        
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        
        // Check if the device supports managed memory
        int supported[] = { 0 };
        cuDeviceGetAttribute(supported, 
            CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device);
        if (supported[0] == 0)
        {
            System.err.println("Device does not support managed memory");
            return;
        }

        // Allocate managed memory that is accessible to the host
        int n = 10;
        long size = n * Sizeof.FLOAT;
        CUdeviceptr p = new CUdeviceptr();
        cuMemAllocManaged(p, size, CU_MEM_ATTACH_HOST);

        // Obtain the byte buffer from the pointer. This is supported only
        // for memory that was allocated to be accessible on the host:
        ByteBuffer bb = p.getByteBuffer(0, size);
        
        System.out.println("Buffer on host side: " + bb);

        // Fill the buffer with sample data
        FloatBuffer fb = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
        for (int i = 0; i < n; i++)
        {
            fb.put(i, i);
        }

        // Make the buffer accessible to all devices
        cuStreamAttachMemAsync(null, p, 0,  CU_MEM_ATTACH_GLOBAL);
        cuStreamSynchronize(null);

        // Use the pointer in a device operation (here, a dot product with 
        // JCublas, for example). The data that was filled in by the host
        // will now be used by the device.
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);
        float result[] = { -1.0f };
        cublasSdot(handle, n, p, 1, p, 1, Pointer.to(result));
        System.out.println("Result: " + result[0]);
    }
}
