/*
 * JCudaVec - Vector operations for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2013-2015 Marco Hutter - http://www.jcuda.org
 */
package jcuda.vec.samples;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.vec.VecFloat;

/**
 * A sample showing how to use the JCuda vector library
 */
public class VecFloatSample
{
    public static void main(String[] args)
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Afterwards, initialize the vector library, which will
        // attach to the current context
        VecFloat.init();
        
        // Allocate and fill the host input data
        int n = 50000;
        float hostX[] = new float[n];
        float hostY[] = new float[n];
        for(int i = 0; i < n; i++)
        {
            hostX[i] = (float)i;
            hostY[i] = (float)i;
        }

        // Allocate the device pointers, and copy the
        // host input data to the device
        CUdeviceptr deviceX = new CUdeviceptr();
        cuMemAlloc(deviceX, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceX, Pointer.to(hostX), n * Sizeof.FLOAT);

        CUdeviceptr deviceY = new CUdeviceptr();
        cuMemAlloc(deviceY, n * Sizeof.FLOAT); 
        cuMemcpyHtoD(deviceY, Pointer.to(hostY), n * Sizeof.FLOAT);

        CUdeviceptr deviceResult = new CUdeviceptr();
        cuMemAlloc(deviceResult, n * Sizeof.FLOAT);

        // Perform the vector operations
        VecFloat.cos(n, deviceX, deviceX);               // x = cos(x)  
        VecFloat.mul(n, deviceX, deviceX, deviceX);      // x = x*x
        VecFloat.sin(n, deviceY, deviceY);               // y = sin(y)
        VecFloat.mul(n, deviceY, deviceY, deviceY);      // y = y*y
        VecFloat.add(n, deviceResult, deviceX, deviceY); // result = x+y

        // Allocate host output memory and copy the device output
        // to the host.
        float hostResult[] = new float[n];
        cuMemcpyDtoH(Pointer.to(hostResult), deviceResult, n * Sizeof.FLOAT);

        // Verify the result
        boolean passed = true;
        for(int i = 0; i < n; i++)
        {
            float expected = (float)(
                Math.cos(hostX[i])*Math.cos(hostX[i])+
                Math.sin(hostY[i])*Math.sin(hostY[i]));
            if (Math.abs(hostResult[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+hostResult[i]+
                    " but expected "+expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        // Clean up.
        cuMemFree(deviceX);
        cuMemFree(deviceY);
        cuMemFree(deviceResult);
        VecFloat.shutdown();
    }

}
