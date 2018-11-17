/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2018 Marco Hutter - http://www.jcuda.org
 */
package jcuda.runtime.samples;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 * A sample that prints information about all available CUDA devices
 */
public class JCudaPrintDeviceInfo
{
    public static void main(String[] args)
    {
        JCuda.setExceptionsEnabled(true);
        int deviceCount[] = { 0 };
        cudaGetDeviceCount(deviceCount);
        System.out.println("Found " + deviceCount[0] + " devices");
        for (int device = 0; device < deviceCount[0]; device++)
        {
            System.out.println("Properties of device " + device + ":");
            cudaDeviceProp deviceProperties = new cudaDeviceProp();
            cudaGetDeviceProperties(deviceProperties, device);
            System.out.println(deviceProperties.toFormattedString());
        }
        
    }
}
