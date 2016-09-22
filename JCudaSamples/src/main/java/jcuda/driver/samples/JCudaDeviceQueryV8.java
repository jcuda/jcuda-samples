/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.samples;

import static jcuda.driver.CUdevice_attribute.*;
import static jcuda.driver.JCudaDriver.*;

import java.util.*;

import jcuda.driver.*;

/**
 * An example that queries and prints all attributes of all available
 * devices. (CUDA 8 version)
 */
public class JCudaDeviceQueryV8
{
    /**
     * Entry point of this program
     * 
     * @param args Not used
     */
    public static void main(String args[])
    {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        // Obtain the number of devices
        int deviceCountArray[] = { 0 };
        cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];
        System.out.println("Found " + deviceCount + " devices");

        for (int i = 0; i < deviceCount; i++)
        {
            CUdevice device = new CUdevice();
            cuDeviceGet(device, i);

            // Obtain the device name
            byte deviceName[] = new byte[1024];
            cuDeviceGetName(deviceName, deviceName.length, device);
            String name = createString(deviceName);

            // Obtain the compute capability
            int majorArray[] = { 0 };
            int minorArray[] = { 0 };
            cuDeviceGetAttribute(majorArray,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
            cuDeviceGetAttribute(minorArray,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
            int major = majorArray[0];
            int minor = minorArray[0];

            System.out.println("Device " + i + ": " + name
                + " with Compute Capability " + major + "." + minor);

            // Obtain the device attributes
            int array[] = { 0 };
            List<Integer> attributes = getAttributes();
            for (Integer attribute : attributes)
            {
                String description = getAttributeDescription(attribute);
                cuDeviceGetAttribute(array, attribute, device);
                int value = array[0];

                System.out.printf("    %-80s : %d\n", description, value);
            }
        }
    }

    /**
     * Returns a short description of the given CUdevice_attribute constant
     * 
     * @param attribute The CUdevice_attribute constant
     * @return A short description of the given constant
     */
    private static String getAttributeDescription(int attribute)
    {
        switch (attribute)
        {
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: 
                return "Maximum number of threads per block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: 
                return "Maximum x-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: 
                return "Maximum y-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: 
                return "Maximum z-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: 
                return "Maximum x-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: 
                return "Maximum y-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: 
                return "Maximum z-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: 
                return "Maximum shared memory per thread block in bytes";
            case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: 
                return "Total constant memory on the device in bytes";
            case CU_DEVICE_ATTRIBUTE_WARP_SIZE: 
                return "Warp size in threads";
            case CU_DEVICE_ATTRIBUTE_MAX_PITCH: 
                return "Maximum pitch in bytes allowed for memory copies";
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: 
                return "Maximum number of 32-bit registers per thread block";
            case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: 
                return "Clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: 
                return "Alignment requirement";
            case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: 
                return "Number of multiprocessors on the device";
            case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: 
                return "Whether there is a run time limit on kernels";
            case CU_DEVICE_ATTRIBUTE_INTEGRATED: 
                return "Device is integrated with host memory";
            case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: 
                return "Device can map host memory into CUDA address space";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: 
                return "Compute mode";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: 
                return "Maximum 1D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: 
                return "Maximum 2D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: 
                return "Maximum 2D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: 
                return "Maximum 3D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: 
                return "Maximum 3D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: 
                return "Maximum 3D texture depth";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH: 
                return "Maximum 2D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT: 
                return "Maximum 2D layered texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS: 
                return "Maximum layers in a 2D layered texture";
            case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT: 
                return "Alignment requirement for surfaces";
            case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: 
                return "Device can execute multiple kernels concurrently";
            case CU_DEVICE_ATTRIBUTE_ECC_ENABLED: 
                return "Device has ECC support enabled";
            case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: 
                return "PCI bus ID of the device";
            case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: 
                return "PCI device ID of the device";
            case CU_DEVICE_ATTRIBUTE_TCC_DRIVER: 
                return "Device is using TCC driver model";
            case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: 
                return "Peak memory clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: 
                return "Global memory bus width in bits";
            case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: 
                return "Size of L2 cache in bytes";
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: 
                return "Maximum resident threads per multiprocessor";
            case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: 
                return "Number of asynchronous engines";
            case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: 
                return "Device shares a unified address space with the host";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH: 
                return "Maximum 1D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS: 
                return "Maximum layers in a 1D layered texture";
            case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: 
                return "PCI domain ID of the device";
            case CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT:
                return "Pitch alignment requirement for textures";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH:
                return "Maximum cubemap texture width/height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH:
                return "Maximum cubemap layered texture width/height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS:
                return "Maximum layers in a cubemap layered texture";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH:
                return "Maximum 1D surface width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH:
                return "Maximum 2D surface width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT:
                return "Maximum 2D surface height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH:
                return "Maximum 3D surface width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT:
                return "Maximum 3D surface height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH:
                return "Maximum 3D surface depth";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH:
                return "Maximum 1D layered surface width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS:
                return "Maximum layers in a 1D layered surface";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH:
                return "Maximum 2D layered surface width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT:
                return "Maximum 2D layered surface height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS:
                return "Maximum layers in a 2D layered surface";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH:
                return "Maximum cubemap surface width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH:
                return "Maximum cubemap layered surface width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS:
                return "Maximum layers in a cubemap layered surface";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH:
                return "Maximum 1D linear texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH:
                return "Maximum 2D linear texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT:
                return "Maximum 2D linear texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH:
                return "Maximum 2D linear texture pitch in bytes";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH:
                return "Maximum mipmapped 2D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT:
                return "Maximum mipmapped 2D texture height";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
                return "Major compute capability version number";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
                return "Minor compute capability version number";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH:
                return "Maximum mipmapped 1D texture width";
            case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED:
                return "Device supports stream priorities";
            case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED:
                return "Device supports caching globals in L1";
            case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED:
                return "Device supports caching locals in L1";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
                return "Maximum shared memory per multiprocessor in bytes";
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
                return "Maximum number of 32-bit registers per multiprocessor";
            case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
                return "Device can allocate managed memory on this system";
            case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD:
                return "Device is on a multi-GPU board";
            case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID:
                return "Unique id for a group of devices on the " + 
                "same multi-GPU board";
            case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED:
                return "Link between the device and the host supports " + 
                "native atomic operations";
            case CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO:
                return "Ratio of single precision to double " + 
                "precision performance ";
            case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:
                return "Device supports pageable memory without " + 
                "calling cudaHostRegister";
            case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
                return "Device can coherently access managed memory " + 
                "concurrently with the CPU ";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED:
                return "Device supports compute preemption. ";
        }
        return "(UNKNOWN ATTRIBUTE)";
    }

    /**
     * Returns a list of all CUdevice_attribute constants
     * 
     * @return A list of all CUdevice_attribute constants
     */
    private static List<Integer> getAttributes()
    {
        // Could just add integers 1 ... 90 here, but let's have them named: 
        List<Integer> list = new ArrayList<Integer>();
        list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_PITCH);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
        list.add(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        list.add(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT);
        list.add(CU_DEVICE_ATTRIBUTE_INTEGRATED);
        list.add(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS);
        list.add(CU_DEVICE_ATTRIBUTE_ECC_ENABLED);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
        list.add(CU_DEVICE_ATTRIBUTE_TCC_DRIVER);
        list.add(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
        list.add(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        list.add(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
        list.add(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
        list.add(CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        list.add(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED);
        list.add(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED);
        list.add(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
        list.add(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD);
        list.add(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID);
        list.add(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED);
        list.add(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO);
        list.add(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS);
        list.add(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS);
        list.add(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED);
        return list;
    }

    /**
     * Creates a String from a zero-terminated string in a byte array
     * 
     * @param bytes The byte array
     * @return The String
     */
    private static String createString(byte bytes[])
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bytes.length; i++)
        {
            char c = (char)bytes[i];
            if (c == 0)
            {
                break;
            }
            sb.append(c);
        }
        return sb.toString();
    }

}