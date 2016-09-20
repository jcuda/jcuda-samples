/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2016 Marco Hutter - http://www.jcuda.org
 */

package jcuda.nvrtc.samples;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.nvrtc.JNvrtc.nvrtcAddNameExpression;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetLoweredName;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;

import java.util.Arrays;
import java.util.List;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

/**
 * An example showing how to obtain the mangled names from kernels that
 * are compiled with the NVRTC at runtime
 */
public class JNvrtcLoweredNames
{
    /**
     * The source code of the program that contains different global
     * functions and function templates. 
     * (Taken from the NVIDIA NVRTC User Guide)
     */
    private static String programSourceCode = 
        "static __global__ void f1(int *result) { *result = 10; }" + "\n" + 
        "namespace N1 {" + "\n" + 
        "    namespace N2 {" + "\n" + 
        "        __global__ void f2(int *result) { *result = 20; }" + "\n" + 
        "    }" + "\n" + 
        "}" + "\n" + 
        "template<typename T>" + "\n" + 
        "__global__ void f3(int *result) { *result = sizeof(T); }" + "\n";
    
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Use the NVRTC to create a program
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, programSourceCode, null, 0, null, null);

        // Add the name expressions that refer to the global functions
        // and template instantiations
        List<String> functionNameExpressions = Arrays.asList(
            "&f1", 
            "N1::N2::f2", 
            "f3<int>", 
            "f3<double>"
        );
        for (String functionNameExpression : functionNameExpressions)
        {
            nvrtcAddNameExpression(program, functionNameExpression);
        }
        List<Integer> expectedResults = Arrays.asList(10, 20, 4, 8);

        // Compile the program
        nvrtcCompileProgram(program, 0, null);

        // Print the compilation log (for the case there are any warnings)
        String programLog[] = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Program compilation log:\n" + programLog[0]);

        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);

        // Create a CUDA module from the PTX code
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Allocate the output memory on the device
        CUdeviceptr dResult = new CUdeviceptr();
        cuMemAlloc(dResult, Sizeof.INT);

        // For each function name expression, obtain the lowered (mangled)
        // function name and print it
        boolean passed = true;
        for (int i = 0; i < functionNameExpressions.size(); i++)
        {
            // Obtain the lowered name. Note that this must be called
            // BEFORE the program is destroyed!
            String functionNameExpression = functionNameExpressions.get(i);
            String loweredName[] = { null };
            nvrtcGetLoweredName(program, functionNameExpression, loweredName);

            System.out.println(
                "Lowered name for " + functionNameExpression
                + " is " + loweredName[0]);

            // Obtain the function pointer to the function from the module,
            // using the lowered name
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, loweredName[0]);

            // Call the kernel function
            Pointer kernelParameters = Pointer.to(Pointer.to(dResult));
            cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, null,
                kernelParameters, null);
            cuCtxSynchronize();

            // Copy the result back to the host, and verify it
            int hResult[] = { 0 };
            cuMemcpyDtoH(Pointer.to(hResult), dResult, Sizeof.INT);

            System.out.println("Result: " + hResult[0]);

            int expectedResult = expectedResults.get(i);
            passed &= (expectedResult == hResult[0]);
        }

        System.out.println("Test " + (passed ? "PASSED" : "FAILED"));

        // Clean up.
        nvrtcDestroyProgram(program);
        cuMemFree(dResult);

    }
}
