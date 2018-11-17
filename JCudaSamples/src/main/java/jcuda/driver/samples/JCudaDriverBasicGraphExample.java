/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2018 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.samples;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuGraphAddKernelNode;
import static jcuda.driver.JCudaDriver.cuGraphCreate;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_KERNEL_NODE_PARAMS;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraph;
import jcuda.driver.CUgraphExec;
import jcuda.driver.CUgraphNode;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

/**
 * A basic example showing how to use the CUDA execution graph API with JCuda.
 * <br>
 * <br>
 * <b>Note:</b> As of JCuda 10.0.0, there is only limited support for the
 * execution graph API. Not all functionalities of this API may sensibly
 * be mapped to Java. The following is a very basic example, but more
 * complex use cases have not yet been tested thoroughly.
 */
public class JCudaDriverBasicGraphExample
{
    /**
     * Create and execute an execution graph with the following kernel 
     * structure:
     * 
     * <pre><code>
     *       A
     *     /   \
     *    B     C
     *     \   /
     *       D
     *       
     * Node A 
     *   will receive two arrays 
     *     a=[1,1,1...]
     *     b=[1,1,1...]
     *   and add 2 to each element:    
     *     a=[3,3,3...]
     *     b=[3,3,3...]
     * 
     * Node B 
     *   will receive array a=[3,3,3...] and multiply it with 3:
     *       a=[9,9,9,...] 
     *             
     * Node C 
     *   will receive array b=[3,3,3...] and multiply it with 4:
     *       b=[12,12,12,...] 
     *             
     * Node D 
     *   will receive two arrays 
     *     a=[9,9,9...]
     *     b=[12,12,12...]
     *   and add them together:    
     *     sum=[21,21,21...]
     *             
     * </code></pre>
     *            
     * @param args Not used
     */
    public static void main(String[] args)
    {
        System.out.println("Initializing");
        initialize();
        
        System.out.println("Creating functions");
        CUfunction addScalar = createFunction("add", addScalarCode);
        CUfunction mulScalar = createFunction("mul", mulScalarCode);
        CUfunction add = createFunction("add", addCode);
        
        System.out.println("Creating input- and output data");
        int numElements = 256 * 10;
        CUdeviceptr deviceInput0 = createDeviceData(numElements, 1.0f);
        CUdeviceptr deviceInput1 = createDeviceData(numElements, 1.0f);
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, numElements * Sizeof.FLOAT);

        System.out.println("Creating graph");
        CUgraph graph = new CUgraph();
        cuGraphCreate(graph, 0);

        System.out.println("Creating graph node A");
        CUgraphNode a = new CUgraphNode();
        CUDA_KERNEL_NODE_PARAMS nodeParamsA = new CUDA_KERNEL_NODE_PARAMS();
        setDims(nodeParamsA, numElements);
        nodeParamsA.func = addScalar;
        nodeParamsA.kernelParams = createKernelParams(
            numElements, deviceInput0, deviceInput1, 2.0f);

        System.out.println("Creating graph node B");
        CUgraphNode b = new CUgraphNode();
        CUDA_KERNEL_NODE_PARAMS nodeParamsB = new CUDA_KERNEL_NODE_PARAMS();
        setDims(nodeParamsB, numElements);
        nodeParamsB.func = mulScalar;
        nodeParamsB.kernelParams = createKernelParams(
            numElements, deviceInput0, 3.0f);
        
        System.out.println("Creating graph node C");
        CUgraphNode c = new CUgraphNode();
        CUDA_KERNEL_NODE_PARAMS nodeParamsC = new CUDA_KERNEL_NODE_PARAMS();
        setDims(nodeParamsC, numElements);
        nodeParamsC.func = mulScalar;
        nodeParamsC.kernelParams = createKernelParams(
            numElements, deviceInput1, 4.0f);
        
        System.out.println("Creating graph node D");
        CUgraphNode d = new CUgraphNode();
        CUDA_KERNEL_NODE_PARAMS nodeParamsD = new CUDA_KERNEL_NODE_PARAMS();
        setDims(nodeParamsD, numElements);
        nodeParamsD.func = add;
        nodeParamsD.kernelParams = createKernelParams(
            numElements, deviceInput0, deviceInput1, deviceOutput);
        
        System.out.println("Adding nodes to graph");
        cuGraphAddKernelNode(a, graph, null, 0, nodeParamsA);
        cuGraphAddKernelNode(b, graph, null, 0, nodeParamsB);
        cuGraphAddKernelNode(c, graph, null, 0, nodeParamsC);
        cuGraphAddKernelNode(d, graph, null, 0, nodeParamsD);

        System.out.println("Defining dependencies");
        CUgraphNode aa[] = { a };
        CUgraphNode bb[] = { b };
        CUgraphNode cc[] = { c };
        CUgraphNode dd[] = { d };
        cuGraphAddDependencies(graph, aa, bb, 1); // A->B
        cuGraphAddDependencies(graph, aa, cc, 1); // A->C
        cuGraphAddDependencies(graph, bb, dd, 1); // B->D
        cuGraphAddDependencies(graph, cc, dd, 1); // C->D
        
        System.out.println("Creating graph execution");
        CUgraphExec graphExec = new CUgraphExec();
        CUgraphNode errorNode = new CUgraphNode();
        int bufferSize = 10000;
        byte logBuffer[] = new byte[bufferSize]; 
        cuGraphInstantiate(graphExec, graph, errorNode, logBuffer, bufferSize);
        
        System.out.println("Launching");
        cuGraphLaunch(graphExec, CU_STREAM_LEGACY);
        
        cuCtxSynchronize();
        
        System.out.println("Obtaining result data");
        float hostOutput[] = new float[numElements];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
            numElements * Sizeof.FLOAT);
        String s = Arrays.toString(hostOutput);
        System.out.println("Result " 
            + s.substring(0, Math.min(100, s.length())) + "...");
        
        // NOTE: Here, many cleanup operations would 
        // be necessary, but are omitted for brevity!
    }
    
    // Source code of a kernel that adds a scalar value to two vectors, in place
    private static String addScalarCode = 
        "extern \"C\"" + "\n" +
        "__global__ void add(int n, float *a, float *b, float x)" + "\n" +
        "{" + "\n" +
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
        "    if (i<n)" + "\n" +
        "    {" + "\n" +
        "        a[i] += x;" + "\n" +
        "        b[i] += x;" + "\n" +
        "    }" + "\n" +
        "}" + "\n";
    
    // Source code for a kernel that multiplies a vector with a scalar, in place
    private static String mulScalarCode = 
        "extern \"C\"" + "\n" +
        "__global__ void mul(int n, float *a, float x)" + "\n" +
        "{" + "\n" +
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
        "    if (i<n)" + "\n" +
        "    {" + "\n" +
        "        a[i] *= x;" + "\n" +
        "    }" + "\n" +
        "}" + "\n";

    // Source code for a kernel that adds two vectors
    private static String addCode = 
        "extern \"C\"" + "\n" +
        "__global__ void add(int n, float *a, float *b, float *sum)" + "\n" +
        "{" + "\n" +
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
        "    if (i<n)" + "\n" +
        "    {" + "\n" +
        "        sum[i] = a[i] + b[i];" + "\n" +
        "    }" + "\n" +
        "}" + "\n";
    
    /**
     * Perform a default initialization of CUDA, creating a context
     * for the first device
     */
    private static void initialize()
    {
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
    }
    
    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     * 
     * @param name The name of the function
     * @param code The source code
     * @return The CUDA function
     */
    private static CUfunction createFunction(String name, String code)
    {
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, code, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
        
        String programLog[] = new String[1];
        nvrtcGetProgramLog(program, programLog);
        String log = programLog[0].trim();
        if (!log.isEmpty())
        {
            System.err.println("Compilation log for " + name + ":\n" + log);
        }
        
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, name);
        
        return function;
    }
    
    /**
     * Create device data containing the given float value, the given number
     * of times
     * 
     * @param numElements The number of elements
     * @param value The value of the elements
     * @return The pointer to the data
     */
    private static CUdeviceptr createDeviceData(int numElements, float value)
    {
        float hostData[] = new float[numElements];
        for (int i = 0; i < numElements; i++)
        {
            hostData[i] = value;
        }
        CUdeviceptr deviceData = new CUdeviceptr();
        cuMemAlloc(deviceData, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceData, Pointer.to(hostData),
            numElements * Sizeof.FLOAT);
        return deviceData;
    }
    
    /**
     * Set the grid- and block size parameters for launching a 1D-kernel 
     * with the given number of elements
     * 
     * @param params The parameters
     * @param numElements The number of elements
     */
    private static void setDims(CUDA_KERNEL_NODE_PARAMS params, int numElements)
    {
        params.blockDimX = 256;
        params.blockDimY = 1;
        params.blockDimZ = 1;
        params.gridDimX = 
            (numElements + params.blockDimX - 1) / params.blockDimX;
        params.gridDimY = 1;
        params.gridDimZ = 1;
    }
    
    /**
     * Create a pointer to the given kernel parameters. Note that only 
     * a limited subset of argument types is supported here.
     * 
     * @param args The kernel parameters
     * @return The pointer with the kernel parameters
     */
    private static Pointer createKernelParams(Object ... args)
    {
        Pointer kernelParameters[] = new Pointer[args.length];
        for (int i=0; i<args.length; i++)
        {
            Object arg = args[i];
            if (arg instanceof Pointer)
            {
                Pointer argPointer = (Pointer)arg;
                Pointer pointer = Pointer.to(argPointer);
                kernelParameters[i] = pointer;
            }
            else if (arg instanceof Integer)
            {
                Integer value = (Integer)arg;
                Pointer pointer = Pointer.to(new int[]{value});
                kernelParameters[i] = pointer;
            }
            else if (arg instanceof Float)
            {
                Float value = (Float)arg;
                Pointer pointer = Pointer.to(new float[]{value});
                kernelParameters[i] = pointer;
            }
            else
            {
                System.out.println("Type not supported: " + arg.getClass());
            }
        }
        return Pointer.to(kernelParameters);
    }
    
    
    
}
