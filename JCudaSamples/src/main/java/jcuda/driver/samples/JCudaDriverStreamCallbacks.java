/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2017 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.samples;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemHostAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.driver.JCudaDriver.cuStreamAddCallback;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.CUstreamCallback;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

/**
 * An example showing stream callbacks involving multiple streams
 * and threads
 */
public class JCudaDriverStreamCallbacks
{
    /**
     * A kernel that increments all elements of an int array by 1
     */
    private static String programSourceCode = 
        "extern \"C\"" + "\n" +
        "__global__ void example(int n, int *data)" + "\n" +
        "{" + "\n" +
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
        "    if (i<n)" + "\n" +
        "    {" + "\n" +
        "        data[i]++;" + "\n" +
        "    }" + "\n" +
        "}" + "\n";
    
    /**
     * The CUDA context
     */
    private static CUcontext context;
    
    /**
     * The kernel function with the dummy workload
     */
    private static CUfunction function;
    
    /**
     * The size of the array in the workload class
     */
    private static final int WORKLOAD_SIZE = 100000;
    
    /**
     * The workload that is processed here: Host and device data, and
     * the stream on which the workload is processed. 
     */
    private static class Workload
    {
        int index;
        CUstream stream;
        Pointer hostData;
        CUdeviceptr deviceData;
    }

    /**
     * Create a Workload instance. This method is called by multiple host
     * threads, to create the individual workloads, and to send the 
     * commands for processing the workloads to CUDA
     * 
     * @param index The index of the workload 
     * @param executor The executor service 
     */
    private static void createWorkloadOnHost(
        final int index, final ExecutorService executor)
    {
        // Make sure that the CUDA context is current for the calling thread
        cuCtxSetCurrent(context);

        // Initialize the workload, and create the CUDA stream

        System.out.println(index + ": Initializing workload");
        final Workload workload = new Workload();
        workload.index = index;
        workload.stream = new CUstream();
        cuStreamCreate(workload.stream, 0);
        
        
        // Create the host data of the workload
        
        System.out.println(index + ": Create host data");
        workload.hostData = new Pointer();
        cuMemHostAlloc(workload.hostData, WORKLOAD_SIZE * Sizeof.INT, 0);
        ByteBuffer hostByteBuffer =
            workload.hostData.getByteBuffer(0, WORKLOAD_SIZE * Sizeof.INT);
        IntBuffer hostIntBuffer = 
            hostByteBuffer.order(ByteOrder.nativeOrder()).asIntBuffer();
        for (int i = 0; i < WORKLOAD_SIZE; i++)
        {
            hostIntBuffer.put(i, i);
        }
        workload.deviceData = new CUdeviceptr();
        cuMemAlloc(workload.deviceData, WORKLOAD_SIZE * Sizeof.INT);

        
        // Execute the CUDA commands:
        // - Copy the host data to the device
        // - Execute the kernel
        // - Copy the modified device data back to the host
        // All this is done asynchronously

        System.out.println(index + ": Execute CUDA commands");

        cuMemcpyHtoDAsync(workload.deviceData, workload.hostData,
            WORKLOAD_SIZE * Sizeof.INT, workload.stream);

        Pointer kernelParameters = Pointer.to(
            Pointer.to(new int[]{WORKLOAD_SIZE}),
            Pointer.to(workload.deviceData)
        );
        int blockSizeX = 256;
        int gridSizeX = (WORKLOAD_SIZE + blockSizeX - 1) / blockSizeX;
        cuLaunchKernel(function, gridSizeX,  1, 1, blockSizeX, 1, 1,
            0, workload.stream, kernelParameters, null);
        
        cuMemcpyDtoHAsync(workload.hostData, workload.deviceData,
            WORKLOAD_SIZE * Sizeof.INT, workload.stream);
        
        
        // Define the callback that will be called when all CUDA commands
        // on the stream have finished. This callback will forward the
        // workload to the "finishWorkloadOnHost" method.
        CUstreamCallback callback = new CUstreamCallback()
        {
            @Override
            public void call(
                CUstream hStream, int status, final Object userData)
            {
                System.out.println(index + ": Callback was called");
                Runnable runnable = new Runnable()
                {
                    @Override
                    public void run()
                    {
                        finishWorkloadOnHost(userData);
                    }
                };
                executor.submit(runnable);
            }
        };
        cuStreamAddCallback(workload.stream, callback, workload, 0);
    }
    
    
    /**
     * A method that will be called by a stream callback, and receive the
     * workload for which the CUDA commands have been finished
     * 
     * @param workloadObject The workload object
     */
    private static void finishWorkloadOnHost(Object workloadObject)
    {
        Workload workload = (Workload)workloadObject;
        int index = workload.index;
        
        // Finish the task, by comparing the host data with the expected values
        
        System.out.println(index + ": Finishing");
        
        boolean passed = true;
        ByteBuffer hostByteBuffer =
            workload.hostData.getByteBuffer(0, WORKLOAD_SIZE * Sizeof.INT);
        IntBuffer hostIntBuffer = 
            hostByteBuffer.order(ByteOrder.nativeOrder()).asIntBuffer();
        for (int i = 0; i < WORKLOAD_SIZE; i++)
        {
            passed &= (hostIntBuffer.get(i) == (i + 1));
        }

        System.out.println(index + ": " + (passed ? "PASSED" : "FAILED"));
    }
    
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        initialize();
        
        final ExecutorService executor = createExecutorService(4);

        // Create tasks to create Workload objects, and pass them to
        // the executor service. Each task will initialize its 
        // workload on the host and pass the workload to CUDA.
        // When the workload is done, the tasks to finish the
        // workloads are created and passed to the executor service
        int numWorkloads = 8;
        for (int n=0; n<numWorkloads; n++)
        {
            final int index = n;
            Runnable runnable = new Runnable()
            {
                @Override
                public void run()
                {
                    createWorkloadOnHost(index, executor);
                }
            };
            executor.submit(runnable);
        }
     
        // Shut down the executor service
        try
        {
            executor.awaitTermination(10,  TimeUnit.SECONDS);
            executor.shutdown();
        }
        catch (InterruptedException e)
        {
            e.printStackTrace();
        }
        System.out.println("Done");
        
    }
    

    /**
     * Initialize the driver API, the {@link #context} and the 
     * kernel {@link #function} 
     */
    private static void initialize()
    {
        System.out.println("Initializing...");
        
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);

        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(
            program, programSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
        
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        function = new CUfunction();
        cuModuleGetFunction(function, module, "example");
        
        System.out.println("Initializing DONE");
    }
    
    /**
     * Create an executor service with the given fixed pool size, whose
     * core threads time out after a short time, and which re-throws
     * all exceptions that happen in the tasks that it processes.
     * 
     * @param poolSize The pool size
     * @return The executor service
     */
    private static ExecutorService createExecutorService(int poolSize)
    {
        ThreadPoolExecutor e = 
            new ThreadPoolExecutor(poolSize, poolSize,
                5, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>())
        {
            @Override
            protected void afterExecute(Runnable r, Throwable t)
            {
                super.afterExecute(r, t);
                if (t == null && r instanceof Future<?>)
                {
                    try
                    {
                        Future<?> future = (Future<?>) r;
                        if (future.isDone())
                        {
                            future.get();
                        }
                    }
                    catch (CancellationException ce)
                    {
                        t = ce;
                    }
                    catch (ExecutionException ee)
                    {
                        t = ee.getCause();
                    }
                    catch (InterruptedException ie)
                    {
                        Thread.currentThread().interrupt();
                    }
                }
                if (t != null)
                {
                    throw new RuntimeException(t);
                }
            }
        };
        e.allowCoreThreadTimeOut(true);
        return e;
    }
    
}
