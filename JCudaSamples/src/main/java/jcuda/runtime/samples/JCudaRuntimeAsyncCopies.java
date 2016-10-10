package jcuda.runtime.samples;
/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaFreeHost;
import static jcuda.runtime.JCuda.cudaHostAlloc;
import static jcuda.runtime.JCuda.cudaHostAllocWriteCombined;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaMemcpyAsync;
import static jcuda.runtime.JCuda.cudaStreamCreate;
import static jcuda.runtime.JCuda.cudaStreamSynchronize;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToHost;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Locale;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;

/**
 * A sample that shows the different configurations of synchronous and
 * asynchronous memory copy operations, and prints timing information.
 */
public class JCudaRuntimeAsyncCopies
{
    /**
     * Memory types
     */
    enum MemoryType
    {
        /**
         * Device memory
         */
        DEVICE,

        /**
         * Pinned host memory, allocated with cudaHostAlloc
         */
        HOST_PINNED,

        /**
         * Pageable memory in form of a Pointer.to(array)
         */
        HOST_PAGEABLE_ARRAY,

        /**
         * Pageable memory in form of a Pointer.to(directBuffer)
         */
        HOST_PAGEABLE_DIRECT,
    }

    /**
     * Simple wrapper class for a block of allocated memory with a certain
     * {@link MemoryType}. 
     * This class is solely intended for testing purposes.
     */
    static class Memory
    {
        /**
         * The {@link MemoryType} of this memory
         */
        private final MemoryType type;

        /**
         * The pointer to the actual memory
         */
        private final Pointer pointer;

        /**
         * The buffer for the memory, if it is no device memory
         */
        private final FloatBuffer buffer;

        /**
         * Creates a block of memory with the given type and size
         * 
         * @param type The {@link MemoryType}
         * @param numBytes The size of the memory, in bytes
         */
        Memory(MemoryType type, int numBytes)
        {
            this.type = type;
            switch (type)
            {
                case DEVICE:
                {
                    // Allocate device memory
                    pointer = new Pointer();
                    buffer = null;
                    cudaMalloc(pointer, numBytes);
                    break;
                }
                case HOST_PINNED:
                {
                    // Allocate pinned (page-locked) host memory
                    pointer = new Pointer();
                    cudaHostAlloc(pointer, numBytes, 
                        cudaHostAllocWriteCombined);
                    ByteBuffer byteBuffer = pointer.getByteBuffer(0, numBytes);
                    byteBuffer.order(ByteOrder.nativeOrder());
                    buffer = byteBuffer.asFloatBuffer();
                    break;
                }
                case HOST_PAGEABLE_ARRAY:
                {
                    // The host memory is pageable and stored in a Java array
                    byte array[] = new byte[numBytes];
                    ByteBuffer byteBuffer = ByteBuffer.wrap(array);
                    byteBuffer.order(ByteOrder.nativeOrder());
                    buffer = byteBuffer.asFloatBuffer();
                    pointer = Pointer.to(array);
                    break;
                }
                default:
                case HOST_PAGEABLE_DIRECT:
                {
                    // The host memory is pageable and stored in a direct
                    // byte buffer
                    ByteBuffer byteBuffer = 
                        ByteBuffer.allocateDirect(numBytes);
                    byteBuffer.order(ByteOrder.nativeOrder());
                    buffer = byteBuffer.asFloatBuffer();
                    pointer = Pointer.to(buffer);
                }
            }
        }

        /**
         * Put the data from the given source array into this memory
         * 
         * @param source The source array
         */
        void put(float source[])
        {
            if (type == MemoryType.DEVICE)
            {
                cudaMemcpy(pointer, Pointer.to(source), 
                    source.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);
            }
            else
            {
                buffer.put(source);
                buffer.rewind();
            }
        }

        /**
         * Write data from this memory into the given target array
         * 
         * @param target The target array
         */
        void get(float target[])
        {
            if (type == MemoryType.DEVICE)
            {
                cudaMemcpy(Pointer.to(target), pointer, 
                    target.length * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            }
            else
            {
                buffer.get(target);
                buffer.rewind();
            }
        }

        /**
         * Returns the pointer to this memory
         * 
         * @return The pointer
         */
        Pointer getPointer()
        {
            return pointer;
        }

        /**
         * Release this memory
         */
        void release()
        {
            if (type == MemoryType.DEVICE)
            {
                cudaFree(pointer);
            }
            else if (type == MemoryType.HOST_PINNED)
            {
                cudaFreeHost(pointer);
            }
        }
    }
    
    /**
     * The number of float values for one chunk of data that
     * is copied during the tests
     */
    private static final int NUM_FLOATS = (1 << 22);
    
    /**
     * How many copy operations are repeated for the timing
     */
    private static final int COPY_RUNS = 10;


    /**
     * Entry point of this sample
     * 
     * @param args  Not used
     */
    public static void main(String args[])
    {
        // Enable exceptions and omit all subsequent error checks
        JCuda.setExceptionsEnabled(true);

        // Create the host input data
        float data[] = new float[NUM_FLOATS];
        for (int i = 0; i < NUM_FLOATS; i++)
        {
            data[i] = i;
        }

        // Run tests for all combinations of target- and
        // source memory types:
        System.out.println(
            "Timing " + COPY_RUNS + " copy operations " + 
            "of " + NUM_FLOATS + " float values");

        System.out.println("Synchronous memory copies");
        System.out.printf("%22s    %22s   %10s    %10s     %s\n", 
            "TARGET", "SOURCE", "", "TOTAL", "PASSED");
        System.out.println(String.format("%90s", " ").replace(' ', '-'));
        
        boolean passed = true;
        
        for (MemoryType targetType : MemoryType.values())
        {
            for (MemoryType sourceType : MemoryType.values())
            {
                passed &= testSync(targetType, sourceType, data);
            }
        }

        System.out.println();
        System.out.println("Asynchronous memory copies");
        System.out.printf("%22s    %22s   %10s    %10s     %s\n", 
            "TARGET", "SOURCE", "CALL", "WAIT", "PASSED");
        System.out.println(String.format("%90s", " ").replace(' ', '-'));

        for (MemoryType targetType : MemoryType.values())
        {
            for (MemoryType sourceType : MemoryType.values())
            {
                passed &= testAsync(targetType, sourceType, data);
            }
        }
        
        System.out.println("DONE, result: " + (passed ? "PASSED" : "FAILED"));
    }

    /**
     * Test a synchronous (blocking) copy of the given data between the 
     * given memory types
     *  
     * @param targetType The target {@link MemoryType}
     * @param sourceType The source {@link MemoryType}
     * @param data The data
     * @return Whether the test passed
     */
    private static boolean testSync(
        MemoryType targetType, MemoryType sourceType, float data[])
    {
        // Allocate source- and target memory, and fill the source
        // memory with the given data
        int numBytes = data.length * Sizeof.FLOAT;
        int kind = getCudaMemcpyKind(targetType, sourceType);
        Memory target = new Memory(targetType, numBytes);
        Memory source = new Memory(sourceType, numBytes);
        source.put(data);
        Pointer t = target.getPointer();
        Pointer s = source.getPointer();

        // Perform the copying operations
        long before = System.nanoTime();
        for (int i = 0; i < COPY_RUNS; i++)
        {
            cudaMemcpy(t, s, numBytes, kind);
        }
        long after = System.nanoTime();
        double durationCopyMS = (after - before) / 1e6;

        // Verify the result and clean up
        boolean passed = verify(target, data);
        target.release();
        source.release();

        // Print the timing information
        String dcs = String.format(Locale.ENGLISH, "%10.3f", durationCopyMS);
        System.out.printf("%22s <- %22s : %10s ms %10s ms  %s\n", 
            targetType, sourceType, "", dcs, passed);
        
        return passed;
    }

    /**
     * Test an a synchronous (non-blocking) copy of the given data between the 
     * given memory types
     *  
     * @param targetType The target {@link MemoryType}
     * @param sourceType The source {@link MemoryType}
     * @param data The data
     * @return Whether the test passed
     */
    private static boolean testAsync(
        MemoryType targetType, MemoryType sourceType, float data[])
    {
        // Allocate source- and target memory, and fill the source
        // memory with the given data
        int numBytes = data.length * Sizeof.FLOAT;
        int kind = getCudaMemcpyKind(targetType, sourceType);
        Memory target = new Memory(targetType, numBytes);
        Memory source = new Memory(sourceType, numBytes);
        source.put(data);
        Pointer t = target.getPointer();
        Pointer s = source.getPointer();

        // Create a stream
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);

        // Issue the asynchronous copies on the stream
        long beforeCall = System.nanoTime();
        for (int i = 0; i < COPY_RUNS; i++)
        {
            cudaMemcpyAsync(t, s, numBytes, kind, stream);
        }
        long afterCall = System.nanoTime();
        double durationCallMS = (afterCall - beforeCall) / 1e6;

        // Wait for the stream to be finished
        long beforeWait = System.nanoTime();
        cudaStreamSynchronize(stream);
        long afterWait = System.nanoTime();
        double durationWaitMS = (afterWait - beforeWait) / 1e6;

        // Verify the result and clean up
        boolean passed = verify(target, data);
        target.release();
        source.release();

        // Print the timing information
        String dcs = String.format(Locale.ENGLISH, "%10.3f", durationCallMS);
        String dws = String.format(Locale.ENGLISH, "%10.3f", durationWaitMS);
        System.out.printf("%22s <- %22s : %10s ms %10s ms  %s\n",
            targetType, sourceType, dcs, dws, passed);
        
        return passed;
    }

    /**
     * Returns the {@link cudaMemcpyKind} constant for the given target- and
     * source {@link MemoryType}
     * 
     * @param targetType The target memory type
     * @param sourceType The source memory type
     * @return The {@link cudaMemcpyKind} constant
     */
    private static int getCudaMemcpyKind(MemoryType targetType,
        MemoryType sourceType)
    {
        if (targetType == MemoryType.DEVICE)
        {
            if (sourceType == MemoryType.DEVICE)
            {
                return cudaMemcpyDeviceToDevice;
            }
            return cudaMemcpyHostToDevice;
        }
        if (sourceType == MemoryType.DEVICE)
        {
            return cudaMemcpyDeviceToHost;
        }
        return cudaMemcpyHostToHost;
    }

    /**
     * Verify that the data that is stored in the given memory is equal to the
     * data in the given array
     * 
     * @param target The memory
     * @param data The data that is expected in the memory
     * @return Whether the data was equal
     */
    private static boolean verify(Memory target, float data[])
    {
        float result[] = new float[data.length];
        target.get(result);
        boolean passed = true;
        for (int i = 0; i < data.length; i++)
        {
            float f0 = data[i];
            float f1 = result[i];
            if (f0 != f1)
            {
                System.out.println(
                    "ERROR: At index " + i + 
                    " expected " + f0 + 
                    " but found " + f1);
                passed = false;
                break;
            }
        }
        return passed;
    }
}
