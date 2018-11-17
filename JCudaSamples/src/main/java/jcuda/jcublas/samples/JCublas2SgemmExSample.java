/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcublas.samples;

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasGemmEx;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO0;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO2;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO4;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO5;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO6;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO7;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

import java.util.Arrays;
import java.util.List;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.samples.utils.JCudaSamplesUtils;

/**
 * This is a sample class demonstrating the application of JCublas2 for
 * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br>
 * <code>C = alpha * A * B + beta * C</code> <br>
 * for single-precision floating point values alpha and beta, and matrices 
 * A, B and C, using the extended CUBLAS GEMM function
 */
public class JCublas2SgemmExSample
{
    public static void main(String args[])
    {
        JCublas2.setExceptionsEnabled(true);
        testSgemm(2000);
    }
    
    // The list of CUBLAS GEMM algorithms to use. Note that the set of
    // supported algorithms will likely depend on the platform, the
    // size of the matrix, and other factors.
    private static final List<Integer> GEMM_ALGORITHMS = Arrays.asList(
        CUBLAS_GEMM_ALGO2,
        CUBLAS_GEMM_ALGO4,
        CUBLAS_GEMM_ALGO5,
        CUBLAS_GEMM_ALGO6,
        CUBLAS_GEMM_ALGO7
    );
    private static int GEMM_ALGO = CUBLAS_GEMM_ALGO0;

    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     * 
     * @param n The matrix size
     */
    public static void testSgemm(int n)
    {
        float alpha = 0.3f;
        float beta = 0.7f;
        int nn = n * n;

        System.out.println("Creating input data...");
        float h_A[] = JCudaSamplesUtils.createRandomFloatData(nn);
        float h_B[] = JCudaSamplesUtils.createRandomFloatData(nn);
        float h_C[] = JCudaSamplesUtils.createRandomFloatData(nn);

        System.out.println("Performing Sgemm with JCublas...");
        for (int i : GEMM_ALGORITHMS)
        {
            GEMM_ALGO = i;
            try
            {
                sgemmJCublas(n, alpha, h_A, h_B, beta, h_C);
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }

    }

    /**
     * Implementation of sgemm using JCublas
     */
    private static void sgemmJCublas(
        int n, float alpha, float A[], float B[], float beta, float C[])
    {
        int nn = n * n;

        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        cudaMalloc(d_A, nn * Sizeof.FLOAT);
        cudaMalloc(d_B, nn * Sizeof.FLOAT);
        cudaMalloc(d_C, nn * Sizeof.FLOAT);

        // Copy the memory from the host to the device
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(A), 1, d_A, 1);
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(B), 1, d_B, 1);
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(C), 1, d_C, 1);

        // Execute sgemm
        Pointer pAlpha = Pointer.to(new float[] { alpha });
        Pointer pBeta = Pointer.to(new float[] { beta });
        
        long before = System.nanoTime();
        
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 
            pAlpha, d_A, CUDA_R_32F, n, d_B, CUDA_R_32F, n, 
            pBeta, d_C, CUDA_R_32F, n, CUDA_R_32F, GEMM_ALGO);
        
        cudaDeviceSynchronize();
        
        long after = System.nanoTime();
        double durationMs = (after - before) / 1e6;
        System.out.println(
            "Algorithm " + GEMM_ALGO + " took " + durationMs + " ms");

        // Copy the result from the device to the host
        cublasGetVector(nn, Sizeof.FLOAT, d_C, 1, Pointer.to(C), 1);

        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

}