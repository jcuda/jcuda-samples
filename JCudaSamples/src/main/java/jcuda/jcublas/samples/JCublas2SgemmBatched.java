/*
 * JCublas - Java bindings for JCublas, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * 
 * http://www.jcuda.org
 *
 * DISCLAIMER: THIS SOFTWARE IS PROVIDED WITHOUT WARRANTY OF ANY KIND
 * If you find any bugs or errors, contact me at http://www.jcuda.org
 *
 * LICENSE: THIS SOFTWARE IS FREE FOR NON-COMMERCIAL USE ONLY
 * For non-commercial applications, you may use this software without
 * any restrictions. If you wish to use it for commercial purposes,
 * contact me at http://www.jcuda.org
 */
package jcuda.jcublas.samples;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasSgemmBatched;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.samples.utils.JCudaSamplesUtils;

/**
 * This is a sample class demonstrating the application of JCublas2 for
 * performing a batched BLAS 'sgemm' operation, i.e. for computing the 
 * multiple matrices <br>
 * <code>C = alpha * A * B + beta * C</code> <br>
 * for single-precision floating point values alpha and beta, and matrices 
 * A, B and C
 */
class JCublas2SgemmBatched
{
    public static void main(String[] args)
    {
        JCublas2.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);
        testSgemmBatched(10, 100);
    }

    public static boolean testSgemmBatched(int b, int n)
    {
        System.out.println("Testing Sgemm with " + b + " batches of size " + n);

        float alpha = 0.3f;
        float beta = 0.7f;
        int nn = n * n;

        float h_A[][] = new float[b][];
        float h_B[][] = new float[b][];
        float h_C[][] = new float[b][];
        float h_C_ref[][] = new float[b][];
        for (int i = 0; i < b; i++)
        {
            h_A[i] = JCudaSamplesUtils.createRandomFloatData(nn);
            h_B[i] = JCudaSamplesUtils.createRandomFloatData(nn);
            h_C[i] = JCudaSamplesUtils.createRandomFloatData(nn);
            h_C_ref[i] = h_C[i].clone();
        }

        System.out.println("Performing Sgemm with Java...");
        sgemmJava(n, alpha, h_A, h_B, beta, h_C_ref);

        System.out.println("Performing Sgemm with JCublas2...");
        sgemmBatchedJCublas2(n, alpha, h_A, h_B, beta, h_C);

        // Print the test results
        boolean passed = true;
        for (int i = 0; i < b; i++)
        {
            passed &= JCudaSamplesUtils.equalByNorm(h_C[i], h_C_ref[i]);
        }
        System.out.println(String.format("testSgemm %s", 
            passed ? "PASSED" : "FAILED"));
        return passed;
    }

    static void sgemmBatchedJCublas2(int n, float alpha, 
        float h_A[][], float h_B[][], float beta, float h_C[][])
    {
        int nn = n * n;
        int b = h_A.length;
        Pointer[] h_Aarray = new Pointer[b];
        Pointer[] h_Barray = new Pointer[b];
        Pointer[] h_Carray = new Pointer[b];
        for (int i = 0; i < b; i++)
        {
            h_Aarray[i] = new Pointer();
            h_Barray[i] = new Pointer();
            h_Carray[i] = new Pointer();
            cudaMalloc(h_Aarray[i], nn * Sizeof.FLOAT);
            cudaMalloc(h_Barray[i], nn * Sizeof.FLOAT);
            cudaMalloc(h_Carray[i], nn * Sizeof.FLOAT);
            cudaMemcpy(h_Aarray[i], Pointer.to(h_A[i]),
                nn * Sizeof.FLOAT, cudaMemcpyHostToDevice);
            cudaMemcpy(h_Barray[i], Pointer.to(h_B[i]),
                nn * Sizeof.FLOAT, cudaMemcpyHostToDevice);
            cudaMemcpy(h_Carray[i], Pointer.to(h_C[i]),
                nn * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        }
        Pointer d_Aarray = new Pointer();
        Pointer d_Barray = new Pointer();
        Pointer d_Carray = new Pointer();
        cudaMalloc(d_Aarray, b * Sizeof.POINTER);
        cudaMalloc(d_Barray, b * Sizeof.POINTER);
        cudaMalloc(d_Carray, b * Sizeof.POINTER);
        cudaMemcpy(d_Aarray, Pointer.to(h_Aarray),
            b * Sizeof.POINTER, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Barray, Pointer.to(h_Barray), 
            b * Sizeof.POINTER, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Carray, Pointer.to(h_Carray), 
            b * Sizeof.POINTER, cudaMemcpyHostToDevice);
        
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        cublasSgemmBatched(
            handle, 
            cublasOperation.CUBLAS_OP_N, 
            cublasOperation.CUBLAS_OP_N, 
            n, n, n, 
            Pointer.to(new float[]{ alpha }),            
            d_Aarray, n, d_Barray, n, 
            Pointer.to(new float[]{ beta }), 
            d_Carray, n, b);

        for (int i = 0; i < b; i++)
        {
            cudaMemcpy(Pointer.to(h_C[i]), h_Carray[i], 
                nn * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaFree(h_Aarray[i]);
            cudaFree(h_Barray[i]);
            cudaFree(h_Carray[i]);
        }
        cudaFree(d_Aarray);
        cudaFree(d_Barray);
        cudaFree(d_Carray);
        cublasDestroy(handle);
        
    }

    static void sgemmJava(int n, float alpha, 
        float A[][], float B[][], float beta, float C[][])
    {
        for (int i = 0; i < A.length; i++)
        {
            sgemmJava(n, alpha, A[i], B[i], beta, C[i]);
        }
    }
    
    static void sgemmJava(int n, float alpha, 
        float A[], float B[], float beta, float C[])
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float prod = 0;
                for (int k = 0; k < n; ++k)
                {
                    prod += A[k * n + i] * B[j * n + k];
                }
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            }
        }
    }
}
