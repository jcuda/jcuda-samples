/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcublas.samples;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasGetMatrix;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasIsamax;
import static jcuda.jcublas.JCublas2.cublasSetMatrix;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.JCublas2.cublasSgemm;
import static jcuda.jcublas.JCublas2.cublasSgemv;
import static jcuda.jcublas.JCublas2.cublasSger;
import static jcuda.jcublas.JCublas2.cublasSscal;
import static jcuda.jcublas.JCublas2.cublasSswap;
import static jcuda.jcublas.JCublas2.cublasStrmv;
import static jcuda.jcublas.cublasFillMode.CUBLAS_FILL_MODE_UPPER;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import jcuda.samples.utils.JCudaSamplesUtils;

/**
 * Example of a matrix inversion using JCublas2.
 */
public class JCublas2MatrixInvert
{
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        // Create the input matrix
        int size = 7;
        float A[] = JCudaSamplesUtils.createRandomFloatData(size * size);

        // Invert the matrix
        float invA[] = A.clone();
        invertMatrix(handle, size, invA);

        // Compute A*invA, which should yield the identity matrix
        float identity[] = new float[size * size];
        multiply(handle, size, A, invA, identity);

        // Print the results
        System.out.println("A:");
        System.out.println(JCudaSamplesUtils.toString2D(A, size));
        System.out.println("invA:");
        System.out.println(JCudaSamplesUtils.toString2D(invA, size));
        System.out.println("identity:");
        System.out.println(JCudaSamplesUtils.toString2D(identity, size));
        
        // Verify the result
        boolean passed = true;
        final float epsilon = 1e-5f;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                int index = i * size + j;
                float value = identity[index];
                if (i == j)
                {
                    passed &= Math.abs(value - 1.0f) <= epsilon;
                }
                else
                {
                    passed &= Math.abs(value) <= epsilon;
                }
            }
        }
        System.out.println((passed ? "PASSED" : "FAILED"));

        // Clean up
        cublasDestroy(handle);
    }

    /**
     * Copies the given n x n matrix into device memory, inverts it by calling
     * {@link #invertMatrix(cublasHandle, int, Pointer)}, and copies it back 
     * into the given array.
     * 
     * @param handle The CUBLAS handle
     * @param n The size of the matrix
     * @param A The matrix
     */
    public static void invertMatrix(cublasHandle handle, int n, float A[])
    {
        Pointer dA = new Pointer();
        cudaMalloc(dA, n * n * Sizeof.FLOAT);
        cublasSetMatrix(n, n, Sizeof.FLOAT, Pointer.to(A), n, dA, n);

        invertMatrix(handle, n, dA);

        cublasGetMatrix(n, n, Sizeof.FLOAT, dA, n, Pointer.to(A), n);
        cudaFree(dA);
    }

    /**
     * Invert the n x n matrix that is given in device memory.
     * 
     * @param n The size of the matrix
     * @param dA The matrix
     */
    public static void invertMatrix(cublasHandle handle, int n, Pointer dA)
    {
        // Perform LU factorization
        int[] pivots = cudaSgetrfSquare(handle, n, dA);

        // Perform inversion on factorized matrix
        cudaSgetri(handle, n, dA, pivots);
    }

    /**
     * Convenience method that returns a pointer with the given offset (in
     * number of 4-byte float elements) from the given pointer.
     * 
     * @param p The pointer
     * @param floatOffset The offset, in number of float elements
     * @return The new pointer
     */
    private static Pointer at(Pointer p, int floatOffset)
    {
        return p.withByteOffset(floatOffset * Sizeof.FLOAT);
    }

    /**
     * cudaSgetrf performs an in-place LU factorization on a square matrix. 
     * Uses the unblocked BLAS2 approach
     * 
     * @param n The matrix size
     * @param dA The pointer to the matrix (in device memory)
     * @return The pivots
     */
    private static int[] cudaSgetrfSquare(
        cublasHandle handle, int n, Pointer dA)
    {
        int[] pivots = new int[n];
        for (int i = 0; i < n; i++)
        {
            pivots[i] = i;
        }

        Pointer minusOne = Pointer.to(new float[] { -1.0f });
        float[] factor = { 0.0f };
        Pointer pFactor = Pointer.to(factor);
        for (int i = 0; i < n - 1; i++)
        {
            Pointer offset = at(dA, i * n + i);

            int max[] = { 0 };
            cublasIsamax(handle, n - i, offset, 1, Pointer.to(max));
            int pivot = i - 1 + max[0];
            if (pivot != i)
            {
                pivots[i] = pivot;
                cublasSswap(handle, n, at(dA, pivot), n, at(dA, i), n);
            }

            cublasGetVector(1, Sizeof.FLOAT, offset, 1, pFactor, 1);
            factor[0] = 1 / factor[0];
            cublasSscal(handle, n - i - 1, pFactor, at(offset, 1), 1);
            cublasSger(handle, n - i - 1, n - i - 1, minusOne, at(offset, 1), 
                1, at(offset, n), n, at(offset, n + 1), n);
        }
        return pivots;
    }

    /***
     * cudaSgetri Computes the inverse of an LU-factorized square matrix
     * 
     * @param n The matrix size
     * @param dA The matrix in device memory
     * @param pivots The pivots
     */
    private static void cudaSgetri(
        cublasHandle handle, int n, Pointer dA, int[] pivots)
    {
        // Perform inv(U)
        cudaStrtri(handle, n, dA);

        // Solve inv(A)*L = inv(U)
        Pointer dWork = new Pointer();
        cudaMalloc(dWork, (n - 1) * Sizeof.FLOAT);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        Pointer minusOne = Pointer.to(new float[]{ -1.0f });
        for (int i = n - 1; i > 0; i--)
        {
            Pointer offset = at(dA, ((i - 1) * n + i));
            cudaMemcpy(dWork, offset, (n - 1) * Sizeof.FLOAT,
                cudaMemcpyDeviceToDevice);
            cublasSscal(handle, n - i, zero, offset, 1);
            cublasSgemv(handle, CUBLAS_OP_N, n, n - i, minusOne, 
                at(dA, i * n), n, dWork, 1, one, at(dA, ((i - 1) * n)), 1);
        }

        cudaFree(dWork);

        // Pivot back to original order
        for (int i = n - 1; i >= 0; i--)
        {
            if (i != pivots[i])
            {
                cublasSswap(handle, n, at(dA, i * n), 1, 
                    at(dA, pivots[i] * n), 1);
            }
        }

    }

    /***
     * cudaStrtri Computes the inverse of an upper triangular matrix in place
     * Uses the unblocked BLAS2 approach
     * 
     * @param n The size of the matrix
     * @param dA The matrix
     */
    private static void cudaStrtri(cublasHandle handle, int n, Pointer dA)
    {
        float[] factor = { 0.0f };
        Pointer pFactor = Pointer.to(factor);
        for (int i = 0; i < n; i++)
        {
            Pointer offset = at(dA, i * n);
            cublasGetVector(1, Sizeof.FLOAT, at(offset, i), 1, pFactor, 1);
            factor[0] = 1 / factor[0];
            cublasSetVector(1, Sizeof.FLOAT, pFactor, 1, at(offset, i), 1);

            factor[0] = -factor[0];
            cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_OP_N, i, dA, n, offset, 1);
            cublasSscal(handle, i, pFactor, offset, 1);
        }
    }

    // === Utility methods for this sample ====================================

    /**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param size The size of the matrices
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
    private static void multiply(cublasHandle handle, int size, float A[],
        float B[], float C[])
    {
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cudaMalloc(dA, size * size * Sizeof.FLOAT);
        cudaMalloc(dB, size * size * Sizeof.FLOAT);
        cudaMalloc(dC, size * size * Sizeof.FLOAT);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, one, 
            dA, size, dB, size, zero, dC, size);

        cublasGetVector(size * size, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

}