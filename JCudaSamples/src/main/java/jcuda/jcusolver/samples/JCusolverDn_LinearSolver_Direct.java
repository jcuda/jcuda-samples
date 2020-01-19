/*
 * JCusolver - Java bindings for CUSOLVER, the NVIDIA CUDA solver
 * library, to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcusolver.samples;

import java.io.FileInputStream;
import java.io.IOException;

import de.javagl.matrixmarketreader.CSR;
import de.javagl.matrixmarketreader.MatrixMarketCSR;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusolver.JCusolver;
import jcuda.jcusolver.cusolverDnHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import static jcuda.jcublas.cublasFillMode.*;
import static jcuda.jcublas.cublasDiagType.*;
import static jcuda.jcublas.cublasOperation.*;
import static jcuda.jcublas.cublasSideMode.*;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcusolver.JCusolverDn.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

/**
 * This is a sample showing how to use JCusolverDn.<br>
 * <br>
 * <br>
 * This is a direct 1:1 port of the NVIDIA sample "JCusolverDn_LinearSolver",
 * with as few modifications as possible. Only the command line argument
 * parsing and the "MatrixMarket" file reading utilities have been rewritten.
 * For details and default values of the command line arguments, see
 * the {@link #main(String[])} method.
 */
public class JCusolverDn_LinearSolver_Direct
{
    /*
     *  solve A*x = b by Cholesky factorization
     *
     */
    private static int linearSolverCHOL(
        cusolverDnHandle handle,
        int n,
        Pointer Acopy,
        int lda,
        Pointer b,
        Pointer x)
    {
        int bufferSize[] =  { 0 };
        Pointer info = new Pointer();
        Pointer buffer = new Pointer();
        Pointer A = new Pointer();
        int h_info[] = { 0 };
        long start, stop;
        double time_solve;
        int uplo = CUBLAS_FILL_MODE_LOWER;

        cusolverDnDpotrf_bufferSize(handle, uplo, n, Acopy, lda, bufferSize);

        cudaMalloc(info, Sizeof.INT);
        cudaMalloc(buffer, Sizeof.DOUBLE*bufferSize[0]);
        cudaMalloc(A, Sizeof.DOUBLE*lda*n);

        // prepare a copy of A because potrf will overwrite A with L
        cudaMemcpy(A, Acopy, Sizeof.DOUBLE*lda*n, cudaMemcpyDeviceToDevice);
        cudaMemset(info, 0, Sizeof.INT);

        start = System.nanoTime();

        cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize[0], info);

        cudaMemcpy(Pointer.to(h_info), info, Sizeof.INT, cudaMemcpyDeviceToHost);

        if ( 0 != h_info[0] ){
            System.err.printf("Error: Cholesky factorization failed\n");
        }

        cudaMemcpy(x, b, Sizeof.DOUBLE*n, cudaMemcpyDeviceToDevice);

        cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info);

        cudaDeviceSynchronize();
        stop = System.nanoTime();

        time_solve = (stop - start) / 1e9;
        System.out.printf("timing: cholesky = %10.6f sec\n", time_solve);

        cudaFree(info);
        cudaFree(buffer);
        cudaFree(A);

        return 0;
    }


    /*
     *  solve A*x = b by LU with partial pivoting
     *
     */
    private static int linearSolverLU(
        cusolverDnHandle handle,
        int n,
        Pointer Acopy,
        int lda,
        Pointer b,
        Pointer x)
    {
        int bufferSize[] = { 0 };
        Pointer info = new Pointer();
        Pointer buffer = new Pointer();
        Pointer A = new Pointer();
        Pointer ipiv = new Pointer(); // pivoting sequence
        int h_info[] = { 0 };
        long start, stop;
        double time_solve;

        cusolverDnDgetrf_bufferSize(handle, n, n, Acopy, lda, bufferSize);

        cudaMalloc(info, Sizeof.INT);
        cudaMalloc(buffer, Sizeof.DOUBLE*bufferSize[0]);
        cudaMalloc(A, Sizeof.DOUBLE*lda*n);
        cudaMalloc(ipiv, Sizeof.INT*n);


        // prepare a copy of A because getrf will overwrite A with L
        cudaMemcpy(A, Acopy, Sizeof.DOUBLE*lda*n, cudaMemcpyDeviceToDevice);
        cudaMemset(info, 0, Sizeof.INT);

        start = System.nanoTime();

        cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info);
        cudaMemcpy(Pointer.to(h_info), info, Sizeof.INT, cudaMemcpyDeviceToHost);

        if ( 0 != h_info[0] ){
            System.err.printf("Error: LU factorization failed\n");
        }

        cudaMemcpy(x, b, Sizeof.DOUBLE*n, cudaMemcpyDeviceToDevice);
        cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info);
        cudaDeviceSynchronize();
        stop = System.nanoTime();

        time_solve = (stop - start) / 1e9;
        System.out.printf("timing: LU = %10.6f sec\n", time_solve);

        cudaFree(info  );
        cudaFree(buffer);
        cudaFree(A);
        cudaFree(ipiv);

        return 0;
    }


    /*
     *  solve A*x = b by QR
     *
     */
    private static int linearSolverQR(
        cusolverDnHandle handle,
        int n,
        Pointer Acopy,
        int lda,
        Pointer b,
        Pointer x)
    {
        cublasHandle cublasHandle = new cublasHandle(); // used in residual evaluation
        int bufferSize[] =  { 0 };
        Pointer info = new Pointer();
        Pointer buffer = new Pointer();
        Pointer A = new Pointer();
        Pointer tau = new Pointer();
        int h_info[] = { 0 };
        long start, stop;
        double time_solve;
        double one = 1.0;

        cublasCreate(cublasHandle);

        cusolverDnDgeqrf_bufferSize(handle, n, n, Acopy, lda, bufferSize);

        cudaMalloc(info, Sizeof.INT);
        cudaMalloc(buffer, Sizeof.DOUBLE*bufferSize[0]);
        cudaMalloc(A, Sizeof.DOUBLE*lda*n);
        cudaMalloc(tau, Sizeof.DOUBLE*n);

        // prepare a copy of A because getrf will overwrite A with L
        cudaMemcpy(A, Acopy, Sizeof.DOUBLE*lda*n, cudaMemcpyDeviceToDevice);

        cudaMemset(info, 0, Sizeof.INT);

        start = System.nanoTime();

        // compute QR factorization
        cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize[0], info);

        cudaMemcpy(Pointer.to(h_info), info, Sizeof.INT, cudaMemcpyDeviceToHost);

        if ( 0 != h_info[0] ){
            System.err.printf("Error: LU factorization failed\n");
        }

        cudaMemcpy(x, b, Sizeof.DOUBLE*n, cudaMemcpyDeviceToDevice);

        // compute Q^T*b
        cusolverDnDormqr(
            handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_OP_T,
            n,
            1,
            n,
            A,
            lda,
            tau,
            x,
            n,
            buffer,
            bufferSize[0],
            info);

        // x = R \ Q^T*b
        cublasDtrsm(
             cublasHandle,
             CUBLAS_SIDE_LEFT,
             CUBLAS_FILL_MODE_UPPER,
             CUBLAS_OP_N,
             CUBLAS_DIAG_NON_UNIT,
             n,
             1,
             Pointer.to(new double[] { one }),
             A,
             lda,
             x,
             n);
        cudaDeviceSynchronize();
        stop = System.nanoTime();

        time_solve = (stop - start) / 1e9;
        System.out.printf("timing: QR = %10.6f sec\n", time_solve);

        cublasDestroy(cublasHandle);
        cudaFree(info  );
        cudaFree(buffer);
        cudaFree(A);
        cudaFree(tau);

        return 0;
    }



    public static void main(String args[])
    {
        JCuda.setExceptionsEnabled(true);;
        JCusparse.setExceptionsEnabled(true);
        JCusolver.setExceptionsEnabled(true);

        String path = "src/main/resources/data/jcusolver/";
        String fileName = path + "gr_900_900_crg.mtx";
        String testFunc = "chol"; // "chol", "lu", "qr"

        runTest(
            "-F="+fileName,
            "-R="+testFunc);
    }

    private static void runTest(String ... args)
    {
        TestOpts opts = parse(args);

        cusolverDnHandle handle = new cusolverDnHandle();
        cublasHandle cublasHandle = new cublasHandle(); // used in residual evaluation
        cudaStream_t stream = new cudaStream_t();

        int rowsA = 0; // number of rows of A
        int colsA = 0; // number of columns of A
        int nnzA  = 0; // number of nonzeros of A
        int baseA = 0; // base index in CSR format
        int lda   = 0; // leading dimension in dense matrix

        // CSR(A) from I/O
        int h_csrRowPtrA[] = null;
        int h_csrColIndA[] = null;
        double h_csrValA[] = null;

        double h_A[] = null; // dense matrix from CSR(A)
        double h_x[] = null; // a copy of d_x
        double h_b[] = null; // b = ones(m,1)
        double h_r[] = null; // r = b - A*x, a copy of d_r

        Pointer d_A = new Pointer(); // a copy of h_A
        Pointer d_x = new Pointer(); // x = A \ b
        Pointer d_b = new Pointer(); // a copy of h_b
        Pointer d_r = new Pointer(); // r = b - A*x

        // the constants are used in residual evaluation, r = b - A*x
        double minus_one = -1.0;
        double one = 1.0;

        double x_inf = 0.0;
        double r_inf = 0.0;
        double A_inf = 0.0;

        CSR csr = null;
        try
        {

            csr = MatrixMarketCSR.readCSR(
                new FileInputStream(opts.sparseMatFilename));
        }
        catch (IOException e)
        {
            e.printStackTrace();
            return;
        }
        rowsA = csr.numRows;
        colsA = csr.numCols;
        nnzA = csr.values.length;
        h_csrValA = csr.values;
        h_csrRowPtrA = csr.rowPointers;
        h_csrColIndA = csr.columnIndices;

        baseA = h_csrRowPtrA[0]; // baseA = {0,1}

        System.out.printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n",
            rowsA, colsA, nnzA, baseA);

        if ( rowsA != colsA )
        {
            System.err.printf("Error: only support square matrix\n");
            return;
        }

        System.out.printf("step 2: convert CSR(A) to dense matrix\n");

        lda = rowsA;

        h_A = new double[lda*colsA];
        h_x = new double[colsA];
        h_b = new double[rowsA];
        h_r = new double[rowsA];

        for(int row = 0 ; row < rowsA ; row++)
        {
            int start = h_csrRowPtrA[row  ] - baseA;
            int end   = h_csrRowPtrA[row+1] - baseA;
            for(int colidx = start ; colidx < end ; colidx++)
            {
                int col = h_csrColIndA[colidx] - baseA;
                double Areg = h_csrValA[colidx];
                h_A[row + col*lda] = Areg;
            }
        }

        System.out.printf("step 3: set right hand side vector (b) to 1\n");
        for(int row = 0 ; row < rowsA ; row++)
        {
            h_b[row] = 1.0;
        }

        // verify if A is symmetric or not.
        if (opts.testFunc.equals("chol") )
        {
            int issym = 1;
            for(int j = 0 ; j < colsA ; j++)
            {
                for(int i = j ; i < rowsA ; i++)
                {
                    double Aij = h_A[i + j*lda];
                    double Aji = h_A[j + i*lda];
                    if ( Aij != Aji )
                    {
                        issym = 0;
                        break;
                    }
                }
            }
            if (issym == 0)
            {
                System.out.printf("Error: A has no symmetric pattern, please use LU or QR \n");
                return;
            }
        }

        cusolverDnCreate(handle);
        cublasCreate(cublasHandle);
        cudaStreamCreate(stream);

        cusolverDnSetStream(handle, stream);
        cublasSetStream(cublasHandle, stream);


        cudaMalloc(d_A, Sizeof.DOUBLE*lda*colsA);
        cudaMalloc(d_x, Sizeof.DOUBLE*colsA);
        cudaMalloc(d_b, Sizeof.DOUBLE*rowsA);
        cudaMalloc(d_r, Sizeof.DOUBLE*rowsA);

        System.out.printf("step 4: prepare data on device\n");
        cudaMemcpy(d_A, Pointer.to(h_A), Sizeof.DOUBLE*lda*colsA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, Pointer.to(h_b), Sizeof.DOUBLE*rowsA, cudaMemcpyHostToDevice);

        System.out.printf("step 5: solve A*x = b \n");
        // d_A and d_b are read-only
        if (opts.testFunc.equals("chol") )
        {
            linearSolverCHOL(handle, rowsA, d_A, lda, d_b, d_x);
        }
        else if (opts.testFunc.equals("lu") )
        {
            linearSolverLU(handle, rowsA, d_A, lda, d_b, d_x);
        }
        else if (opts.testFunc.equals("qr") )
        {
            linearSolverQR(handle, rowsA, d_A, lda, d_b, d_x);
        }
        else
        {
            System.err.printf("Error: %s is unknown function\n", opts.testFunc);
            return;
        }
        System.out.printf("step 6: evaluate residual\n");
        cudaMemcpy(d_r, d_b, Sizeof.DOUBLE*rowsA, cudaMemcpyDeviceToDevice);

        // r = b - A*x
        cublasDgemm(
            cublasHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            rowsA,
            1,
            colsA,
            Pointer.to(new double[] { minus_one }),
            d_A,
            lda,
            d_x,
            rowsA,
            Pointer.to(new double[] { one }),
            d_r,
            rowsA);

        cudaMemcpy(Pointer.to(h_x), d_x, Sizeof.DOUBLE*colsA, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(h_r), d_r, Sizeof.DOUBLE*rowsA, cudaMemcpyDeviceToHost);

        x_inf = vec_norminf(colsA, h_x);
        r_inf = vec_norminf(rowsA, h_r);
        A_inf = mat_norminf(rowsA, colsA, h_A, lda);

        System.out.printf("|b - A*x| = %E \n", r_inf);
        System.out.printf("|A| = %E \n", A_inf);
        System.out.printf("|x| = %E \n", x_inf);
        System.out.printf("|b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));

        cusolverDnDestroy(handle);
        cublasDestroy(cublasHandle);
        cudaStreamDestroy(stream);

        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_b);
        cudaFree(d_r);

        cudaDeviceReset();
    }

    private static double vec_norminf(int n, double x[])
    {
        double norminf = 0;
        for (int j = 0; j < n; j++)
        {
            double x_abs = Math.abs(x[j]);
            norminf = (norminf > x_abs) ? norminf : x_abs;
        }
        return norminf;
    }

    /*
     * |A| = max { |A|*ones(m,1) }
     */
    private static double mat_norminf(
        int m,
        int n,
        double A[],
        int lda)
    {
        double norminf = 0;
        for(int i = 0 ; i < m ; i++){
            double sum = 0.0;
            for(int j = 0 ; j < n ; j++){
               double A_abs = Math.abs(A[i + j*lda]);
               sum += A_abs;
            }
            norminf = (norminf > sum)? norminf : sum;
        }
        return norminf;
    }

    private static class TestOpts
    {
        String sparseMatFilename;  // by switch -F<filename>
        String testFunc; // by switch -R<name>
        String reorder; // by switch -P<name>
        int lda; // by switch -lda<int>
    };

    private static TestOpts parse(String ... args)
    {
        TestOpts testOpts = new TestOpts();
        testOpts.sparseMatFilename = "lap2D_5pt_n100.mtx";
        testOpts.testFunc = "chol";
        testOpts.reorder = "symrcm";
        testOpts.lda = 0;

        for (String arg : args)
        {
            if (arg.startsWith("-F="))
            {
                testOpts.sparseMatFilename = arg.substring(3).trim();
            }
            if (arg.startsWith("-R="))
            {
                testOpts.testFunc = arg.substring(3).trim();
            }
            if (arg.startsWith("-P="))
            {
                testOpts.reorder = arg.substring(3).trim();
            }
            if (arg.startsWith("-lda="))
            {
                String s = arg.substring(5).trim();
                try
                {
                    testOpts.lda = Integer.parseInt(s);
                }
                catch (NumberFormatException e)
                {
                    System.err.println("Invalid lda: "+s+", ignored");
                }
            }
        }
        return testOpts;
    }

}
