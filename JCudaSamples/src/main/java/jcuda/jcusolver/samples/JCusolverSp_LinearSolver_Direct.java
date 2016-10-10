/*
 * JCusolver - Java bindings for CUSOLVER, the NVIDIA CUDA solver
 * library, to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcusolver.samples;

import static jcuda.jcusolver.JCusolverSp.cusolverSpCreate;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvchol;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvcholHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvluHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvqr;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvqrHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpSetStream;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrissymHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrpermHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrperm_bufferSizeHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrsymamdHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrsymrcmHost;
import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDcsrmv;
import static jcuda.jcusparse.JCusparse.cusparseGetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.JCusparse.cusparseSetStream;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ONE;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaDeviceReset;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaStreamCreate;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.io.FileInputStream;
import java.io.IOException;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusolver.JCusolver;
import jcuda.jcusolver.cusolverSpHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import de.javagl.matrixmarketreader.CSR;
import de.javagl.matrixmarketreader.MatrixMarketCSR;

/**
 * This is a sample showing how to use JCusolverSp.<br>
 * <br>
 * <br>
 * This is a direct 1:1 port of the NVIDIA sample "JCusolverSp_LinearSolver",
 * with as few modifications as possible. Only the command line argument
 * parsing and the "MatrixMarket" file reading utilities have been rewritten.
 * For details and default values of the command line arguments, see
 * the {@link #main(String[])} method.
 */
public class JCusolverSp_LinearSolver_Direct
{
    public static void main(String[] args)
    {
        JCuda.setExceptionsEnabled(true);;
        JCusparse.setExceptionsEnabled(true);
        JCusolver.setExceptionsEnabled(true);

        String path = "src/main/resources/data/jcusolver/";
        String fileName = path + "lap2D_5pt_n100.mtx";
        String testFunc = "chol"; // "chol", "lu", "qr"
        String reorder = "symrcm"; // "symrcm", "symamd", null

        runTest(
            "-F="+fileName,
            "-R="+testFunc,
            "-P="+reorder);
    }

    private static void runTest(String ... args)
    {
        TestOpts opts = parse(args);

        cusolverSpHandle handle = new cusolverSpHandle();
        cusparseHandle cusparseHandle = new cusparseHandle();
        cudaStream_t stream = new cudaStream_t();
        cusparseMatDescr descrA = new cusparseMatDescr();

        int rowsA = 0; // number of rows of A
        int colsA = 0; // number of columns of A
        int nnzA  = 0; // number of nonzeros of A
        int baseA = 0; // base index in CSR format

        // CSR(A) from I/O
        int h_csrRowPtrA[] = null;
        int h_csrColIndA[] = null;
        double h_csrValA[] = null;

        double h_x[] = null; // x = A \ b
        double h_b[] = null; // b = ones(m,1)
        double h_r[] = null; // r = b - A*x

        int h_Q[] = null; // <int> n
        // reorder to reduce zero fill-in
        // Q = symrcm(A) or Q = symamd(A)
        // B = Q*A*Q^T
        int h_csrRowPtrB[] = null; // <int> n+1
        int h_csrColIndB[] = null; // <int> nnzA
        double h_csrValB[] = null; // <double> nnzA
        int h_mapBfromA[] = null;  // <int> nnzA

        long size_perm = 0;
        byte buffer_cpu[] = null; // working space for permutation: B = Q*A*Q^T

        Pointer d_csrRowPtrA = new Pointer(); // int
        Pointer d_csrColIndA = new Pointer(); // int
        Pointer d_csrValA = new Pointer(); // double
        Pointer d_x = new Pointer(); // x = A \ b // double
        Pointer d_b = new Pointer(); // a copy of h_b // double
        Pointer d_r = new Pointer(); // r = b - A*x // double

        double tol = 1.e-12;
        int reorder = 0; // no reordering
        int singularity = 0; // -1 if A is invertible under tol.

        // the constants are used in residual evaluation, r = b - A*x
        double minus_one = -1.0;
        double one = 1.0;

        double x_inf = 0.0;
        double r_inf = 0.0;
        double A_inf = 0.0;
        int issym = 0;

        long start, stop;
        double time_solve_cpu;
        double time_solve_gpu;

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
            System.err.println("Error: only support square matrix");
            return;
        }

        cusolverSpCreate(handle);
        cusparseCreate(cusparseHandle);

        cudaStreamCreate(stream);
        cusolverSpSetStream(handle, stream);

        cusparseSetStream(cusparseHandle, stream);

        cusparseCreateMatDescr(descrA);

        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

        if (baseA != 0)
        {
            cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
        }
        else
        {
            cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        }

        h_x = new double[colsA];
        h_b = new double[rowsA];
        h_r = new double[rowsA];

        cudaMalloc(d_csrRowPtrA, Sizeof.INT*(rowsA+1));
        cudaMalloc(d_csrColIndA, Sizeof.INT*nnzA);
        cudaMalloc(d_csrValA   , Sizeof.DOUBLE*nnzA);
        cudaMalloc(d_x, Sizeof.DOUBLE*colsA);
        cudaMalloc(d_b, Sizeof.DOUBLE*rowsA);
        cudaMalloc(d_r, Sizeof.DOUBLE*rowsA);

        // verify if A has symmetric pattern or not
        int issymArray[] = new int[] { -1 };
        cusolverSpXcsrissymHost(
            handle, rowsA, nnzA, descrA, Pointer.to(h_csrRowPtrA),
            Pointer.to(h_csrRowPtrA).withByteOffset(1 * Sizeof.INT),
            Pointer.to(h_csrColIndA), Pointer.to(issymArray));
        issym = issymArray[0];
        if (opts.testFunc.equals("chol"))
        {
            if (issym != 1)
            {
                System.err.printf("Error: A has no symmetric pattern, please use LU or QR \n");
                return;
            }
        }

        System.out.printf("step 2: reorder the matrix A to minimize zero fill-in\n");
        System.out.printf("        if the user choose a reordering by -P=symrcm or -P=symamd\n");
        System.out.printf("        The reordering will overwrite A such that \n");
        System.out.printf("            A := A(Q,Q) where Q = symrcm(A) or Q = symamd(A)\n");
        if (opts.reorder != null && !opts.reorder.equals("null"))
        {
            h_Q          = new int[colsA];
            h_csrRowPtrB = new int[rowsA+1];
            h_csrColIndB = new int[nnzA];
            h_csrValB    = new double[nnzA];
            h_mapBfromA  = new int[nnzA];


            if (opts.reorder.equalsIgnoreCase("symrcm"))
            {
                cusolverSpXcsrsymrcmHost(
                    handle, rowsA, nnzA,
                    descrA, Pointer.to(h_csrRowPtrA), Pointer.to(h_csrColIndA),
                    Pointer.to(h_Q));
            }
            else if (opts.reorder.equalsIgnoreCase("symamd"))
            {
                cusolverSpXcsrsymamdHost(
                    handle, rowsA, nnzA,
                    descrA, Pointer.to(h_csrRowPtrA), Pointer.to(h_csrColIndA),
                    Pointer.to(h_Q));
            }
            else
            {
                System.out.printf("Error: %s is unknown reordering\n", opts.reorder);
                return;
            }

            // B = Q*A*Q^T
            memcpy(h_csrRowPtrB, h_csrRowPtrA, rowsA+1);
            memcpy(h_csrColIndB, h_csrColIndA, nnzA);

            long size_permArray[] =  { -1 };
            cusolverSpXcsrperm_bufferSizeHost(
                handle, rowsA, colsA, nnzA,
                descrA, Pointer.to(h_csrRowPtrB), Pointer.to(h_csrColIndB),
                Pointer.to(h_Q), Pointer.to(h_Q),
                size_permArray);
            size_perm = size_permArray[0];

            buffer_cpu = new byte[(int)size_perm];

            // h_mapBfromA = Identity
            for(int j = 0 ; j < nnzA ; j++)
            {
                h_mapBfromA[j] = j;
            }
            cusolverSpXcsrpermHost(
                handle, rowsA, colsA, nnzA,
                descrA, Pointer.to(h_csrRowPtrB), Pointer.to(h_csrColIndB),
                Pointer.to(h_Q), Pointer.to(h_Q),
                Pointer.to(h_mapBfromA),
                Pointer.to(buffer_cpu));

            // B = A( mapBfromA )
            for(int j = 0 ; j < nnzA ; j++)
            {
                h_csrValB[j] = h_csrValA[ h_mapBfromA[j] ];
            }

            // A := B
            memcpy(h_csrRowPtrA, h_csrRowPtrB, rowsA+1);
            memcpy(h_csrColIndA, h_csrColIndB, nnzA);
            memcpy(h_csrValA   , h_csrValB   , nnzA);
        }

        System.out.printf("step 2: set right hand side vector (b) to 1\n");
        for(int row = 0 ; row < rowsA ; row++)
        {
            h_b[row] = 1.0;
        }



        System.out.printf("step 3: prepare data on device\n");
        cudaMemcpy(d_csrRowPtrA, Pointer.to(h_csrRowPtrA), Sizeof.INT*(rowsA+1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrColIndA, Pointer.to(h_csrColIndA), Sizeof.INT*nnzA     , cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrValA   , Pointer.to(h_csrValA)   , Sizeof.DOUBLE*nnzA  , cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, Pointer.to(h_b), Sizeof.DOUBLE*rowsA, cudaMemcpyHostToDevice);

        System.out.printf("step 4: solve A*x = b on CPU\n");
        // A and b are read-only
        start = System.nanoTime();

        int singularityArray[] = { -1 };
        if (opts.testFunc.equals("chol"))
        {
            cusolverSpDcsrlsvcholHost(
                handle, rowsA, nnzA,
                descrA, Pointer.to(h_csrValA), Pointer.to(h_csrRowPtrA), Pointer.to(h_csrColIndA),
                Pointer.to(h_b), tol, reorder, Pointer.to(h_x), singularityArray);
        }
        else if (opts.testFunc.equals("lu"))
        {
            cusolverSpDcsrlsvluHost(
                handle, rowsA, nnzA,
                descrA, Pointer.to(h_csrValA), Pointer.to(h_csrRowPtrA), Pointer.to(h_csrColIndA),
                Pointer.to(h_b), tol, reorder, Pointer.to(h_x), singularityArray);

        }
        else if (opts.testFunc.equals("qr"))
        {
            cusolverSpDcsrlsvqrHost(
                handle, rowsA, nnzA,
                descrA, Pointer.to(h_csrValA), Pointer.to(h_csrRowPtrA), Pointer.to(h_csrColIndA),
                Pointer.to(h_b), tol, reorder, Pointer.to(h_x), singularityArray);

        }
        else
        {
            System.out.printf("Error: %s is unknown function\n", opts.testFunc);
            return;
        }
        stop = System.nanoTime();

        time_solve_cpu = (stop - start) / 1e9;

        singularity = singularityArray[0];
        if (0 <= singularity)
        {
            System.out.printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
        }

        System.out.printf("step 5: evaluate residual r = b - A*x (result on CPU)\n");
        cudaMemcpy(d_r, Pointer.to(h_b), Sizeof.DOUBLE*rowsA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, Pointer.to(h_x), Sizeof.DOUBLE*colsA, cudaMemcpyHostToDevice);
        cusparseDcsrmv(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            rowsA,
            colsA,
            nnzA,
            Pointer.to(new double[] { minus_one }),
            descrA,
            d_csrValA,
            d_csrRowPtrA,
            d_csrColIndA,
            d_x,
            Pointer.to(new double[] { one }),
            d_r);
        cudaMemcpy(Pointer.to(h_r), d_r, Sizeof.DOUBLE*rowsA, cudaMemcpyDeviceToHost);

        x_inf = vec_norminf(colsA, h_x);
        r_inf = vec_norminf(rowsA, h_r);
        A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA);

        System.out.printf("(CPU) |b - A*x| = %E \n", r_inf);
        System.out.printf("(CPU) |A| = %E \n", A_inf);
        System.out.printf("(CPU) |x| = %E \n", x_inf);
        System.out.printf("(CPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));

        System.out.printf("step 6: solve A*x = b on GPU\n");
        // d_A and d_b are read-only
        start = System.nanoTime();

        if (opts.testFunc.equals("chol") )
        {
            cusolverSpDcsrlsvchol(
                handle, rowsA, nnzA,
                descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
                d_b, tol, reorder, d_x, singularityArray);

        }
        else if (opts.testFunc.equals("lu"))
        {
            System.out.printf("WARNING: no LU available on GPU \n");
        }
        else if (opts.testFunc.equals("qr") )
        {
            cusolverSpDcsrlsvqr(
                handle, rowsA, nnzA,
                descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
                d_b, tol, reorder, d_x, singularityArray);
        }
        else
        {
            System.out.printf("Error: %s is unknow function\n", opts.testFunc);
            return ;
        }
        cudaDeviceSynchronize();
        stop = System.nanoTime();

        time_solve_gpu = (stop - start) / 1e9;

        singularity = singularityArray[0];
        if (0 <= singularity)
        {
            System.out.printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
        }


        System.out.printf("step 7: evaluate residual r = b - A*x (result on GPU)\n");
        cudaMemcpy(d_r, d_b, Sizeof.DOUBLE*rowsA, cudaMemcpyDeviceToDevice);

        cusparseDcsrmv(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            rowsA,
            colsA,
            nnzA,
            Pointer.to(new double[] { minus_one }),
            descrA,
            d_csrValA,
            d_csrRowPtrA,
            d_csrColIndA,
            d_x,
            Pointer.to(new double[] { one }),
            d_r);

        cudaMemcpy(Pointer.to(h_x), d_x, Sizeof.DOUBLE*colsA, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(h_r), d_r, Sizeof.DOUBLE*rowsA, cudaMemcpyDeviceToHost);

        x_inf = vec_norminf(colsA, h_x);
        r_inf = vec_norminf(rowsA, h_r);

        if (!opts.testFunc.equals("lu"))
        {
            // only cholesky and qr have GPU version
            System.out.printf("(GPU) |b - A*x| = %E \n", r_inf);
            System.out.printf("(GPU) |A| = %E \n", A_inf);
            System.out.printf("(GPU) |x| = %E \n", x_inf);
            System.out.printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));
        }

        System.out.printf("timing %s: CPU = %10.6f sec , GPU = %10.6f sec\n",
            opts.testFunc, time_solve_cpu, time_solve_gpu);
        cudaDeviceReset();
    }

    private static void memcpy(int dst[], int src[], int n)
    {
        System.arraycopy(src, 0, dst, 0, n);
    }
    private static void memcpy(double dst[], double src[], int n)
    {
        System.arraycopy(src, 0, dst, 0, n);
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

    private static double csr_mat_norminf(
        int m,
        int n,
        int nnzA,
        cusparseMatDescr descrA,
        double csrValA[],
        int csrRowPtrA[],
        int csrColIndA[])
    {
        int baseA = (CUSPARSE_INDEX_BASE_ONE ==
            cusparseGetMatIndexBase(descrA))? 1:0;
        double norminf = 0;
        for (int i = 0; i < m; i++)
        {
            double sum = 0.0;
            int start = csrRowPtrA[i] - baseA;
            int end = csrRowPtrA[i + 1] - baseA;
            for (int colidx = start; colidx < end; colidx++)
            {
                // const int j = csrColIndA[colidx] - baseA;
                double A_abs = Math.abs(csrValA[colidx]);
                sum += A_abs;
            }
            norminf = (norminf > sum) ? norminf : sum;
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
