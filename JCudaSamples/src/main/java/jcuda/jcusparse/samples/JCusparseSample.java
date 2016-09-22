/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcusparse.samples;

import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.jcusparse.JCusparse.cusparseScsrmm;
import static jcuda.jcusparse.JCusparse.cusparseScsrmv;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.JCusparse.cusparseSsctr;
import static jcuda.jcusparse.JCusparse.cusparseXcoo2csr;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaMemset;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.runtime.JCuda;


/**
 * A sample application showing how to use JCusparse.<br>
 * <br>
 * This is a direct port of the NVIDIA CUSPARSE documentation example.
 */
public class JCusparseSample
{
    public static void main(String args[])
    {
        // Enable exceptions and subsequently omit error checks in this sample
        JCusparse.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);

        // Variable declarations
        cusparseHandle handle = new cusparseHandle();
        cusparseMatDescr descra = new cusparseMatDescr();

        int cooRowIndexHostPtr[];
        int cooColIndexHostPtr[];
        float cooValHostPtr[];

        Pointer cooRowIndex = new Pointer();
        Pointer cooColIndex = new Pointer();
        Pointer cooVal = new Pointer();

        int xIndHostPtr[];
        float xValHostPtr[];
        float yHostPtr[];

        Pointer xInd = new Pointer();
        Pointer xVal = new Pointer();
        Pointer y = new Pointer();
        Pointer csrRowPtr = new Pointer();

        float zHostPtr[];
        Pointer z = new Pointer();
        int n, nnz, nnz_vector, i, j;

        System.out.println("Testing example");

        // Create the following sparse test matrix in COO format
        // | 1.0     2.0 3.0 |
        // |     4.0         |
        // | 5.0     6.0 7.0 |
        // |     8.0     9.0 |
        n = 4;
        nnz = 9;
        cooRowIndexHostPtr = new int[nnz];
        cooColIndexHostPtr = new int[nnz];
        cooValHostPtr      = new float[nnz];

        cooRowIndexHostPtr[0]=0; cooColIndexHostPtr[0]=0; cooValHostPtr[0]=1.0f;
        cooRowIndexHostPtr[1]=0; cooColIndexHostPtr[1]=2; cooValHostPtr[1]=2.0f;
        cooRowIndexHostPtr[2]=0; cooColIndexHostPtr[2]=3; cooValHostPtr[2]=3.0f;
        cooRowIndexHostPtr[3]=1; cooColIndexHostPtr[3]=1; cooValHostPtr[3]=4.0f;
        cooRowIndexHostPtr[4]=2; cooColIndexHostPtr[4]=0; cooValHostPtr[4]=5.0f;
        cooRowIndexHostPtr[5]=2; cooColIndexHostPtr[5]=2; cooValHostPtr[5]=6.0f;
        cooRowIndexHostPtr[6]=2; cooColIndexHostPtr[6]=3; cooValHostPtr[6]=7.0f;
        cooRowIndexHostPtr[7]=3; cooColIndexHostPtr[7]=1; cooValHostPtr[7]=8.0f;
        cooRowIndexHostPtr[8]=3; cooColIndexHostPtr[8]=3; cooValHostPtr[8]=9.0f;

        // Print the matrix
        System.out.printf("Input data:\n");
        for (i=0; i<nnz; i++)
        {
            System.out.printf("cooRowIndedHostPtr[%d]=%d  ",i,cooRowIndexHostPtr[i]);
            System.out.printf("cooColIndedHostPtr[%d]=%d  ",i,cooColIndexHostPtr[i]);
            System.out.printf("cooValHostPtr[%d]=%f     \n",i,cooValHostPtr[i]);
        }

        // Create a sparse and a dense vector
        // xVal=[100.0, 200.0, 400.0] (sparse)
        // xInd=[0      1      3    ]
        // y   =[10.0, 20.0, 30.0, 40.0 | 50.0, 60.0, 70.0, 80.0] (dense)
        nnz_vector = 3;
        xIndHostPtr = new int[nnz_vector];
        xValHostPtr = new float[nnz_vector];
        yHostPtr    = new float[2*n];
        zHostPtr    = new float[2*(n+1)];

        yHostPtr[0] = 10.0f;  xIndHostPtr[0]=0; xValHostPtr[0]=100.0f;
        yHostPtr[1] = 20.0f;  xIndHostPtr[1]=1; xValHostPtr[1]=200.0f;
        yHostPtr[2] = 30.0f;
        yHostPtr[3] = 40.0f;  xIndHostPtr[2]=3; xValHostPtr[2]=400.0f;
        yHostPtr[4] = 50.0f;
        yHostPtr[5] = 60.0f;
        yHostPtr[6] = 70.0f;
        yHostPtr[7] = 80.0f;

        // Print the vectors
        for (j=0; j<2; j++)
        {
            for (i=0; i<n; i++)
            {
                System.out.printf("yHostPtr[%d,%d]=%f\n", i,j,yHostPtr[i+n*j]);
            }
        }
        for (i=0; i<nnz_vector; i++)
        {
            System.out.printf("xIndHostPtr[%d]=%d  ", i, xIndHostPtr[i]);
            System.out.printf("xValHostPtr[%d]=%f\n", i, xValHostPtr[i]);
        }

        // Allocate GPU memory and copy the matrix and vectors into it
        cudaMalloc(cooRowIndex, nnz*Sizeof.INT);
        cudaMalloc(cooColIndex, nnz*Sizeof.INT);
        cudaMalloc(cooVal,      nnz*Sizeof.FLOAT);
        cudaMalloc(y,           2*n*Sizeof.FLOAT);
        cudaMalloc(xInd,        nnz_vector*Sizeof.INT);
        cudaMalloc(xVal,        nnz_vector*Sizeof.FLOAT);
        cudaMemcpy(cooRowIndex, Pointer.to(cooRowIndexHostPtr), nnz*Sizeof.INT,          cudaMemcpyHostToDevice);
        cudaMemcpy(cooColIndex, Pointer.to(cooColIndexHostPtr), nnz*Sizeof.INT,          cudaMemcpyHostToDevice);
        cudaMemcpy(cooVal,      Pointer.to(cooValHostPtr),      nnz*Sizeof.FLOAT,        cudaMemcpyHostToDevice);
        cudaMemcpy(y,           Pointer.to(yHostPtr),           2*n*Sizeof.FLOAT,        cudaMemcpyHostToDevice);
        cudaMemcpy(xInd,        Pointer.to(xIndHostPtr),        nnz_vector*Sizeof.INT,   cudaMemcpyHostToDevice);
        cudaMemcpy(xVal,        Pointer.to(xValHostPtr),        nnz_vector*Sizeof.FLOAT, cudaMemcpyHostToDevice);

        // Initialize JCusparse library
        cusparseCreate(handle);

        // Create and set up matrix descriptor
        cusparseCreateMatDescr(descra);
        cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO);

        // Exercise conversion routines (convert matrix from COO 2 CSR format)
        cudaMalloc(csrRowPtr, (n+1)*Sizeof.INT);
        cusparseXcoo2csr(handle, cooRowIndex, nnz, n,
            csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
        //csrRowPtr = [0 3 4 7 9]

        // Exercise Level 1 routines (scatter vector elements)
        Pointer yn = y.withByteOffset(n*Sizeof.FLOAT);
        cusparseSsctr(handle, nnz_vector, xVal, xInd,
            yn, CUSPARSE_INDEX_BASE_ZERO);
        // y = [10 20 30 40 | 100 200 70 400]

        // Exercise Level 2 routines (csrmv)
        Pointer y0 = y.withByteOffset(0);
        cusparseScsrmv(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, 
            Pointer.to(new float[]{2.0f}), descra, cooVal, csrRowPtr, 
            cooColIndex, y0, Pointer.to(new float[]{3.0f}), yn);

        // Print intermediate results (y)
        // y = [10 20 30 40 | 680 760 1230 2240]
        cudaMemcpy(Pointer.to(yHostPtr), y, 2*n*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        for (j=0; j<2; j++)
        {
            for (i=0; i<n; i++)
            {
                System.out.printf("yHostPtr[%d,%d]=%f\n", i,j,yHostPtr[i+n*j]);
            }
        }

        // Exercise Level 3 routines (csrmm)
        cudaMalloc(z, 2*(n+1)*Sizeof.FLOAT);
        cudaMemset(z, 0, 2*(n+1)*Sizeof.FLOAT);
        cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, 2, n, nnz,
            Pointer.to(new float[]{5.0f}), descra, cooVal, csrRowPtr, 
            cooColIndex, y, n, Pointer.to(new float[]{0.0f}), z, n+1);

        // Print final results (z)
        // z = [950 400 2550 2600 0 | 49300 15200 132300 131200 0]
        cudaMemcpy(Pointer.to(zHostPtr), z, 2*(n+1)*Sizeof.FLOAT, cudaMemcpyDeviceToHost);

        System.out.printf("Final results:\n");
        for (j=0; j<2; j++)
        {
            for (i=0; i<n-1; i++)
            {
                System.out.printf("z[%d,%d]=%f\n", i,j,zHostPtr[i+(n+1)*j]);
            }
        }
        if ((zHostPtr[0] !=    950.0) ||
            (zHostPtr[1] !=    400.0) ||
            (zHostPtr[2] !=   2550.0) ||
            (zHostPtr[3] !=   2600.0) ||
            (zHostPtr[4] !=      0.0) ||
            (zHostPtr[5] !=  49300.0) ||
            (zHostPtr[6] !=  15200.0) ||
            (zHostPtr[7] != 132300.0) ||
            (zHostPtr[8] != 131200.0) ||
            (zHostPtr[9] !=      0.0) ||
            (yHostPtr[0] !=     10.0) ||
            (yHostPtr[1] !=     20.0) ||
            (yHostPtr[2] !=     30.0) ||
            (yHostPtr[3] !=     40.0) ||
            (yHostPtr[4] !=    680.0) ||
            (yHostPtr[5] !=    760.0) ||
            (yHostPtr[6] !=   1230.0) ||
            (yHostPtr[7] !=   2240.0))
        {
            System.out.println("example test FAILED");
        }
        else
        {
            System.out.println("example test PASSED");
        }

        // Clean up
        cudaFree(y);
        cudaFree(z);
        cudaFree(xInd);
        cudaFree(xVal);
        cudaFree(csrRowPtr);
        cudaFree(cooRowIndex);
        cudaFree(cooColIndex);
        cudaFree(cooVal);
        cusparseDestroy(handle);
    }
}
