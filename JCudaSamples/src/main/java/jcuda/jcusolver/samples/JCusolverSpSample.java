/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcusolver.samples;

import static jcuda.jcusolver.JCusolverSp.cusolverSpCreate;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvchol;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvcholHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvluHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvqr;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDcsrlsvqrHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpDestroy;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrissymHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrpermHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrperm_bufferSizeHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrsymamdHost;
import static jcuda.jcusolver.JCusolverSp.cusolverSpXcsrsymrcmHost;
import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDcsrmv;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.jcusparse.JCusparse.cusparseDestroyMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import org.jcuda.matrix.d.DoubleMatrices;
import org.jcuda.matrix.d.DoubleMatrixDeviceCSR;
import org.jcuda.matrix.d.DoubleMatrixHostCSR;

import de.javagl.matrixmarketreader.CSR;
import de.javagl.matrixmarketreader.MatrixMarketCSR;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusolver.JCusolver;
import jcuda.jcusolver.cusolverSpHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.runtime.JCuda;

/**
 * This is a sample showing how to use JCusolverSp. It is based on the
 * CUSOLVER sample from NVIDIA, but heavily refactored. It uses the
 * "jcuda-matrix-utils" and "MatrixMarketReader" utility libraries
 * that are distributed with the JCuda samples.
 */
public class JCusolverSpSample
{
    /**
     * The tolerance value for the solvers, used to decide whether a
     * matrix is singular.
     */
    private static final double TOLERANCE = 1e-12;
    
    /**
     * Constant pointer to 1.0, used for the residual computations
     */
    private static final Pointer POINTER_TO_ONE = 
        Pointer.to(new double[] { 1.0 });
    
    /**
     * Constant pointer to -1.0, used for the residual computations
     */
    private static final Pointer POINTER_TO_MINUS_ONE = 
        Pointer.to(new double[] { -1.0 });
    
    /**
     * Reordering flag for CUSOLVER. In this sample, reordering is a 
     * dedicated step, and different reorderings may be performed 
     * manually, so this flag is always 0 (meaning that no reordering
     * should be done internally).  
     */
    private static final int REORDER = 0; 
    
    /**
     * Whether the norms that are computed during the evaluation should
     * be printed
     */
    private static boolean PRINT_NORMS = false;
    
    /**
     * The CUSOLVER handle
     */
    private static cusolverSpHandle handle;
    
    /**
     * The CUSPARSE handle
     */
    private static cusparseHandle cusparseHandle;

    /**
     * The matrix descriptor
     */
    private static cusparseMatDescr descriptor;
    
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     * @throws IOException If the sample input file cannot be read 
     */
    public static void main(String[] args) throws IOException
    {
        // Enable exceptions and omit subsequent error checks
        JCuda.setExceptionsEnabled(true);;
        JCusparse.setExceptionsEnabled(true);
        JCusolver.setExceptionsEnabled(true);
        
        // Read the input matrix, and print basic information
        String path = "src/main/resources/data/jcusolver/";
        String fullFileName = path + "lap2D_5pt_n100.mtx";
        //String fullFileName = path + "gr_900_900_crg.mtx";
        //String fullFileName = path + "lap3D_7pt_n20.mtx";

        System.out.println("Reading " + fullFileName);
        DoubleMatrixHostCSR inputMatrix = read(fullFileName);
        validate(inputMatrix);
        int rows = inputMatrix.getNumRows();
        int cols = inputMatrix.getNumCols();
        int nnz = inputMatrix.getNumNonZeros();
        System.out.println("Input matrix has " + 
            rows + " rows and " + 
            cols + " columns, with " + 
            nnz + " non-zero elements");        
        
        // Set up the internal handles
        setup();

        // Create the lists of reorderers and solvers that will be tested
        List<Reorderer> reorderers = 
            Arrays.asList(null, new ReordererAMD(), new ReordererRCM());
        
        List<SolverDevice> solverDevices =
            Arrays.asList(new SolverDeviceCholesky(), new SolverDeviceQR());
        
        List<SolverHost> solverHosts =
            Arrays.asList(new SolverHostCholesky(), 
                new SolverHostQR(), new SolverHostLU());
        
        for (Reorderer reorderer : reorderers)
        {
            DoubleMatrixHostCSR h_A = copy(inputMatrix);

            // Apply the reorderer
            if (reorderer != null)
            {
                System.out.println("Apply reorderer " + 
                    reorderer.getClass().getSimpleName());
                reorderer.reorder(handle, descriptor, h_A);
            }
            
            // Run tests with all host solvers
            for (SolverHost solverHost : solverHosts)
            {
                // Set up the vectors for the equation A * x = b
                double h_x[] = new double[cols];
                double h_b[] = new double[rows];
                for(int row = 0 ; row < rows ; row++)
                {
                    h_b[row] = 1.0;
                }
                runTestHost(solverHost, h_A, h_x, h_b);
            }
            
            // Run tests with all device solvers
            for (SolverDevice solverDevice : solverDevices)
            {
                // Set up the vectors for the equation A * x = b
                double h_x[] = new double[cols];
                double h_b[] = new double[rows];
                for(int row = 0 ; row < rows ; row++)
                {
                    h_b[row] = 1.0;
                }
                runTestDevice(solverDevice, h_A, h_x, h_b);
            }
        }
        
        // Clean up
        shutdown();
        
        System.out.println("Done");
    }
    
    
    /**
     * Read the MatrixMarket file with the given name, and return it as
     * a matrix in CSR format, stored in host memory
     * 
     * @param matrixMarketFileName The matrix market file name
     * @return The matrix
     * @throws IOException If the file cannot be read
     */
    private static DoubleMatrixHostCSR read(String matrixMarketFileName) 
        throws IOException
    {
        CSR csr = MatrixMarketCSR.readCSR(
            new FileInputStream(matrixMarketFileName));
        DoubleMatrixHostCSR matrix = 
            DoubleMatrices.createDoubleMatrixHostCSR(
                csr.numRows, csr.numCols, csr.values, 
                csr.rowPointers, csr.columnIndices);
        return matrix;
    }
    
    
    /**
     * Validate that the given matrix is a zero-based square matrix,
     * and throw an IllegalArgumentException if not
     * 
     * @param matrix The matrix
     * @throws IllegalArgumentException If the matrix is not valid
     */
    private static void validate(DoubleMatrixHostCSR matrix)
    {
        int[] rowPointers = matrix.getRowPointers();
        int base = rowPointers[0];
        if (base != 0)
        {
            throw new IllegalArgumentException("Input matrix must be 0-based");
        }
        if (matrix.getNumRows() != matrix.getNumCols())
        {
            throw new IllegalArgumentException(
                "Input matrix must be a square matrix, but it has " + 
                matrix.getNumRows() + " rows and " + matrix.getNumCols() + 
                " columns");
        }
    }
    
    /**
     * Return whether the host matrix in the given matrix is symmetric
     * 
     * @param handle The handle
     * @param descriptor The matrix descriptor
     * @param hostMatrix The matrix
     * @return Whether the matrix is symmetric
     */
    private static boolean isSymmetric(
        cusolverSpHandle handle, cusparseMatDescr descriptor, 
        DoubleMatrixHostCSR hostMatrix)
    {
        int isSymmetricArray[] = new int[] { -1 };
        cusolverSpXcsrissymHost(
            handle, 
            hostMatrix.getNumRows(), 
            hostMatrix.getNumNonZeros(), 
            descriptor, 
            Pointer.to(hostMatrix.getRowPointers()),
            Pointer.to(hostMatrix.getRowPointers())
                .withByteOffset(1 * Sizeof.INT),
            Pointer.to(hostMatrix.getColumnIndices()), 
            Pointer.to(isSymmetricArray));
        int isSymmetric = isSymmetricArray[0];
        return isSymmetric == 1;
    }
    
    /**
     * Check whether the given value indicates a singularity. The given value
     * is taken from the last parameter of the solver methods (i.e. from
     * the <code>singularity</code> array). If this value is not negative,
     * then it indicates the row in which a singularity was encountered. If
     * this is the case, then a warning will be printed. 
     * 
     * @param singularityRow The row
     */
    private static void checkSingularity(int singularityRow)
    {
        if (singularityRow >= 0)
        {
            System.out.println("WARNING: The matrix is singular at row " + 
                singularityRow + " under tolerance " + TOLERANCE);
        }
    }

    
    /**
     * Create all handles that are required in this sample
     */
    private static void setup()
    {
        // Create the CUSOLVER handle
        handle = new cusolverSpHandle();
        cusolverSpCreate(handle);
        
        // Create the CUSPARSE handle
        cusparseHandle = new cusparseHandle();
        cusparseCreate(cusparseHandle);

        // Create the matrix descriptor
        descriptor = new cusparseMatDescr();
        cusparseCreateMatDescr(descriptor);
        cusparseSetMatType(descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descriptor, CUSPARSE_INDEX_BASE_ZERO);
    }
    
    /**
     * Destroy all handles that have been created in the {@link #setup()} method
     */
    private static void shutdown()
    {
        cusparseDestroyMatDescr(descriptor);
        cusparseDestroy(cusparseHandle);
        cusolverSpDestroy(handle);
    }
    
    /**
     * Run a test for the given solver, solving the equation A * x = b,
     * and print the results.
     * 
     * @param solverHost The solver
     * @param h_A The matrix 
     * @param h_x The solution vector
     * @param h_b The right-hand side of the equation
     */
    private static void runTestHost(SolverHost solverHost, 
        DoubleMatrixHostCSR h_A, double h_x[], double h_b[])
    {
        // For the cholesky solver, check whether the matrix is symmetric
        if (solverHost instanceof SolverHostCholesky)
        {
            if (!isSymmetric(handle, descriptor, h_A))
            {
                System.out.println("Cholesky solver may only be " + 
                    "applied to symmetric matrices");
                return;
            }
        }
        
        // Execute the host solver
        long before = System.nanoTime();
        solverHost.solve(handle, descriptor, h_A, h_x, h_b);
        cudaDeviceSynchronize();
        long after = System.nanoTime();

        double residualNorm = 
            evaluateHost(cusparseHandle, descriptor, h_A, h_x, h_b);
        
        // Print the results
        double ms = (after - before) / 1e6;
        System.out.printf(Locale.ENGLISH, 
            "Solver %-25s, duration %10.3fms, residual norm = %e\n",
            solverHost.getClass().getSimpleName(), ms, residualNorm);
        
    }
    
    /**
     * Run a test for the given solver, solving the equation A * x = b,
     * and print the results.
     * 
     * @param solverDevice The solver
     * @param h_A The matrix 
     * @param h_x The solution vector
     * @param h_b The right-hand side of the equation
     */
    private static void runTestDevice(SolverDevice solverDevice, 
        DoubleMatrixHostCSR h_A, double h_x[], double h_b[])
    {
        int rows = h_A.getNumRows();
        int cols = h_A.getNumCols();
        
        // Copy the host matrix to the device
        DoubleMatrixDeviceCSR d_A =
            DoubleMatrices.createDoubleMatrixDeviceCSR(cusparseHandle, h_A);
        
        // Create the vectors for the equation A * x = b on the device
        Pointer d_x = new Pointer();
        cudaMalloc(d_x, Sizeof.DOUBLE * cols);

        Pointer d_b = new Pointer();
        cudaMalloc(d_b, Sizeof.DOUBLE * rows);
        cudaMemcpy(d_b, Pointer.to(h_b), Sizeof.DOUBLE * rows,
            cudaMemcpyHostToDevice);
        
        // Execute the device solver
        long before = System.nanoTime();
        solverDevice.solve(handle, descriptor, d_A, d_x, d_b);
        cudaDeviceSynchronize();
        long after = System.nanoTime();
        
        double residualNorm = 
            evaluateDevice(cusparseHandle, descriptor, d_A, d_x, d_b);

        // Print the results
        double ms = (after - before) / 1e6;
        System.out.printf(Locale.ENGLISH, 
            "Solver %-25s, duration %10.3fms, residual norm = %e\n",
            solverDevice.getClass().getSimpleName(), ms, residualNorm);
        
        // Clean up
        cudaFree(d_b);
        cudaFree(d_x);
        d_A.dispose();
    }
    
    
    /**
     * Abstract base class for classes that can reorder matrix columns
     */
    private static abstract class Reorderer
    {
        /**
         * Reorder the columns of the given matrix, in place. The exact
         * method for reordering is determined by the implementations.
         * 
         * @param handle The handle
         * @param descriptor The matrix descriptor
         * @param hostMatrix The matrix
         */
        void reorder(
            cusolverSpHandle handle, cusparseMatDescr descriptor,
            DoubleMatrixHostCSR hostMatrix)
        {
            int rows = hostMatrix.getNumRows();
            int cols = hostMatrix.getNumCols();
            int nnz = hostMatrix.getNumNonZeros();
            
            // Compute the permutation of the columns 
            int columnPermutation[] = new int[cols];
            computeColumnPermutation(
                handle, descriptor, hostMatrix, columnPermutation);

            // Compute the size of the buffer that will be required
            // by CUSPARSE to perform the permutation
            long permutationBufferSizeArray[] =  { -1 };
            cusolverSpXcsrperm_bufferSizeHost(
                handle, rows, cols, nnz, descriptor, 
                Pointer.to(hostMatrix.getRowPointers()), 
                Pointer.to(hostMatrix.getColumnIndices()),
                Pointer.to(columnPermutation), 
                Pointer.to(columnPermutation),
                permutationBufferSizeArray);
            byte permutationBuffer[] = 
                new byte[(int)permutationBufferSizeArray[0]];

            // Prepare the matrix that will receive the reordered results.
            // For a column permutation matrix Q, this will be B = Q*A*Q^T
            DoubleMatrixHostCSR hostB =
                DoubleMatrices.createDoubleMatrixHostCSR(rows, cols, nnz);
            System.arraycopy(hostMatrix.getRowPointers(), 0, 
                hostB.getRowPointers(), 0, rows + 1);
            System.arraycopy(hostMatrix.getColumnIndices(), 0, 
                hostB.getColumnIndices(), 0, nnz);
            
            // Compute the mapping from of the values of B from A. For each
            // non-zero value, this contains the index that value 'j' from
            // matrix A will have in the array of values of matrix B. It
            // is initialized with the identity mapping.
            int mapBfromA[] = new int[nnz];
            for (int j = 0; j < nnz; j++)
            {
                mapBfromA[j] = j;
            }
            cusolverSpXcsrpermHost(
                handle, rows, cols, nnz, descriptor,
                Pointer.to(hostB.getRowPointers()),
                Pointer.to(hostB.getColumnIndices()), 
                Pointer.to(columnPermutation),
                Pointer.to(columnPermutation), 
                Pointer.to(mapBfromA),
                Pointer.to(permutationBuffer));

            // Apply the mapping of values, B = A(mapBfromA)
            double valuesA[] = hostMatrix.getValues();
            double valuesB[] = hostB.getValues();
            for (int j = 0; j < nnz; j++)
            {
                valuesB[j] = valuesA[mapBfromA[j]];
            }

            // Finally, write the permuted values from B to A
            System.arraycopy(hostB.getRowPointers(), 0, 
                hostMatrix.getRowPointers(), 0, rows + 1);
            System.arraycopy(hostB.getColumnIndices(), 0, 
                hostMatrix.getColumnIndices(), 0, nnz);
            System.arraycopy(hostB.getValues(), 0, 
                hostMatrix.getValues(), 0, nnz);
        }
        
        /**
         * Compute the permutation for reordering the columns of the
         * host matrix in the given matrix
         * 
         * @param handle The {@link cusolverSpHandle}
         * @param descriptor The matrix descriptor
         * @param hostMatrix The matrix
         * @param columnPermutation The array (with length = number of columns)
         * that will contain the permutation
         */
        protected abstract void computeColumnPermutation(
            cusolverSpHandle handle, cusparseMatDescr descriptor,
            DoubleMatrixHostCSR hostMatrix,
            int columnPermutation[]);
    }
    
    /**
     * Implementation of a {@link Reorderer} that computes a symmetric reverse 
     * Cuthill McKee permutation of the columns.
     */
    private static class ReordererRCM extends Reorderer
    {
        @Override
        protected void computeColumnPermutation(
            cusolverSpHandle handle, cusparseMatDescr descriptor,
            DoubleMatrixHostCSR hostMatrix,
            int columnPermutation[])
        {
            cusolverSpXcsrsymrcmHost(handle, 
                hostMatrix.getNumRows(), 
                hostMatrix.getNumNonZeros(), 
                descriptor, 
                Pointer.to(hostMatrix.getRowPointers()), 
                Pointer.to(hostMatrix.getColumnIndices()),
                Pointer.to(columnPermutation));
        }
        
    }
    
    /**
     * Implementation of a {@link Reorderer} that computes a a permutation
     * of the matrix columns with a symmetric approximate minimum degree 
     * algorithm based on quotient graph. Whatever this means.
     */
    private static class ReordererAMD extends Reorderer
    {
        @Override
        protected void computeColumnPermutation(
            cusolverSpHandle handle, cusparseMatDescr descriptor,
            DoubleMatrixHostCSR hostMatrix,
            int columnPermutation[])
        {
            cusolverSpXcsrsymamdHost(handle, 
                hostMatrix.getNumRows(), 
                hostMatrix.getNumNonZeros(), 
                descriptor, 
                Pointer.to(hostMatrix.getRowPointers()), 
                Pointer.to(hostMatrix.getColumnIndices()),
                Pointer.to(columnPermutation));
        }
    }
    
    
    /**
     * Interface for a CUSOLVER based solver that operates on the device
     */
    private static interface SolverDevice
    {
        /**
         * Solves the equation A * x = b
         * 
         * @param handle The CUSOLVER handle
         * @param descriptor The matrix descriptor
         * @param d_A The matrix
         * @param d_x The vector that will contain the solution 
         * @param d_b The right-hand side of the equation
         */
        void solve(cusolverSpHandle handle, cusparseMatDescr descriptor,
            DoubleMatrixDeviceCSR d_A, Pointer d_x, Pointer d_b);
    }
    
    /**
     * Implementation of a {@link SolverDevice} using the Cholesky method
     */
    private static class SolverDeviceCholesky implements SolverDevice
    {
        @Override
        public void solve(cusolverSpHandle handle, cusparseMatDescr descriptor, 
            DoubleMatrixDeviceCSR d_A, Pointer d_x, Pointer d_b)
        {
            int singularityRowArray[] = { -1 };
            cusolverSpDcsrlsvchol(
                handle, 
                d_A.getNumRows(), 
                d_A.getNumNonZeros(), 
                descriptor, 
                d_A.getValues(), 
                d_A.getRowPointers(), 
                d_A.getColumnIndices(),
                d_b, 
                TOLERANCE, REORDER, 
                d_x, 
                singularityRowArray);
            checkSingularity(singularityRowArray[0]);
        }
    }
    
    /**
     * Implementation of a {@link SolverDevice} using the QR method
     */
    static class SolverDeviceQR implements SolverDevice
    {
        @Override
        public void solve(cusolverSpHandle handle, cusparseMatDescr descriptor, 
            DoubleMatrixDeviceCSR d_A, Pointer d_x, Pointer d_b)
        {
            int singularityRowArray[] = { -1 };
            cusolverSpDcsrlsvqr(
                handle, 
                d_A.getNumRows(), 
                d_A.getNumNonZeros(), 
                descriptor,
                d_A.getValues(), 
                d_A.getRowPointers(),
                d_A.getColumnIndices(), 
                d_b, 
                TOLERANCE, REORDER, 
                d_x,
                singularityRowArray);
            checkSingularity(singularityRowArray[0]);
        }
    }
    

    /**
     * Interface for a CUSOLVER based solver that operates on the host
     */
    private static interface SolverHost
    {
        /**
         * Solves the equation A * x = b
         * 
         * @param handle The CUSOLVER handle
         * @param descriptor The matrix descriptor
         * @param h_A The matrix
         * @param h_x The vector that will contain the solution 
         * @param h_b The right-hand side of the equation
         */
        void solve(cusolverSpHandle handle, cusparseMatDescr descriptor, 
            DoubleMatrixHostCSR h_A, double h_x[], double h_b[]);
    }
    
    
    /**
     * Implementation of the {@link SolverHost} using the Cholesky method
     */
    private static class SolverHostCholesky implements SolverHost
    { 
        @Override
        public void solve(cusolverSpHandle handle, cusparseMatDescr descriptor, 
            DoubleMatrixHostCSR h_A, double h_x[], double h_b[])
        {
            int rows = h_A.getNumRows();
            int nnz = h_A.getNumNonZeros();
            
            int singularityRowArray[] = { -1 };
            cusolverSpDcsrlsvcholHost(
                handle, rows, nnz, descriptor, 
                Pointer.to(h_A.getValues()), 
                Pointer.to(h_A.getRowPointers()), 
                Pointer.to(h_A.getColumnIndices()),
                Pointer.to(h_b), 
                TOLERANCE, REORDER, 
                Pointer.to(h_x), 
                singularityRowArray);
            checkSingularity(singularityRowArray[0]);
        }
    }
    
    
    /**
     * Implementation of the {@link SolverHost} using the LU method
     */
    private static class SolverHostLU implements SolverHost
    { 
        @Override
        public void solve(cusolverSpHandle handle, cusparseMatDescr descriptor, 
            DoubleMatrixHostCSR h_A, double h_x[], double h_b[])
        {
            int singularityRowArray[] = { -1 };
            cusolverSpDcsrlsvluHost(
                handle, 
                h_A.getNumRows(), 
                h_A.getNumNonZeros(), 
                descriptor, 
                Pointer.to(h_A.getValues()), 
                Pointer.to(h_A.getRowPointers()), 
                Pointer.to(h_A.getColumnIndices()),
                Pointer.to(h_b), 
                TOLERANCE, REORDER, 
                Pointer.to(h_x), 
                singularityRowArray);
            checkSingularity(singularityRowArray[0]);
        }
    }

    
    /**
     * Implementation of the {@link SolverHost} using the QR method
     */
    private static class SolverHostQR implements SolverHost
    { 
        @Override
        public void solve(cusolverSpHandle handle, cusparseMatDescr descriptor, 
            DoubleMatrixHostCSR h_A, double h_x[], double h_b[])
        {
            int singularityRowArray[] = { -1 };
            cusolverSpDcsrlsvqrHost(
                handle, 
                h_A.getNumRows(), 
                h_A.getNumNonZeros(), 
                descriptor, 
                Pointer.to(h_A.getValues()), 
                Pointer.to(h_A.getRowPointers()), 
                Pointer.to(h_A.getColumnIndices()),
                Pointer.to(h_b), 
                TOLERANCE, REORDER, 
                Pointer.to(h_x), 
                singularityRowArray);
            checkSingularity(singularityRowArray[0]);
        }
    }
    
    
    /**
     * Evaluate the results computed by a solver on the device.
     * 
     * @param cusparseHandle The handle
     * @param descriptor The matrix descriptor
     * @param d_A The device matrix
     * @param d_x The solution
     * @param d_b The right-hand side of A * x = b
     * @return The infinity norm of the residual
     */
    private static double evaluateDevice(
        cusparseHandle cusparseHandle, cusparseMatDescr descriptor, 
        DoubleMatrixDeviceCSR d_A, Pointer d_x, Pointer d_b)
    {
        int cols = d_A.getNumRows();

        // Copy the matrix to the device (for the norm computation)
        DoubleMatrixHostCSR h_A =
            DoubleMatrices.createDoubleMatrixHostCSR(d_A);
        
        // Copy x to the host
        double h_x[] = new double[cols];
        cudaMemcpy(Pointer.to(h_x), d_x, Sizeof.DOUBLE * cols,
            cudaMemcpyDeviceToHost);
        
        double h_r[] = computeResidual(
            cusparseHandle, descriptor, d_A, d_x, d_b);

        // Compute the infinity norms of x, r and A
        double xInf = infinityNorm(h_x);
        double rInf = infinityNorm(h_r);
        double AInf = infinityNorm(h_A);

        // Print the results
        if (PRINT_NORMS)
        {
            System.out.printf("    (GPU) |b - A*x| = %E \n", rInf);
            System.out.printf("    (GPU) |A| = %E \n", AInf);
            System.out.printf("    (GPU) |x| = %E \n", xInf);
            System.out.printf("    (GPU) |b - A*x|/(|A|*|x|) = %E \n",
                rInf / (AInf * xInf));
        }
        
        return rInf;
    }
    

    /**
     * Evaluate the results computed by a solver on the host.
     * 
     * @param cusparseHandle The handle
     * @param descriptor The matrix descriptor
     * @param h_A The host matrix
     * @param h_x The solution
     * @param h_b The right-hand side of A * x = b
     * @return The infinity norm of the residual
     */
    private static double evaluateHost(
        cusparseHandle cusparseHandle, cusparseMatDescr descriptor, 
        DoubleMatrixHostCSR h_A, double h_x[], double h_b[])
    {
        int rows = h_A.getNumRows();
        int cols = h_A.getNumRows();

        // Copy the matrix to the device (for the residual computation)
        DoubleMatrixDeviceCSR d_A =
            DoubleMatrices.createDoubleMatrixDeviceCSR(cusparseHandle, h_A);
        
        // Copy x = A \ b to the device
        Pointer d_x = new Pointer(); 
        cudaMalloc(d_x, Sizeof.DOUBLE * cols);
        cudaMemcpy(d_x, Pointer.to(h_x), Sizeof.DOUBLE * cols,
            cudaMemcpyHostToDevice);

        // Copy b to the device
        Pointer d_b = new Pointer(); 
        cudaMalloc(d_b, Sizeof.DOUBLE * rows);
        cudaMemcpy(d_b, Pointer.to(h_b), Sizeof.DOUBLE * rows,
            cudaMemcpyHostToDevice);
        
        double h_r[] = computeResidual(
            cusparseHandle, descriptor, d_A, d_x, d_b);
        
        // Compute the infinity norms of x, r and A
        double xInf = infinityNorm(h_x);
        double rInf = infinityNorm(h_r);
        double AInf = infinityNorm(h_A);

        // Print the results
        if (PRINT_NORMS)
        {
            System.out.printf("    (CPU) |b - A*x| = %E \n", rInf);
            System.out.printf("    (CPU) |A| = %E \n", AInf);
            System.out.printf("    (CPU) |x| = %E \n", xInf);
            System.out.printf("    (CPU) |b - A*x|/(|A|*|x|) = %E \n",
                rInf / (AInf * xInf));
        }
        
        // Clean up
        cudaFree(d_b);
        cudaFree(d_x);
        d_A.dispose();
        
        return rInf;
    }
    
    
    /**
     * Compute and return the residual of A * x = b
     * 
     * @param cusparseHandle The CUSPARSE handle
     * @param descriptor The matrix descriptor
     * @param d_A The matrix
     * @param d_x The solution
     * @param d_b The right hand side of the equation
     * @return The residual vector
     */
    private static double[] computeResidual(
        cusparseHandle cusparseHandle, cusparseMatDescr descriptor, 
        DoubleMatrixDeviceCSR d_A, Pointer d_x, Pointer d_b)
    {
        int rows = d_A.getNumRows();
        int cols = d_A.getNumRows();
        int nnz = d_A.getNumNonZeros();

        // Allocate the residual, r = b - A*x, and initialize it with b
        Pointer d_r = new Pointer();
        cudaMalloc(d_r, Sizeof.DOUBLE * rows);
        cudaMemcpy(d_r, d_b, Sizeof.DOUBLE * rows,
            cudaMemcpyDeviceToDevice);
        
        // Compute the residual
        cusparseDcsrmv(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            rows,
            cols,
            nnz,
            POINTER_TO_MINUS_ONE,
            descriptor,
            d_A.getValues(),
            d_A.getRowPointers(),
            d_A.getColumnIndices(),
            d_x,
            POINTER_TO_ONE,
            d_r);
        
        // Copy the residual to the host
        double h_r[] = new double[rows];
        cudaMemcpy(Pointer.to(h_r), d_r, Sizeof.DOUBLE * rows,
            cudaMemcpyDeviceToHost);
        
        // Clean up
        cudaFree(d_r);
        
        return h_r;
    }

    /**
     * Create a copy of the given matrix
     * 
     * @param matrix The matrix 
     * @return The copy
     */
    private static DoubleMatrixHostCSR copy(DoubleMatrixHostCSR matrix)
    {
        return DoubleMatrices.createDoubleMatrixHostCSR(
            matrix.getNumRows(),
            matrix.getNumCols(),
            matrix.getValues().clone(),
            matrix.getRowPointers().clone(),
            matrix.getColumnIndices().clone());
    }
    
    /**
     * Compute the infinity norm of the given array
     * 
     * @param x The array
     * @return The norm
     */
    private static double infinityNorm(double x[])
    {
        double norm = 0;
        for (int i = 0; i < x.length; i++)
        {
            norm = Math.max(norm, Math.abs(x[i]));
        }
        return norm;
    }

    /**
     * Compute the infinity norm of the given matrix
     * 
     * @param matrix The matrix
     * @return The norm
     */
    private static double infinityNorm(
        DoubleMatrixHostCSR matrix)
    {
        int rows = matrix.getNumRows();
        int rowPointers[] = matrix.getRowPointers();
        double values[] = matrix.getValues();
        double norm = 0;
        for (int i = 0; i < rows; i++)
        {
            double sum = 0.0;
            int start = rowPointers[i];
            int end = rowPointers[i + 1];
            for (int colidx = start; colidx < end; colidx++)
            {
                sum += Math.abs(values[colidx]);
            }
            norm = Math.max(norm, sum);
        }
        return norm;
    }


}
