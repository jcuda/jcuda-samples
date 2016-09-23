/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.samples.utils;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.JCudaDriver.cuCtxGetDevice;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.Random;
import java.util.logging.Logger;

import jcuda.CudaException;
import jcuda.driver.CUdevice;
import jcuda.driver.CUresult;

/**
 * Utility methods that are used in the JCuda samples.<br>
 * <br>
 * NOTE: This class is not part of a public API. It is only intended for
 * the use in the samples. Parts of its functionality could be replaced 
 * with the runtime compilation features that have been added in CUDA 7.5. 
 */
public class JCudaSamplesUtils
{
    /**
     * The logger used in this class
     */
    private static final Logger logger =
        Logger.getLogger(JCudaSamplesUtils.class.getName());

    /**
     * Compiles the given CUDA file into a PTX file using NVCC, and returns
     * the name of the resulting PTX file
     * 
     * @param cuFileName The CUDA file name
     * @return The PTX file name
     * @throws CudaException If an error occurs - i.e. when the input file
     * does not exist, or the NVCC call caused an error.
     */
    public static String preparePtxFile(String cuFileName)
    {
        return invokeNvcc(cuFileName, "ptx", true);
    }

    /**
     * Compiles the given CUDA file into a CUBIN file using NVCC, and returns
     * the name of the resulting CUBIN file. By default, the NVCC will be
     * invoked with the <code>-dlink</code> parameter, and an 
     * <code>-arch</code> parameter for the compute capability of the
     * device of the current context.<br>
     * <br>
     * Note that there must be a current context when this function 
     * is called!
     * 
     * @param cuFileName The CUDA file name
     * @return The PTX file name
     * @throws CudaException If an error occurs - i.e. when the input file
     * does not exist, or the NVCC call caused an error.
     * @throws CudaException If there is no current context
     */
    public static String prepareDefaultCubinFile(String cuFileName)
    {
        int computeCapability = computeComputeCapability();
        String nvccArguments[] = new String[] {
            "-dlink",
            "-arch=sm_"+computeCapability
        };
        return invokeNvcc(cuFileName, "cubin", true, nvccArguments);
    }

    /**
     * Tries to create a PTX or CUBIN file for the given CUDA file. <br>
     * <br>
     * The extension of the given file name is replaced with 
     * <code>"cubin"</code> or <code>"ptx"</code>, depending on the 
     * <code>targetFileType</code>.<br>
     * <br>
     * If the file with the resulting name does not exist yet, or if 
     * <code>forceRebuild</code> is <code>true</code>, then it is compiled 
     * from the given file using NVCC, using the given parameters.<br>
     * <br>
     * The name of the resulting output file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @param targetFileType The target file type. Must be <code>"cubin"</code>
     * or <code>"ptx"</code> (case-insensitively)
     * @param forceRebuild Whether the PTX file should be created even if
     * it already exists
     * @return The name of the PTX file
     * @throws CudaException If an error occurs - i.e. when the input file
     * does not exist, or the NVCC call caused an error.
     * @throws IllegalArgumentException If the target file type is not valid
     */
    private static String invokeNvcc(
        String cuFileName, String targetFileType, 
        boolean forceRebuild, String ... nvccArguments)
    {
        if (!"cubin".equalsIgnoreCase(targetFileType) &&
            !"ptx".equalsIgnoreCase(targetFileType))
        {
            throw new IllegalArgumentException(
                "Target file type must be \"ptx\" or \"cubin\", but is " + 
                    targetFileType);
        }
        logger.info("Creating " + targetFileType + " file for " + cuFileName);

        int dotIndex = cuFileName.lastIndexOf('.');
        if (dotIndex == -1)
        {
            dotIndex = cuFileName.length();
        }
        String otuputFileName = cuFileName.substring(0, dotIndex) + 
            "." + targetFileType.toLowerCase();
        File ptxFile = new File(otuputFileName);
        if (ptxFile.exists() && !forceRebuild)
        {
            return otuputFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new CudaException("Input file not found: " + cuFileName + 
                " (" + cuFile.getAbsolutePath() + ")");
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command = "nvcc ";
        command += modelString + " ";
        command += "-" + targetFileType + " ";
        for (String a : nvccArguments)
        {
            command += a + " ";
        }
        command += cuFileName + " -o " + otuputFileName;

        logger.info("Executing\n" + command);
        try
        {
            Process process = Runtime.getRuntime().exec(command);

            String errorMessage = 
                new String(toByteArray(process.getErrorStream()));
            String outputMessage =
                new String(toByteArray(process.getInputStream()));
            int exitValue = 0;
            try
            {
                exitValue = process.waitFor();
            }
            catch (InterruptedException e)
            {
                Thread.currentThread().interrupt();
                throw new CudaException(
                    "Interrupted while waiting for nvcc output", e);
            }
            if (exitValue != 0)
            {
                logger.severe("nvcc process exitValue " + exitValue);
                logger.severe("errorMessage:\n" + errorMessage);
                logger.severe("outputMessage:\n" + outputMessage);
                throw new CudaException("Could not create " + targetFileType + 
                    " file: " + errorMessage);
            }
        }
        catch (IOException e)
        {
            throw new CudaException("Could not create " + targetFileType + 
                " file", e);
        }

        logger.info("Finished creating " + targetFileType + " file");
        return otuputFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    /**
     * Compute the compute capability of the device device of the current
     * context. The compute capability will be returned as an int value 
     * <code>major * 10 + minor</code>. For example, the return value
     * will be <code>52</code> for a device with compute capability 5.2.
     * 
     * @return The compute capability of the current device
     * @throws CudaException If there is no current context
     */
    private static int computeComputeCapability()
    {
        CUdevice device = new CUdevice();
        int status = cuCtxGetDevice(device);
        if (status != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(CUresult.stringFor(status));
        }
        return computeComputeCapability(device);
    }


    /**
     * Compute the compute capability of the given device. The compute 
     * capability will be returned as an int value 
     * <code>major * 10 + minor</code>. For example, the return value
     * will be <code>52</code> for a device with compute capability 5.2.
     * 
     * @param device The device
     * @return The compute capability
     */
    private static int computeComputeCapability(CUdevice device)
    {
        int majorArray[] = { 0 };
        int minorArray[] = { 0 };
        cuDeviceGetAttribute(majorArray,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(minorArray,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        int major = majorArray[0];
        int minor = minorArray[0];
        return major * 10 + minor;
    }


    /**
     * Creates an array of the specified size, containing float values from
     * the range [0.0f, 1.0f)
     * 
     * @param n The size of the array
     * @return The array of random values
     */
    public static float[] createRandomFloatData(int n)
    {
        Random random = new Random(0);
        float a[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = random.nextFloat();
        }
        return a;
    }

    /**
     * Compares the given result against a reference, and returns whether the
     * error norm is below a small epsilon threshold
     * 
     * @param result The result
     * @param reference The reference
     * @return Whether the arrays are equal based on the error norm
     * @throws NullPointerException If any argument is <code>null</code>
     * @throws IllegalArgumentException If the arrays have different lengths
     */
    public static boolean equalByNorm(float result[], float reference[])
    {
        if (result == null)
        {
            throw new NullPointerException("The result is null");
        }
        if (reference == null)
        {
            throw new NullPointerException("The reference is null");
        }
        if (result.length != reference.length)
        {
            throw new IllegalArgumentException(
                "The result and reference array have different lengths: " + 
                    result.length + " and " + reference.length);
        }
        final float epsilon = 1e-6f;
        float errorNorm = 0;
        float refNorm = 0;
        for (int i = 0; i < result.length; ++i)
        {
            float diff = reference[i] - result[i];
            errorNorm += diff * diff;
            refNorm += reference[i] * result[i];
        }
        errorNorm = (float) Math.sqrt(errorNorm);
        refNorm = (float) Math.sqrt(refNorm);
        if (Math.abs(refNorm) < epsilon)
        {
            return false;
        }
        return (errorNorm / refNorm < epsilon);
    }
    
    
    /**
     * Creates a string representation of the given array as a matrix with 
     * with given number of columns.
     * 
     * @param a The array
     * @param columns The number of columns
     * @return The string representation
     */
    public static String toString2D(float[] a, int columns)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++)
        {
            if ((i > 0) && (i % columns == 0))
            {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }
    


}
