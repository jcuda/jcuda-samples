/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcufft.samples;

import static jcuda.jcufft.JCufft.CUFFT_FORWARD;
import static jcuda.jcufft.JCufft.cufftDestroy;
import static jcuda.jcufft.JCufft.cufftExecC2C;
import static jcuda.jcufft.JCufft.cufftPlan1d;

import org.jtransforms.fft.FloatFFT_1D;

import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;
import jcuda.samples.utils.JCudaSamplesUtils;

/**
 * This is a sample class that performs a 1D Complex-To-Complex 
 * forward FFT with JCufft, and compares the result to the 
 * reference computed with JTransforms.
 */
class JCufftSample
{
    public static void main(String args[])
    {
        testC2C1D(1<<20);
    }

    /**
     * Test the 1D C2C transform with the given size.
     * 
     * @param size The size of the transform
     */
    private static void testC2C1D(int size)
    {
        System.out.println("Creating input data...");
        float input[] = JCudaSamplesUtils.createRandomFloatData(size * 2);

        System.out.println("Performing 1D C2C transform with JTransforms...");
        float outputJTransforms[] = input.clone();
        FloatFFT_1D fft = new FloatFFT_1D(size);
        fft.complexForward(outputJTransforms);

        System.out.println("Performing 1D C2C transform with JCufft...");
        float outputJCufft[] = input.clone();
        cufftHandle plan = new cufftHandle();
        cufftPlan1d(plan, size, cufftType.CUFFT_C2C, 1);
        cufftExecC2C(plan, outputJCufft, outputJCufft, CUFFT_FORWARD);
        cufftDestroy(plan);

        boolean passed = JCudaSamplesUtils.equalByNorm(
            outputJTransforms, outputJCufft);
        System.out.println("testC2C1D " + (passed ? "PASSED" : "FAILED"));
    }

}
