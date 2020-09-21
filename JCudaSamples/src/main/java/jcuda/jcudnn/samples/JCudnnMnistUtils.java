/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2020 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcudnn.samples;

import static jcuda.runtime.JCuda.cudaDeviceReset;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnDataType;

/**
 * Utility methods for the JCudnnMnist sample. These are mainly file IO 
 * methods for the sample files that contain the binary data of the 
 * trained network, and the images.
 */
class JCudnnMnistUtils
{
    static Pointer readBinaryFileAsDeviceDataUnchecked(
        String fileName, int dataType)
    {
        if (dataType == cudnnDataType.CUDNN_DATA_FLOAT)
        {
            float data[] = readBinaryFileAsFloatsUnchecked(fileName);
            return createDevicePointer(data);
        }
        if (dataType == cudnnDataType.CUDNN_DATA_DOUBLE)
        {
            float data[] = readBinaryFileAsFloatsUnchecked(fileName);
            double doubleData[] = toDouble(data);
            return createDevicePointer(doubleData);
        }
        throw new IllegalArgumentException(
            "Invalid data type: " + cudnnDataType.stringFor(dataType));
    }
    
    private static float[] readBinaryFileAsFloatsUnchecked(String fileName)
    {
        try
        {
            return readBinaryFileAsFloats(fileName);
        }
        catch (IOException e)
        {
            cudaDeviceReset();
            throw new CudaException("Could not read input file", e);
        }
    }
    
    private static float[] readBinaryFileAsFloats(String fileName) 
        throws IOException
    {
        FileInputStream fis = new FileInputStream(new File(fileName));
        byte data[] = readFully(fis);
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        float result[] = new float[fb.capacity()];
        fb.get(result);
        return result;
    }
    
    private static double[] toDouble(float array[])
    {
        double result[] = new double[array.length];
        for (int i = 0; i < array.length; i++)
        {
            result[i] = array[i];
        }
        return result;
    }

    private static byte[] readFully(InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[1024];
        while (true)
        {
            int n = inputStream.read(buffer);
            if (n < 0)
            {
                break;
            }
            baos.write(buffer, 0, n);
        }
        byte data[] = baos.toByteArray();
        return data;
    }

    static Pointer readImageDataUnchecked(String fileName, int dataType)
    {
        if (dataType == cudnnDataType.CUDNN_DATA_FLOAT)
        {
            float data[] = readImageDataAsFloatsUnchecked(fileName);
            return Pointer.to(data);
        }
        if (dataType == cudnnDataType.CUDNN_DATA_DOUBLE)
        {
            double data[] = readImageDataAsDoublesUnchecked(fileName);
            return Pointer.to(data);
        }
        throw new IllegalArgumentException(
            "Invalid data type: " + cudnnDataType.stringFor(dataType));
    }

    private static double[] readImageDataAsDoublesUnchecked(String fileName)
    {
        try
        {
            return readImageDataAsDoubles(fileName);
        }
        catch (IOException e)
        {
            cudaDeviceReset();
            throw new CudaException("Could not read input file", e);
        }
    }

    private static double[] readImageDataAsDoubles(String fileName) throws IOException
    {
        InputStream is = new FileInputStream(new File(fileName));
        byte data[] = readBinaryPortableGraymap8bitData(is);
        double imageData[] = new double[data.length];
        for (int i = 0; i < data.length; i++)
        {
            imageData[i] = (((int) data[i]) & 0xff) / 255.0;
        }
        return imageData;
    }
    
    private static float[] readImageDataAsFloatsUnchecked(String fileName)
    {
        try
        {
            return readImageDataAsFloats(fileName);
        }
        catch (IOException e)
        {
            cudaDeviceReset();
            throw new CudaException("Could not read input file", e);
        }
    }

    private static float[] readImageDataAsFloats(String fileName) throws IOException
    {
        InputStream is = new FileInputStream(new File(fileName));
        byte data[] = readBinaryPortableGraymap8bitData(is);
        float imageData[] = new float[data.length];
        for (int i = 0; i < data.length; i++)
        {
            imageData[i] = (((int) data[i]) & 0xff) / 255.0f;
        }
        return imageData;
    }
    
    @SuppressWarnings("deprecation")
    private static byte[] readBinaryPortableGraymap8bitData(
        InputStream inputStream) throws IOException
    {
        DataInputStream dis = new DataInputStream(inputStream);
        String line = null;
        boolean firstLine = true;
        Integer width = null;
        Integer maxBrightness = null;
        while (true)
        {
            // The DataInputStream#readLine is deprecated,
            // but for ASCII input, it is safe to use it
            line = dis.readLine();
            if (line == null)
            {
                break;
            }
            line = line.trim();
            if (line.startsWith("#"))
            {
                continue;
            }
            if (firstLine)
            {
                firstLine = false;
                if (!line.equals("P5"))
                {
                    throw new IOException(
                        "Data is not a binary portable " + 
                        "graymap (P5), but " + line);
                }
                else
                {
                    continue;
                }
            }
            if (width == null)
            {
                String tokens[] = line.split(" ");
                if (tokens.length < 2)
                {
                    throw new IOException(
                        "Expected dimensions, found " + line);
                }
                width = parseInt(tokens[0]);
            }
            else if (maxBrightness == null)
            {
                maxBrightness = parseInt(line);
                if (maxBrightness > 255)
                {
                    throw new IOException(
                        "Only 8 bit values supported. " + 
                        "Maximum value is " + maxBrightness);
                }
                break;
            }
        }
        byte data[] = readFully(inputStream);
        return data;
    }

    private static Integer parseInt(String s) throws IOException
    {
        try
        {
            return Integer.parseInt(s);
        }
        catch (NumberFormatException e)
        {
            throw new IOException(e);
        }
    }

    static void printDeviceVector(int size, Pointer d, int dataType)
    {
        if (dataType == cudnnDataType.CUDNN_DATA_FLOAT)
        {
            printFloatDeviceVector(size, d);
        }
        else if (dataType == cudnnDataType.CUDNN_DATA_DOUBLE)
        {
            printDoubleDeviceVector(size, d);
        }
        else
        {
            throw new IllegalArgumentException(
                "Invalid data type: " + cudnnDataType.stringFor(dataType));
        }
    }
    
    private static void printFloatDeviceVector(int size, Pointer d)
    {
        float h[] = new float[size];
        cudaDeviceSynchronize();
        cudaMemcpy(Pointer.to(h), d, size * Sizeof.FLOAT,
            cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++)
        {
            System.out.print(h[i] + " ");
        }
        System.out.println();
    }
    private static void printDoubleDeviceVector(int size, Pointer d)
    {
        double h[] = new double[size];
        cudaDeviceSynchronize();
        cudaMemcpy(Pointer.to(h), d, size * Sizeof.DOUBLE,
            cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++)
        {
            System.out.print(h[i] + " ");
        }
        System.out.println();
    }

    static int computeIndexOfMax(Pointer d, int length, int dataType)
    {
        if (dataType == cudnnDataType.CUDNN_DATA_FLOAT)
        {
            return computeIndexOfMaxFloat(d, length);
        }
        if (dataType == cudnnDataType.CUDNN_DATA_DOUBLE)
        {
            return computeIndexOfMaxDouble(d, length);
        }
        throw new IllegalArgumentException(
            "Invalid data type: " + cudnnDataType.stringFor(dataType));
    }

    private static int computeIndexOfMaxFloat(Pointer d, int length)
    {
        float result[] = new float[length];
        cudaMemcpy(Pointer.to(result), d, 
            length * Sizeof.FLOAT,
            cudaMemcpyDeviceToHost);
        int id = 0;
        for (int i = 1; i < length; i++)
        {
            if (result[id] < result[i])
                id = i;
        }
        return id;
    }
    
    private static int computeIndexOfMaxDouble(Pointer d, int length)
    {
        double result[] = new double[length];
        cudaMemcpy(Pointer.to(result), d, 
            length * Sizeof.DOUBLE,
            cudaMemcpyDeviceToHost);
        int id = 0;
        for (int i = 1; i < length; i++)
        {
            if (result[id] < result[i])
                id = i;
        }
        return id;
    }
    
    private static Pointer createDevicePointer(float data[])
    {
        int size = data.length * Sizeof.FLOAT;
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, size);
        cudaMemcpy(deviceData, Pointer.to(data), size, cudaMemcpyHostToDevice);
        return deviceData;
    }
    
    private static Pointer createDevicePointer(double data[])
    {
        int size = data.length * Sizeof.DOUBLE;
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, size);
        cudaMemcpy(deviceData, Pointer.to(data), size, cudaMemcpyHostToDevice);
        return deviceData;
    }

}
