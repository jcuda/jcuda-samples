/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2020 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jcudnn.samples;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasDgemv;
import static jcuda.jcublas.JCublas2.cublasSgemv;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcudnn.JCudnn.CUDNN_VERSION;
import static jcuda.jcudnn.JCudnn.cudnnActivationForward;
import static jcuda.jcudnn.JCudnn.cudnnAddTensor;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcudnn.JCudnn.cudnnCreateActivationDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateLRNDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcudnn.JCudnn.cudnnDestroyActivationDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyLRNDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnFindConvolutionForwardAlgorithm;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardAlgorithm_v7;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionNdForwardOutputDim;
import static jcuda.jcudnn.JCudnn.cudnnGetErrorString;
import static jcuda.jcudnn.JCudnn.cudnnGetPoolingNdForwardOutputDim;
import static jcuda.jcudnn.JCudnn.cudnnGetVersion;
import static jcuda.jcudnn.JCudnn.cudnnLRNCrossChannelForward;
import static jcuda.jcudnn.JCudnn.cudnnPoolingForward;
import static jcuda.jcudnn.JCudnn.cudnnSetActivationDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolutionNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilterNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetLRNDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetPoolingNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptorEx;
import static jcuda.jcudnn.JCudnn.cudnnSetTensorNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSoftmaxForward;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE;
import static jcuda.jcudnn.cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdAlgoPerf;
import jcuda.jcudnn.cudnnDataType;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnLRNDescriptor;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;

/**
 * A port of the "mnistCUDNN" sample.<br> 
 * <br>
 * This sample expects the data files that are part of the mnistCUDNN 
 * sample to be present in a "src/main/resources/data/jcudnn/mnist/" 
 * subdirectory.<br>
 * <br>
 * Additionally, this sample requires the cuDNN DLL to be present in
 * a known path (PATH on Windows, LD_LIBRARY_PATH on Linux). 
 */
public class JCudnnMnist
{
    private static final int IMAGE_H = 28;
    private static final int IMAGE_W = 28;

    private static final String dataDirectory = 
        "src/main/resources/data/jcudnn/mnist/";

    private static final String first_image = "one_28x28.pgm";
    private static final String second_image = "three_28x28.pgm";
    private static final String third_image = "five_28x28.pgm";

    private static final String conv1_bin = "conv1.bin";
    private static final String conv1_bias_bin = "conv1.bias.bin";
    private static final String conv2_bin = "conv2.bin";
    private static final String conv2_bias_bin = "conv2.bias.bin";
    private static final String ip1_bin = "ip1.bin";
    private static final String ip1_bias_bin = "ip1.bias.bin";
    private static final String ip2_bin = "ip2.bin";
    private static final String ip2_bias_bin = "ip2.bias.bin";

    public static void main(String args[])
    {
        JCuda.setExceptionsEnabled(true);
        JCudnn.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);

        int version = (int) cudnnGetVersion();
        System.out.printf("cudnnGetVersion() : %d , " + 
            "CUDNN_VERSION from cudnn.h : %d\n",
            version, CUDNN_VERSION);

        System.out.println("Creating network and layers...");
        int dataType = cudnnDataType.CUDNN_DATA_FLOAT;
        Network mnist = new Network(dataType);
        
        System.out.println("Classifying...");
        int i1 = mnist.classifyExample(dataDirectory + first_image);
        int i2 = mnist.classifyExample(dataDirectory + second_image);

        mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
        int i3 = mnist.classifyExample(dataDirectory + third_image);
        
        System.out.println(
            "\nResult of classification: " + i1 + " " + i2 + " " + i3);
        if (i1 != 1 || i2 != 3 || i3 != 5)
        {
            System.out.println("\nTest failed!\n");
        }
        else
        {
            System.out.println("\nTest passed!\n");
        }
        mnist.destroy();
    }

    
    // The CUDNN_TENSOR_NCHW tensor format specifies that the 
    // data is laid out in the following order: 
    // image, features map, rows, columns.
    private static class TensorLayout
    {
        int n;
        int c;
        int h;
        int w;
        
        TensorLayout(int n, int c, int h, int w)
        {
            this.n = n;
            this.c = c;
            this.h = h;
            this.w = w;
        }
    }
    
    private static class Layer
    {
        int inputs;
        int outputs;
        int kernel_dim;
        Pointer data_d;
        Pointer bias_d;

        Layer(int inputs, int outputs, int kernelDim, 
            String weightsFileName, String biasFileName,
            int dataType)
        {
            this.inputs = inputs;
            this.outputs = outputs;
            this.kernel_dim = kernelDim;

            String weightsPath = dataDirectory + weightsFileName;
            String biasPath = dataDirectory + biasFileName;

            data_d = JCudnnMnistUtils.readBinaryFileAsDeviceDataUnchecked(
                weightsPath, dataType);

            bias_d = JCudnnMnistUtils.readBinaryFileAsDeviceDataUnchecked(
                biasPath, dataType);
        }

        void destroy()
        {
            cudaFree(data_d);
            cudaFree(bias_d);
        }
    };
    
    // Demonstrate different ways of setting tensor descriptor.
    // This enum covers the #defines from the original sample.
    private enum TensorDescriptorType
    {
        NONE,
        SIMPLE_TENSOR_DESCRIPTOR,
        ND_TENSOR_DESCRIPTOR
    }
    private static final TensorDescriptorType tensorDescriptorType = 
        TensorDescriptorType.ND_TENSOR_DESCRIPTOR;


    private static void setTensorDesc(cudnnTensorDescriptor tensorDesc, 
        int tensorFormat, int dataType, TensorLayout t)
    {
        setTensorDesc(tensorDesc, tensorFormat, dataType, t.n, t.c, t.h, t.w);
    }
    
    private static void setTensorDesc(cudnnTensorDescriptor tensorDesc, 
        int tensorFormat, int dataType, int n, int c, int h, int w)
    {
        if (tensorDescriptorType == TensorDescriptorType.SIMPLE_TENSOR_DESCRIPTOR)
        {
            cudnnSetTensor4dDescriptor(tensorDesc,
                CUDNN_TENSOR_NCHW, dataType, n, c, h, w);
        }
        else if (tensorDescriptorType == TensorDescriptorType.ND_TENSOR_DESCRIPTOR)
        {
            int dimA[] = {n,c,h,w};
            int strideA[] = {c*h*w, h*w, w, 1};
            cudnnSetTensorNdDescriptor(tensorDesc, 
                dataType, 4, dimA, strideA);
        }
        else
        {
            cudnnSetTensor4dDescriptorEx(tensorDesc,
                dataType, n, c, h, w, c*h*w, h*w, w, 1);
        }
    }
    
    
    private static class Network
    {
        private int convAlgorithm; // cudnnConvolutionFwdAlgo_t
        private int dataType; // cudnnDataType_t
        private int tensorFormat; // cudnnTensorFormat_t
        private cudnnHandle cudnnHandle;
        private cudnnTensorDescriptor srcTensorDesc;
        private cudnnTensorDescriptor dstTensorDesc;
        private cudnnTensorDescriptor biasTensorDesc;
        private cudnnFilterDescriptor filterDesc;
        private cudnnConvolutionDescriptor convDesc;
        private cudnnPoolingDescriptor poolingDesc;
        private cudnnActivationDescriptor activDesc;
        private cudnnLRNDescriptor normDesc;
        private cublasHandle cublasHandle;
        
        private final Layer conv1;
        private final Layer conv2;
        private final Layer ip1;
        private final Layer ip2;
        
        private int dataTypeSize;
        private Pointer pointerToOne;
        private Pointer pointerToZero;

        Network(int valueType)
        {
            convAlgorithm = -1;
            dataType = valueType;
            if (dataType == cudnnDataType.CUDNN_DATA_FLOAT)
            {
                dataTypeSize = Sizeof.FLOAT;
                pointerToOne = pointerTo(1.0f);
                pointerToZero = pointerTo(0.0f);
            }
            else if (dataType == cudnnDataType.CUDNN_DATA_DOUBLE)
            {
                dataTypeSize = Sizeof.DOUBLE;
                pointerToOne = pointerTo(1.0);
                pointerToZero = pointerTo(0.0);
            }
            else
            {
                throw new IllegalArgumentException(
                    "Invalid data type: " + cudnnDataType.stringFor(valueType));
            }
            
            tensorFormat = CUDNN_TENSOR_NCHW;
            createHandles();
            
            conv1 = new Layer(1, 20, 5, conv1_bin, conv1_bias_bin, dataType);
            conv2 = new Layer(20, 50, 5, conv2_bin, conv2_bias_bin, dataType);
            ip1 = new Layer(800, 500, 1, ip1_bin, ip1_bias_bin, dataType);
            ip2 = new Layer(500, 10, 1, ip2_bin, ip2_bias_bin, dataType);
        }

        void createHandles()
        {
            cudnnHandle = new cudnnHandle();
            srcTensorDesc = new cudnnTensorDescriptor();
            dstTensorDesc = new cudnnTensorDescriptor();
            biasTensorDesc = new cudnnTensorDescriptor();
            filterDesc = new cudnnFilterDescriptor();
            convDesc = new cudnnConvolutionDescriptor();
            poolingDesc = new cudnnPoolingDescriptor();
            activDesc = new cudnnActivationDescriptor();
            normDesc = new cudnnLRNDescriptor();

            cudnnCreate(cudnnHandle);
            cudnnCreateTensorDescriptor(srcTensorDesc);
            cudnnCreateTensorDescriptor(dstTensorDesc);
            cudnnCreateTensorDescriptor(biasTensorDesc);
            cudnnCreateFilterDescriptor(filterDesc);
            cudnnCreateConvolutionDescriptor(convDesc);
            cudnnCreatePoolingDescriptor(poolingDesc);
            cudnnCreateActivationDescriptor(activDesc);
            cudnnCreateLRNDescriptor(normDesc);

            cublasHandle = new cublasHandle();
            cublasCreate(cublasHandle);
        }

        void destroy()
        {
            cudnnDestroyLRNDescriptor(normDesc);
            cudnnDestroyPoolingDescriptor(poolingDesc);
            cudnnDestroyActivationDescriptor(activDesc);
            cudnnDestroyConvolutionDescriptor(convDesc);
            cudnnDestroyFilterDescriptor(filterDesc);
            cudnnDestroyTensorDescriptor(srcTensorDesc);
            cudnnDestroyTensorDescriptor(dstTensorDesc);
            cudnnDestroyTensorDescriptor(biasTensorDesc);
            cudnnDestroy(cudnnHandle);

            cublasDestroy(cublasHandle);
            
            conv1.destroy();
            conv2.destroy();
            ip1.destroy();
            ip2.destroy();
        }


        void setConvolutionAlgorithm(int algo)
        {
            convAlgorithm = algo;
        }

        void addBias(cudnnTensorDescriptor dstTensorDesc, 
            Layer layer, int c, Pointer data)
        {
            setTensorDesc(biasTensorDesc, tensorFormat, 
                dataType, new TensorLayout(1, c, 1, 1));
            Pointer alpha = pointerToOne;
            Pointer beta = pointerToOne;
            cudnnAddTensor(cudnnHandle, alpha,
                biasTensorDesc, layer.bias_d, beta, dstTensorDesc, data);
        }

        void fullyConnectedForward(Layer ip, TensorLayout t, 
            Pointer srcData, Pointer dstData)
        {
            if (t.n != 1)
            {
                System.out.println("Not Implemented");
                return;
            }
            int dim_x = t.c * t.h * t.w;
            int dim_y = ip.outputs;
            resize(dim_y, dataTypeSize, dstData);

            Pointer alpha = pointerToOne;
            Pointer beta = pointerToOne;

            // place bias into dstData
            cudaMemcpy(dstData, ip.bias_d, dim_y * dataTypeSize,
                cudaMemcpyDeviceToDevice);

            if (dataType == CUDNN_DATA_FLOAT) 
            {
                cublasSgemv(cublasHandle, CUBLAS_OP_T, 
                    dim_x, dim_y, alpha, ip.data_d, 
                    dim_x, srcData, 1, beta, dstData, 1);
            }
            else
            {
                cublasDgemv(cublasHandle, CUBLAS_OP_T, 
                    dim_x, dim_y, alpha, ip.data_d, 
                    dim_x, srcData, 1, beta, dstData, 1);
            }

            t.h = 1;
            t.w = 1;
            t.c = dim_y;
        }

        void convoluteForward(Layer conv, TensorLayout t,
            Pointer srcData, Pointer dstData)
        {
            int algo = 0; // cudnnConvolutionFwdAlgo_t

            setTensorDesc(srcTensorDesc, 
                tensorFormat, dataType, t);

            int tensorDims = 4;
            int tensorOuputDimA[] = { t.n, t.c, t.h, t.w };
            int filterDimA[] = { 
                conv.outputs, conv.inputs, 
                conv.kernel_dim, conv.kernel_dim };
            
            // TODO The CUDNN_TENSOR_NCHW was used in the sample. It should
            // probably be 'tensorFormat', but it does not make a difference:
            cudnnSetFilterNdDescriptor(filterDesc, 
                dataType, CUDNN_TENSOR_NCHW, tensorDims, filterDimA);

            int convDims = 2;
            int padA[] = { 0, 0 };
            int filterStrideA[] = { 1, 1 };
            int upscaleA[] = { 1, 1 };
            cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA,
                filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, dataType);

            // find dimension of convolution output
            cudnnGetConvolutionNdForwardOutputDim(convDesc, 
                srcTensorDesc, filterDesc, 
                tensorDims, tensorOuputDimA);
            t.n = tensorOuputDimA[0];
            t.c = tensorOuputDimA[1];
            t.h = tensorOuputDimA[2];
            t.w = tensorOuputDimA[3];

            setTensorDesc(dstTensorDesc, 
                tensorFormat, dataType, t);

            if (convAlgorithm < 0)
            {
                int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT; 
                int returnedAlgoCount = -1;
                int returnedAlgoCountArray[] = { returnedAlgoCount }; 
                cudnnConvolutionFwdAlgoPerf results[] = 
                    new cudnnConvolutionFwdAlgoPerf[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

                // Choose the best according to the preference
                System.out.println("Testing cudnnGetConvolutionForwardAlgorithm_v7 ...");
                cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
                    srcTensorDesc, filterDesc, convDesc,
                    dstTensorDesc, requestedAlgoCount,
                    returnedAlgoCountArray,
                    results);
                returnedAlgoCount = returnedAlgoCountArray[0];    
                for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex)
                {
                    String errorString = cudnnGetErrorString(results[algoIndex].status);
                    System.out.printf("^^^^ %s for Algo %d: %f time requiring %d memory\n", 
                        errorString, results[algoIndex].algo, results[algoIndex].time, 
                        (long)results[algoIndex].memory);
                }

                // New way of finding the fastest config
                // Setup for findFastest call
                System.out.println("Testing cudnnFindConvolutionForwardAlgorithm ...");
                cudnnFindConvolutionForwardAlgorithm(cudnnHandle, 
                    srcTensorDesc, filterDesc, convDesc,
                    dstTensorDesc, requestedAlgoCount,
                    returnedAlgoCountArray, results);
                returnedAlgoCount = returnedAlgoCountArray[0];    
                for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex)
                {
                    String errorString = cudnnGetErrorString(results[algoIndex].status);
                    System.out.printf("^^^^ %s for Algo %d: %f time requiring %d memory\n", 
                        errorString, 
                        results[algoIndex].algo, results[algoIndex].time, 
                        (long)results[algoIndex].memory);
                }
                algo = results[0].algo;            
            } 
            else 
            {
                algo = convAlgorithm;
            }

            resize(t.n * t.c * t.h * t.w, dataTypeSize, dstData);
            long sizeInBytes = 0;
            long sizeInBytesArray[] = { sizeInBytes };
            Pointer workSpace = new Pointer();
            cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, 
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc, 
                algo, sizeInBytesArray);
            sizeInBytes = sizeInBytesArray[0];
            if (sizeInBytes != 0)
            {
                cudaMalloc(workSpace, sizeInBytes);
            }
            
            Pointer alpha = pointerToOne;
            Pointer beta = pointerToZero;
            cudnnConvolutionForward(cudnnHandle, alpha, srcTensorDesc, 
                srcData, filterDesc, conv.data_d, convDesc, algo, 
                workSpace, sizeInBytes, beta, dstTensorDesc, dstData);
            addBias(dstTensorDesc, conv, t.c, dstData);
            if (sizeInBytes != 0)
            {
                cudaFree(workSpace);
            }
        }

        void poolForward(TensorLayout t, Pointer srcData,
            Pointer dstData)
        {
            int poolDims = 2;
            int windowDimA[] = { 2, 2 };
            int paddingA[] = { 0, 0 };
            int strideA[] = { 2, 2 };
            cudnnSetPoolingNdDescriptor(poolingDesc, 
                CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, poolDims, windowDimA, 
                paddingA, strideA);

            setTensorDesc(srcTensorDesc, 
                tensorFormat, dataType, t);

            int tensorDims = 4;
            int tensorOuputDimA[] = { t.n, t.c, t.h, t.w };
            cudnnGetPoolingNdForwardOutputDim(
                poolingDesc, srcTensorDesc,
                tensorDims, tensorOuputDimA);
            t.n = tensorOuputDimA[0];
            t.c = tensorOuputDimA[1];
            t.h = tensorOuputDimA[2];
            t.w = tensorOuputDimA[3];

            setTensorDesc(dstTensorDesc, 
                tensorFormat, dataType, t);

            resize(t.n * t.c * t.h * t.w, dataTypeSize, dstData);
            Pointer alpha = pointerToOne;
            Pointer beta = pointerToZero;
            cudnnPoolingForward(cudnnHandle, poolingDesc, 
                alpha, srcTensorDesc, srcData, beta, 
                dstTensorDesc, dstData);
        }

        void softmaxForward(TensorLayout t,
            Pointer srcData, Pointer dstData)
        {
            resize(t.n * t.c * t.h * t.w, dataTypeSize, dstData);

            setTensorDesc(srcTensorDesc, tensorFormat, dataType, t);
            setTensorDesc(dstTensorDesc, tensorFormat, dataType, t);
            
            Pointer alpha = pointerToOne;
            Pointer beta = pointerToZero;
            cudnnSoftmaxForward(cudnnHandle, 
                CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, 
                alpha, srcTensorDesc, srcData,
                beta, dstTensorDesc, dstData);
        }

        void lrnForward(TensorLayout t, 
            Pointer srcData, Pointer dstData)
        {
            int lrnN = 5;
            double lrnAlpha, lrnBeta, lrnK;
            lrnAlpha = 0.0001;
            lrnBeta = 0.75;
            lrnK = 1.0;
            cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);

            resize(t.n * t.c * t.h * t.w, dataTypeSize, dstData);

            setTensorDesc(srcTensorDesc, tensorFormat, dataType, t);
            setTensorDesc(dstTensorDesc, tensorFormat, dataType, t);
            
            Pointer alpha = pointerToOne;
            Pointer beta = pointerToZero;
            cudnnLRNCrossChannelForward(cudnnHandle, normDesc,
                CUDNN_LRN_CROSS_CHANNEL_DIM1, 
                alpha, srcTensorDesc, srcData,
                beta, dstTensorDesc, dstData);
        }

        
        void activationForward(TensorLayout t,
            Pointer srcData, Pointer dstData)
        {
            cudnnSetActivationDescriptor(activDesc, 
                CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
            
            resize(t.n * t.c * t.h * t.w, dataTypeSize, dstData);

            setTensorDesc(srcTensorDesc, tensorFormat, dataType, t);
            setTensorDesc(dstTensorDesc, tensorFormat, dataType, t);

            Pointer alpha = pointerTo(1.0f);
            Pointer beta = pointerTo(0.0f);
            cudnnActivationForward(cudnnHandle, activDesc, 
                alpha, srcTensorDesc, srcData, 
                beta, dstTensorDesc, dstData);
        }

        
        int classifyExample(String imageFileName)
        {
            TensorLayout t = new TensorLayout(0,0,0,0);
            Pointer srcData = new Pointer();
            Pointer dstData = new Pointer();

            Pointer imgData_h = 
                JCudnnMnistUtils.readImageDataUnchecked(imageFileName, dataType);

            System.out.println("Performing forward propagation ...");

            cudaMalloc(srcData, IMAGE_H * IMAGE_W * dataTypeSize);
            cudaMemcpy(srcData, imgData_h, IMAGE_H * IMAGE_W
                * dataTypeSize, cudaMemcpyHostToDevice);

            t.n = 1;
            t.c = 1;
            t.h = IMAGE_H;
            t.w = IMAGE_W;
            convoluteForward(conv1, t, srcData, dstData);
            poolForward(t, dstData, srcData);

            convoluteForward(conv2, t, srcData, dstData);
            poolForward(t, dstData, srcData);

            fullyConnectedForward(ip1, t, srcData, dstData);
            activationForward(t, dstData, srcData);
            lrnForward(t, srcData, dstData);

            fullyConnectedForward(ip2, t, dstData, srcData);
            softmaxForward(t, srcData, dstData);

            int max_digits = 10;
            int id = JCudnnMnistUtils.computeIndexOfMax(
                dstData, max_digits, dataType);
            
            System.out.println("Resulting weights from Softmax:");
            JCudnnMnistUtils.printDeviceVector(
                t.n * t.c * t.h * t.w, dstData, dataType);

            cudaFree(srcData);
            cudaFree(dstData);
            return id;
        }
    }


    private static void resize(int numberOfElements, int dataTypeSize, Pointer data)
    {
        cudaFree(data);
        cudaMalloc(data, numberOfElements * dataTypeSize);
    }
    
    private static Pointer pointerTo(float value)
    {
        return Pointer.to(new float[] { value });
    }
    private static Pointer pointerTo(double value)
    {
        return Pointer.to(new double[] { value });
    }
    
    
    
}
