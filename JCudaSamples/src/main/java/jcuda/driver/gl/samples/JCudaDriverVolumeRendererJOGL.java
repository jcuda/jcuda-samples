/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.gl.samples;

import static jcuda.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP;
import static jcuda.driver.CUarray_format.CU_AD_FORMAT_FLOAT;
import static jcuda.driver.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
import static jcuda.driver.CUfilter_mode.CU_TR_FILTER_MODE_LINEAR;
import static jcuda.driver.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD;
import static jcuda.driver.CUmemorytype.CU_MEMORYTYPE_ARRAY;
import static jcuda.driver.CUmemorytype.CU_MEMORYTYPE_HOST;
import static jcuda.driver.JCudaDriver.CU_TRSA_OVERRIDE_FORMAT;
import static jcuda.driver.JCudaDriver.CU_TRSF_NORMALIZED_COORDINATES;
import static jcuda.driver.JCudaDriver.cuArray3DCreate;
import static jcuda.driver.JCudaDriver.cuArrayCreate;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuGraphicsGLRegisterBuffer;
import static jcuda.driver.JCudaDriver.cuGraphicsMapResources;
import static jcuda.driver.JCudaDriver.cuGraphicsResourceGetMappedPointer;
import static jcuda.driver.JCudaDriver.cuGraphicsUnmapResources;
import static jcuda.driver.JCudaDriver.cuGraphicsUnregisterResource;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemcpy2D;
import static jcuda.driver.JCudaDriver.cuMemcpy3D;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemsetD32;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleGetGlobal;
import static jcuda.driver.JCudaDriver.cuModuleGetTexRef;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuTexRefSetAddressMode;
import static jcuda.driver.JCudaDriver.cuTexRefSetArray;
import static jcuda.driver.JCudaDriver.cuTexRefSetFilterMode;
import static jcuda.driver.JCudaDriver.cuTexRefSetFlags;
import static jcuda.driver.JCudaDriver.cuTexRefSetFormat;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.GridLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.FileInputStream;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.jogamp.opengl.GL;
import com.jogamp.opengl.GL2;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.GLProfile;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.Animator;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY3D_DESCRIPTOR;
import jcuda.driver.CUDA_ARRAY_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUDA_MEMCPY3D;
import jcuda.driver.CUarray;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.CUmodule;
import jcuda.driver.CUtexref;
import jcuda.driver.JCudaDriver;
import jcuda.driver.gl.samples.SimpleInteraction.MouseControl;
import jcuda.runtime.dim3;
import jcuda.samples.utils.JCudaSamplesUtils;

/**
 * A sample showing how to do volume rendering with JCuda. <br>
 * <br>
 * The program loads an 8 bit RAW volume data set and copies it into a 
 * 3D texture. The texture is accessed by the kernel to render an image
 * of the volume data. The resulting image is written into a pixel 
 * buffer object (PBO) which is then displayed using JOGL.<br>
 * <br>
 * (Note: This sample is heavily based on the NVIDIA CUDA "volumeRender" 
 * sample, and this uses parts of the OpenGL fixed function pipeline,
 * which is actually deprecated)
 */
public class JCudaDriverVolumeRendererJOGL implements GLEventListener
{
    /**
     * Entry point for this sample.
     * 
     * @param args not used
     */
    public static void main(String args[])
    {
        String fileName = "src/main/resources/data/driver/gl/Bucky.raw";
        startSample(fileName, 32, 32, 32);
        
        // Other input files may be obtained from http://www.volvis.org
        //startSample("mri_ventricles.raw", 256, 256, 124);
        //startSample("backpack8.raw", 512, 512, 373);
        //startSample("foot.raw", 256, 256, 256);
    }    
    
    /**
     * Starts this sample with the data that is read from the file
     * with the given name. The data is assumed to have the 
     * specified dimensions.
     * 
     * @param fileName The name of the volume data file to load
     * @param sizeX The size of the data set in X direction
     * @param sizeY The size of the data set in Y direction
     * @param sizeZ The size of the data set in Z direction
     */
    private static void startSample(
        String fileName, final int sizeX, final int sizeY, final int sizeZ)
    {
        // Try to read the specified file
        byte data[] = null;
        FileInputStream fis = null;
        try
        {
            fis = new FileInputStream(fileName);
            int size = sizeX * sizeY * sizeZ; 
            data = new byte[size];
            fis.read(data);
        }
        catch (IOException e)
        {
            System.err.println("Could not load input file");
            e.printStackTrace();
            return;
        }
        finally
        {
            if (fis != null)
            {
                try
                {
                    fis.close();
                }
                catch (IOException e)
                {
                    System.err.println("Could not close input file");
                    e.printStackTrace();
                    return;
                }
            }
        }
        
        // Start the sample with the data that was read from the file
        final byte volumeData[] = data;
        GLProfile profile = GLProfile.get(GLProfile.GL2);
        final GLCapabilities capabilities = new GLCapabilities(profile);
        SwingUtilities.invokeLater(new Runnable()
        {
            @Override
            public void run()
            {
                new JCudaDriverVolumeRendererJOGL(capabilities, 
                    volumeData, sizeX, sizeY, sizeZ);
            }
        });
    }

    /**
     * The GL component which is used for rendering
     */
    private GLCanvas glComponent;

    /**
     * The animator used for rendering
     */
    private Animator animator;
    
    /**
     * The CUDA module containing the kernel
     */
    private CUmodule module = new CUmodule();

    /**
     * The handle for the CUDA function of the kernel that is to be called
     */
    private CUfunction function;

    /**
     * The width of the rendered area and the PBO
     */
    private int width = 0;

    /**
     * The height of the rendered area and the PBO
     */
    private int height = 0;

    /**
     * The size of the volume data that is to be rendered
     */
    private dim3 volumeSize = new dim3();

    /**
     * The volume data that is to be rendered
     */
    private byte h_volume[];

    /**
     * The block size for the kernel execution
     */
    private dim3 blockSize = new dim3(16, 16, 1);

    /**
     * The grid size of the kernel execution
     */
    private dim3 gridSize = 
        new dim3(width / blockSize.x, height / blockSize.y, 1);

    /**
     * The global variable of the module which stores the
     * inverted view matrix.
     */
    private CUdeviceptr c_invViewMatrix = new CUdeviceptr();

    /**
     * The inverted view matrix, which will be copied to the global
     * variable of the kernel.
     */
    private float invViewMatrix[] = new float[12];

    /**
     * The density of the rendered volume data
     */
    private float density = 0.05f;

    /**
     * The brightness of the rendered volume data
     */
    private float brightness = 1.0f;

    /**
     * The transferOffset of the rendered volume data
     */
    private float transferOffset = 0.0f;

    /**
     * The transferScale of the rendered volume data
     */
    private float transferScale = 1.0f;

    /**
     * The OpenGL pixel buffer object
     */
    private int pbo = 0;

    /**
     * The CUDA graphics resource for the PBO
     */
    private CUgraphicsResource pboGraphicsResource = new CUgraphicsResource();
    
    /**
     * The 3D texture reference
     */
    private CUtexref tex = new CUtexref();

    /**
     * The 1D transfer texture reference
     */
    private CUtexref transferTex = new CUtexref();

    /**
     * Step counter for FPS computation
     */
    private int step = 0;
    
    /**
     * Time stamp for FPS computation
     */
    private long prevTimeNS = -1;
    
    /**
     * The main frame of the application
     */
    private Frame frame;
    
    /**
     * The interaction instance, encapsulating a VERY basic mouse interaction
     */
    private final SimpleInteraction simpleInteraction;

    /**
     * Creates a new JCudaTextureSample that displays the given volume
     * data, which has the specified size.
     * 
     * @param volumeData The volume data
     * @param sizeX The size of the data set in X direction
     * @param sizeY The size of the data set in Y direction
     * @param sizeZ The size of the data set in Z direction
     */
    public JCudaDriverVolumeRendererJOGL(GLCapabilities capabilities,
        byte volumeData[], int sizeX, int sizeY, int sizeZ)
    {
        this.simpleInteraction = new SimpleInteraction();
        
        h_volume = volumeData;
        volumeSize.x = sizeX;
        volumeSize.y = sizeY;
        volumeSize.z = sizeZ;

        width = 800;
        height = 800;

        // Initialize the GL component 
        glComponent = new GLCanvas(capabilities);
        glComponent.addGLEventListener(this);
        
        // Initialize the mouse controls
        MouseControl mouseControl = simpleInteraction.getMouseControl();
        glComponent.addMouseMotionListener(mouseControl);
        glComponent.addMouseWheelListener(mouseControl);

        // Create the main frame
        frame = new JFrame("JCuda 3D texture volume rendering sample");
        frame.addWindowListener(new WindowAdapter()
        {
            @Override
            public void windowClosing(WindowEvent e)
            {
                runExit();
            }
        });
        frame.setLayout(new BorderLayout());
        glComponent.setPreferredSize(new Dimension(width, height));
        frame.add(glComponent, BorderLayout.CENTER);
        frame.add(createControlPanel(), BorderLayout.SOUTH);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        
        // Create and start the animator
        animator = new Animator(glComponent);
        animator.setRunAsFastAsPossible(true);
        animator.start();
    }

    /**
     * Create the control panel containing the sliders for setting
     * the visualization parameters.
     * 
     * @return The control panel
     */
    private JPanel createControlPanel()
    {
        JPanel controlPanel = new JPanel(new GridLayout(2, 2));
        JPanel panel = null;
        JSlider slider = null;

        // Density
        panel = new JPanel(new GridLayout(1, 2));
        panel.add(new JLabel("Density:"));
        slider = new JSlider(0, 100, 5);
        slider.addChangeListener(new ChangeListener()
        {
            @Override
            public void stateChanged(ChangeEvent e)
            {
                JSlider source = (JSlider) e.getSource();
                float a = source.getValue() / 100.0f;
                density = a;
            }
        });
        slider.setPreferredSize(new Dimension(0, 0));
        panel.add(slider);
        controlPanel.add(panel);

        // Brightness
        panel = new JPanel(new GridLayout(1, 2));
        panel.add(new JLabel("Brightness:"));
        slider = new JSlider(0, 100, 10);
        slider.addChangeListener(new ChangeListener()
        {
            @Override
            public void stateChanged(ChangeEvent e)
            {
                JSlider source = (JSlider) e.getSource();
                float a = source.getValue() / 100.0f;
                brightness = a * 10;
            }
        });
        slider.setPreferredSize(new Dimension(0, 0));
        panel.add(slider);
        controlPanel.add(panel);

        // Transfer offset
        panel = new JPanel(new GridLayout(1, 2));
        panel.add(new JLabel("Transfer Offset:"));
        slider = new JSlider(0, 100, 55);
        slider.addChangeListener(new ChangeListener()
        {
            @Override
            public void stateChanged(ChangeEvent e)
            {
                JSlider source = (JSlider) e.getSource();
                float a = source.getValue() / 100.0f;
                transferOffset = (-0.5f + a) * 2;
            }
        });
        slider.setPreferredSize(new Dimension(0, 0));
        panel.add(slider);
        controlPanel.add(panel);

        // Transfer scale
        panel = new JPanel(new GridLayout(1, 2));
        panel.add(new JLabel("Transfer Scale:"));
        slider = new JSlider(0, 100, 10);
        slider.addChangeListener(new ChangeListener()
        {
            @Override
            public void stateChanged(ChangeEvent e)
            {
                JSlider source = (JSlider) e.getSource();
                float a = source.getValue() / 100.0f;
                transferScale = a * 10;
            }
        });
        slider.setPreferredSize(new Dimension(0, 0));
        panel.add(slider);
        controlPanel.add(panel);

        return controlPanel;
    }

    @Override
    public void init(GLAutoDrawable drawable)
    {
        // Perform the default GL initialization
        GL gl = drawable.getGL();
        gl.setSwapInterval(0);
        gl.glEnable(GL.GL_DEPTH_TEST);
        gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        setupView(drawable);
        
        // Initialize CUDA with the current volume data
        initCuda();

        // Initialize the OpenGL pixel buffer object
        initPBO(gl);
    }

    /**
     * Initialize CUDA and the 3D texture with the current volume data.
     */
    void initCuda()
    {
        // Initialize the JCudaDriver. Note that this has to be done from 
        // the same thread that will later use the JCudaDriver API.
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        CUcontext glCtx = new CUcontext();
        cuCtxCreate(glCtx, 0, dev);

        // Prepare the PTX file containing the kernel
        String ptxFileName = JCudaSamplesUtils.preparePtxFile(
            "src/main/resources/kernels/JCudaDriverVolumeRendererKernel.cu");
        
        // Load the PTX file containing the kernel
        cuModuleLoad(module, ptxFileName);

        // Obtain the global pointer to the inverted view matrix from 
        // the module
        cuModuleGetGlobal(c_invViewMatrix, new long[1], module,
            "c_invViewMatrix");
        
        // Obtain a function pointer to the kernel function. This function
        // will later be called in the display method of this 
        // GLEventListener.
        function = new CUfunction();
        cuModuleGetFunction(function, module, "d_render");

        
        // Initialize the data for the transfer function and the volume data
        CUarray d_transferFuncArray = new CUarray();
        CUarray d_volumeArray = new CUarray();

        // Create the 3D array that will contain the volume data
        // and will be accessed via the 3D texture
        CUDA_ARRAY3D_DESCRIPTOR allocateArray = new CUDA_ARRAY3D_DESCRIPTOR();
        allocateArray.Width = volumeSize.x;
        allocateArray.Height = volumeSize.y;
        allocateArray.Depth = volumeSize.z;
        allocateArray.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        allocateArray.NumChannels = 1;
        cuArray3DCreate(d_volumeArray, allocateArray);

        // Copy the volume data data to the 3D array
        CUDA_MEMCPY3D copy = new CUDA_MEMCPY3D();
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = Pointer.to(h_volume);
        copy.srcPitch = volumeSize.x;
        copy.srcHeight = volumeSize.y;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = d_volumeArray;
        copy.dstPitch = volumeSize.x;
        copy.dstHeight = volumeSize.y;
        copy.WidthInBytes = volumeSize.x;
        copy.Height = volumeSize.y;
        copy.Depth = volumeSize.z;
        cuMemcpy3D(copy);

        // Obtain the 3D texture reference for the volume data from 
        // the module, set its parameters and assign the 3D volume 
        // data array as its reference.
        cuModuleGetTexRef(tex, module, "tex");
        cuTexRefSetFilterMode(tex, CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(tex, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(tex, 1, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFormat(tex, CU_AD_FORMAT_UNSIGNED_INT8, 1);
        cuTexRefSetFlags(tex, CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetArray(tex, d_volumeArray, CU_TRSA_OVERRIDE_FORMAT);

        // The RGBA components of the transfer function texture 
        float transferFunc[] = new float[]
        { 
            0.0f, 0.0f, 0.0f, 0.0f, 
            1.0f, 0.0f, 0.0f, 1.0f, 
            1.0f, 0.5f, 0.0f, 1.0f, 
            1.0f, 1.0f, 0.0f, 1.0f, 
            0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 
            0.0f, 0.0f, 1.0f, 1.0f, 
            1.0f, 0.0f, 1.0f, 1.0f, 
            0.0f, 0.0f, 0.0f, 0.0f
        };

        // Create the 2D (float4) array that will contain the
        // transfer function data. 
        CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.Width = transferFunc.length / 4;
        ad.Height = 1;
        ad.NumChannels = 4;
        cuArrayCreate(d_transferFuncArray, ad);

        // Copy the transfer function data to the array  
        CUDA_MEMCPY2D copy2 = new CUDA_MEMCPY2D();
        copy2.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy2.srcHost = Pointer.to(transferFunc);
        copy2.srcPitch = transferFunc.length * Sizeof.FLOAT;
        copy2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy2.dstArray = d_transferFuncArray;
        copy2.WidthInBytes = transferFunc.length * Sizeof.FLOAT;
        copy2.Height = 1;
        cuMemcpy2D(copy2);

        // Obtain the transfer texture reference from the module, 
        // set its parameters and assign the transfer function  
        // array as its reference.
        cuModuleGetTexRef(transferTex, module, "transferTex");
        cuTexRefSetFilterMode(transferTex, CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(transferTex, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(transferTex, CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetFormat(transferTex, CU_AD_FORMAT_FLOAT, 4);
        cuTexRefSetArray(transferTex, 
            d_transferFuncArray, CU_TRSA_OVERRIDE_FORMAT);
    }

    /**
     * Creates a pixel buffer object (PBO) which stores the image that
     * is created by the kernel, and which will later be rendered 
     * by JOGL.
     * 
     * @param gl The GL context
     */
    private void initPBO(GL gl)
    {
        if (pbo != 0)
        {
            cuGraphicsUnregisterResource(pboGraphicsResource);
            gl.glDeleteBuffers(1, new int[]{ pbo }, 0);
            pbo = 0;
        }

        // Create and bind a pixel buffer object with the current 
        // width and height of the rendering component.
        int pboArray[] = new int[1];
        gl.glGenBuffers(1, pboArray, 0);
        pbo = pboArray[0];
        gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, pbo);
        gl.glBufferData(GL2.GL_PIXEL_UNPACK_BUFFER, 
            width * height * Sizeof.BYTE * 4, null, GL.GL_DYNAMIC_DRAW);
        gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);

        // Register the PBO for usage with CUDA
        cuGraphicsGLRegisterBuffer(pboGraphicsResource, pbo, 
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);

        // Calculate new grid size
        gridSize = new dim3(
            iDivUp(width, blockSize.x), 
            iDivUp(height, blockSize.y), 1);
    }

    /**
     * Integral division, rounding the result to the next highest integer.
     * 
     * @param a Dividend
     * @param b Divisor
     * @return a/b rounded to the next highest integer.
     */
    private static int iDivUp(int a, int b)
    {
        return ((a - 1) / b) + 1;
    }

    /**
     * Set up a default view for the given GLAutoDrawable
     * 
     * @param drawable The GLAutoDrawable to set the view for
     */
    private void setupView(GLAutoDrawable drawable)
    {
        GL2 gl = drawable.getGL().getGL2();

        int w = drawable.getSurfaceWidth();
        int h = drawable.getSurfaceHeight();
        gl.glViewport(0, 0, w, h);

        gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glLoadIdentity();

        gl.glMatrixMode(GL2.GL_PROJECTION);
        gl.glLoadIdentity();
        gl.glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    }

    /**
     * Call the kernel function, rendering the 3D volume data image
     * into the PBO
     */
    private void render()
    {
        // Map the PBO to get a CUDA device pointer
        CUdeviceptr d_output = new CUdeviceptr();
        cuGraphicsMapResources(
            1, new CUgraphicsResource[]{ pboGraphicsResource }, null);
        cuGraphicsResourceGetMappedPointer(
            d_output, new long[1], pboGraphicsResource);
        cuMemsetD32(d_output, 0, width * height);

        // Set up the execution parameters for the kernel:
        // - One pointer for the output that is mapped to the PBO
        // - Two ints for the width and height of the image to render
        // - Four floats for the visualization parameters of the renderer
        Pointer dOut = Pointer.to(d_output);
        Pointer pWidth = Pointer.to(new int[]{width});
        Pointer pHeight = Pointer.to(new int[]{height});
        Pointer pDensity = Pointer.to(new float[]{density});
        Pointer pBrightness = Pointer.to(new float[]{brightness});
        Pointer pTransferOffset = Pointer.to(new float[]{transferOffset});
        Pointer pTransferScale = Pointer.to(new float[]{transferScale});
        Pointer kernelParameters = Pointer.to(
            dOut, 
            pWidth, 
            pHeight,
            pDensity, 
            pBrightness,
            pTransferOffset,
            pTransferScale
        );
        
        // Call the kernel function.
        cuLaunchKernel(function,
            gridSize.x, gridSize.y, gridSize.z,
            blockSize.x, blockSize.y, blockSize.z,
            0, null, kernelParameters, null);
        cuCtxSynchronize();
        
        // Unmap buffer object
        cuGraphicsUnmapResources(
            1, new CUgraphicsResource[]{pboGraphicsResource}, null);
    }

    @Override
    public void display(GLAutoDrawable drawable)
    {
        GL2 gl = drawable.getGL().getGL2();

        // Use OpenGL to build view matrix
        float modelView[] = new float[16];
        gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glPushMatrix();
        gl.glLoadIdentity();
        gl.glRotatef(-simpleInteraction.getRotationDegX(), 1.0f, 0.0f, 0.0f);
        gl.glRotatef(-simpleInteraction.getRotationDegY(), 0.0f, 1.0f, 0.0f);
        gl.glTranslatef(
            -simpleInteraction.getTranslationX(), 
            -simpleInteraction.getTranslationY(), 
            -simpleInteraction.getTranslationZ());
        gl.glGetFloatv(GL2.GL_MODELVIEW_MATRIX, modelView, 0);
        gl.glPopMatrix();

        // Build the inverted view matrix
        invViewMatrix[0] = modelView[0];
        invViewMatrix[1] = modelView[4];
        invViewMatrix[2] = modelView[8];
        invViewMatrix[3] = modelView[12];
        invViewMatrix[4] = modelView[1];
        invViewMatrix[5] = modelView[5];
        invViewMatrix[6] = modelView[9];
        invViewMatrix[7] = modelView[13];
        invViewMatrix[8] = modelView[2];
        invViewMatrix[9] = modelView[6];
        invViewMatrix[10] = modelView[10];
        invViewMatrix[11] = modelView[14];

        // Copy the inverted view matrix to the global variable that
        // was obtained from the module. The inverted view matrix
        // will be used by the kernel during rendering.
        cuMemcpyHtoD(c_invViewMatrix, Pointer.to(invViewMatrix),
            invViewMatrix.length * Sizeof.FLOAT);

        // Render and fill the PBO with pixel data
        render();

        // Draw the image from the PBO
        gl.glClear(GL.GL_COLOR_BUFFER_BIT);
        gl.glDisable(GL.GL_DEPTH_TEST);
        gl.glRasterPos2i(0, 0);
        gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, pbo);
        gl.glDrawPixels(width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, 0);
        gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);

        // Update FPS information in main frame title
        step++;
        long currentTime = System.nanoTime();
        if (prevTimeNS == -1)
        {
            prevTimeNS = currentTime;
        }
        long diff = currentTime - prevTimeNS;
        if (diff > 1e9)
        {
            double fps = (diff / 1e9) * step;
            String t = "JCuda 3D texture volume rendering sample - ";
            t += String.format("%.2f", fps)+" FPS";
            frame.setTitle(t);
            prevTimeNS = currentTime;
            step = 0;
        }

    }

    @Override
    public void reshape(
        GLAutoDrawable drawable, int x, int y, int width, int height)
    {
        this.width = width;
        this.height = height;

        initPBO(drawable.getGL());

        setupView(drawable);
    }

    @Override
    public void dispose(GLAutoDrawable arg0)
    {
        // Not used
    }

    /**
     * Stops the animator and calls System.exit() in a new Thread.
     * (System.exit() may not be called synchronously inside one 
     * of the JOGL callbacks)
     */
    private void runExit()
    {
        new Thread(new Runnable()
        {
            @Override
            public void run()
            {
                animator.stop();
                System.exit(0);
            }
        }).start();
    }

}
