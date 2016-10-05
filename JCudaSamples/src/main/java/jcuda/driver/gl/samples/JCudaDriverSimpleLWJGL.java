/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.gl.samples;

import static jcuda.driver.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuGraphicsGLRegisterBuffer;
import static jcuda.driver.JCudaDriver.cuGraphicsMapResources;
import static jcuda.driver.JCudaDriver.cuGraphicsResourceGetMappedPointer;
import static jcuda.driver.JCudaDriver.cuGraphicsUnmapResources;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static org.lwjgl.opengl.GL11.GL_COLOR_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.GL_DEPTH_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.GL_DEPTH_TEST;
import static org.lwjgl.opengl.GL11.GL_FLOAT;
import static org.lwjgl.opengl.GL11.GL_POINTS;
import static org.lwjgl.opengl.GL11.glClear;
import static org.lwjgl.opengl.GL11.glClearColor;
import static org.lwjgl.opengl.GL11.glDrawArrays;
import static org.lwjgl.opengl.GL11.glEnable;
import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL15.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL15.GL_READ_WRITE;
import static org.lwjgl.opengl.GL15.glBindBuffer;
import static org.lwjgl.opengl.GL15.glBufferData;
import static org.lwjgl.opengl.GL15.glGenBuffers;
import static org.lwjgl.opengl.GL15.glMapBuffer;
import static org.lwjgl.opengl.GL15.glUnmapBuffer;
import static org.lwjgl.opengl.GL20.GL_FRAGMENT_SHADER;
import static org.lwjgl.opengl.GL20.GL_VERTEX_SHADER;
import static org.lwjgl.opengl.GL20.glAttachShader;
import static org.lwjgl.opengl.GL20.glCompileShader;
import static org.lwjgl.opengl.GL20.glCreateProgram;
import static org.lwjgl.opengl.GL20.glCreateShader;
import static org.lwjgl.opengl.GL20.glDeleteShader;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glGetAttribLocation;
import static org.lwjgl.opengl.GL20.glGetUniformLocation;
import static org.lwjgl.opengl.GL20.glLinkProgram;
import static org.lwjgl.opengl.GL20.glShaderSource;
import static org.lwjgl.opengl.GL20.glUniformMatrix4;
import static org.lwjgl.opengl.GL20.glUseProgram;
import static org.lwjgl.opengl.GL20.glValidateProgram;
import static org.lwjgl.opengl.GL20.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import org.lwjgl.LWJGLException;
import org.lwjgl.opengl.AWTGLCanvas;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.driver.gl.samples.SimpleInteraction.MouseControl;
import jcuda.samples.utils.JCudaSamplesUtils;


/**
 * This class demonstrates how to use the JCudaDriver GL bindings API 
 * to interact with LWJGL, the Java Lightweight Game Library. It creates
 * a vertex buffer object (VBO) consisting of a rectangular grid of
 * points, and animates it with a sine wave.<br> 
 * <br>
 * Pressing the 't' key will toggle between the CUDA computation and
 * the Java computation mode.<br>
 * <br>
 * This sample uses the kernel from the "Simple OpenGL" NVIDIA CUDA sample 
 */
public class JCudaDriverSimpleLWJGL
{
    /**
     * Entry point for this sample.
     * 
     * @param args not used
     */
    public static void main(String args[])
    {
        SwingUtilities.invokeLater(new Runnable()
        {
            public void run()
            {
                new JCudaDriverSimpleLWJGL();
            }
        });
    }

    /**
     * The source code for the vertex shader
     */
    private static String vertexShaderSource = 
        "#version 330 core" + "\n" +
        "in  vec4 inVertex;" + "\n" +
        "in  vec3 inColor;" + "\n" +
        "uniform mat4 modelviewMatrix;" + "\n" +
        "uniform mat4 projectionMatrix;" + "\n" +
        "void main(void)" + "\n" +
        "{" + "\n" +
        "    gl_Position = " + "\n" +
        "        projectionMatrix * modelviewMatrix * inVertex;" + "\n" +
        "}";
    
    /**
     * The source code for the fragment shader
     */
    private static String fragmentShaderSource =
        "#version 330 core" + "\n" +
        "out vec4 outColor;" + "\n" +
        "void main(void)" + "\n" +
        "{" + "\n" +
        "    outColor = vec4(1.0,0.0,0.0,1.0);" + "\n" +
        "}";
    
    /**
     * The width segments of the mesh to be displayed.
     * Should be a multiple of 8.
     */
    private static final int meshWidth = 8 * 64;

    /**
     * The height segments of the mesh to be displayed
     * Should be a multiple of 8.
     */
    private static final int meshHeight = 8 * 64;

    /**
     * The VAO identifier
     */
    private int vertexArrayObject;
    
    /**
     * The VBO identifier
     */
    private int vertexBufferObject;

    /**
     * The Graphics resource associated with the VBO 
     */
    private CUgraphicsResource vboGraphicsResource;
    
    /**
     * The current animation state of the mesh
     */
    private float animationState = 0.0f;

    /**
     * The GL canvas
     */
    private AWTGLCanvas glComponent;

    /**
     * The handle for the CUDA function of the kernel that is to be called
     */
    private CUfunction function;

    /**
     * Whether the computation should be performed with CUDA or
     * with Java. May be toggled by pressing the 't' key.
     */
    private boolean useCUDA = true;

    /**
     * The ID of the OpenGL shader program
     */
    private int shaderProgramID;
    
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
     * The currently mapped VBO data buffer
     */
    private ByteBuffer mappedBuffer; 
    
    /**
     * A direct FloatBuffer for 16-element matrices
     */
    private final FloatBuffer matrixBuffer = ByteBuffer
        .allocateDirect(16 * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer();
    
    /**
     * Inner class extending a KeyAdapter for the keyboard
     * interaction
     */
    class KeyboardControl extends KeyAdapter
    {
        public void keyTyped(KeyEvent e)
        {
            char c = e.getKeyChar();
            if (c == 't')
            {
                useCUDA = !useCUDA;
            }
        }
    }

    /**
     * Creates a new JCudaDriverSimpleLWJGL.
     */
    public JCudaDriverSimpleLWJGL()
    {
        simpleInteraction = new SimpleInteraction();
        
        // Initialize the GL component
        createCanvas();

        // Initialize the mouse and keyboard controls
        MouseControl mouseControl = simpleInteraction.getMouseControl();
        glComponent.addMouseMotionListener(mouseControl);
        glComponent.addMouseWheelListener(mouseControl);
        KeyboardControl keyboardControl = new KeyboardControl();
        glComponent.addKeyListener(keyboardControl);

        // Create the main frame 
        frame = new JFrame("JCuda / LWJGL interaction sample");
        frame.addWindowListener(new WindowAdapter()
        {
            @Override
            public void windowClosing(WindowEvent e)
            {
                System.exit(0);
            }
        });
        frame.setLayout(new BorderLayout());
        glComponent.setPreferredSize(new Dimension(800, 800));
        frame.add(glComponent, BorderLayout.CENTER);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        glComponent.requestFocus();
    }
    
    /**
     * Create the AWTGLCanvas 
     */
    private void createCanvas()
    {
        try
        {
            glComponent = new AWTGLCanvas()
            {
                /**
                 * Serial UID
                 */
                private static final long serialVersionUID = 1L;
                
                /**
                 * Whether 
                 */
                private boolean initialized = false;
                private Dimension previousSize = null;

                public void paintGL()
                {
                    if (!initialized)
                    {
                        init();
                        glComponent.setVSyncEnabled(false);
                        initialized = true;
                    }
                    if (previousSize == null || !previousSize.equals(getSize()))
                    {
                        previousSize = getSize();
                        setupView();
                    }
                    render();
                    try
                    {
                        swapBuffers();
                    }
                    catch (LWJGLException e)
                    {
                        throw new RuntimeException(
                            "Could not swap buffers", e);
                    }
                }
            };
        }
        catch (LWJGLException e)
        {
            throw new RuntimeException(
                "Could not create canvas", e);
        }
        glComponent.setFocusable(true);

        // Create the thread that triggers a repaint of the component
        Thread thread = new Thread(new Runnable()
        {
            @Override
            public void run()
            {
                while (true)
                {
                    glComponent.repaint();
                    try
                    {
                        Thread.sleep(1);
                    }
                    catch (InterruptedException e)
                    {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        });
        thread.setDaemon(true);
        thread.start();
    }
    

    /**
     * Default initialization of JCuda and OpenGL
     */
    private void init()
    {
        // Perform the default GL initialization 
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        setupView();

        // Initialize the shaders
        initShaders();

        // Initialize JCuda
        initJCuda();

        // Create the VBO containing the vertex data
        initVBO();
    }

    /**
     * Set up a default view for the given GLAutoDrawable
     * 
     * @param drawable The GLAutoDrawable to set the view for
     */
    private void setupView()
    {
        int w = glComponent.getWidth();
        int h = glComponent.getHeight();
        glViewport(0, 0, w, h);
        simpleInteraction.updateProjectionMatrix(w, h);
    }
    
    
    /**
     * Initialize the shaders and the shader program
     */
    private void initShaders()
    {
        shaderProgramID = glCreateProgram();

        int vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShaderID, vertexShaderSource);
        glCompileShader(vertexShaderID);
        glAttachShader(shaderProgramID, vertexShaderID);
        glDeleteShader(vertexShaderID);

        int fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShaderID, fragmentShaderSource);
        glCompileShader(fragmentShaderID);
        glAttachShader(shaderProgramID, fragmentShaderID);
        glDeleteShader(fragmentShaderID);
        
        glLinkProgram(shaderProgramID);
        glValidateProgram(shaderProgramID);
    }
    
    /**
     * Initialize the JCudaDriver. Note that this has to be done from the
     * same thread that will later use the JCudaDriver API
     */
    private void initJCuda()
    {
        JCudaDriver.setExceptionsEnabled(true);

        // Create a device and a context
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Prepare the PTX file containing the kernel
        String ptxFileName = JCudaSamplesUtils.preparePtxFile(
            "src/main/resources/kernels/JCudaDriverSimpleGLKernel.cu");
        
        // Load the PTX file containing the kernel
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the kernel function. This function
        // will later be called during the animation, in the display 
        // method of this GLEventListener.
        function = new CUfunction();
        cuModuleGetFunction(function, module, "simple_vbo_kernel");
    }
    

    
    /**
     * Create the vertex buffer object (VBO) that stores the
     * vertex positions.
     */
    private void initVBO()
    {
        // Create the vertex array object
        vertexArrayObject = glGenVertexArrays();

        glBindVertexArray(vertexArrayObject);
        
        // Create the vertex buffer object
        vertexBufferObject = glGenBuffers();

        // Initialize the vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
        int size = meshWidth * meshHeight * 4 * Sizeof.FLOAT;
        glBufferData(GL_ARRAY_BUFFER, size, GL_DYNAMIC_DRAW);

        // Initialize the attribute location of the input
        // vertices for the shader program
        int location = glGetAttribLocation(shaderProgramID, "inVertex");
        glVertexAttribPointer(location, 4, GL_FLOAT, false, 0, 0);
        glEnableVertexAttribArray(location);

        // Register the vertexBufferObject for use with CUDA
        vboGraphicsResource = new CUgraphicsResource();
        cuGraphicsGLRegisterBuffer(
            vboGraphicsResource, vertexBufferObject, 
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
    }
    

    /**
     * Update the vertex buffer, and render its current state
     */
    public void render()
    {
        if (useCUDA)
        {
            // Run the CUDA kernel to generate new vertex positions.
            runCuda();
        }
        else
        {
            // Run the Java method to generate new vertex positions.
            runJava();
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate the shader program
        glUseProgram(shaderProgramID);
        
        // Set the current projection matrix
        int projectionMatrixLocation = 
            glGetUniformLocation(shaderProgramID, "projectionMatrix");
        matrixBuffer.rewind();
        matrixBuffer.put(simpleInteraction.getProjectionMatrix());
        matrixBuffer.rewind();
        glUniformMatrix4(projectionMatrixLocation, false, matrixBuffer);
        
        // Set the current modelview matrix
        int modelviewMatrixLocation = 
            glGetUniformLocation(shaderProgramID, "modelviewMatrix");
        matrixBuffer.rewind();
        matrixBuffer.put(simpleInteraction.getModelviewMatrix());
        matrixBuffer.rewind();
        glUniformMatrix4(modelviewMatrixLocation, false, matrixBuffer);
        
        // Render the VBO
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
        glDrawArrays(GL_POINTS, 0, meshWidth * meshHeight);
        
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
            String t = "JCuda / LWJGL interaction sample - ";
            t += useCUDA ? "JCuda" : "Java";
            t += " mode: " + String.format("%.2f", fps) + " FPS";
            frame.setTitle(t);
            prevTimeNS = currentTime;
            step = 0;
        }

        animationState += 0.01;
    }

    /**
     * Run the CUDA computation to create new vertex positions
     * inside the vertexBufferObject.
     */
    private void runCuda()
    {
        // Map the vertexBufferObject for writing from CUDA.
        // The basePointer will afterwards point to the
        // beginning of the memory area of the VBO.
        CUdeviceptr basePointer = new CUdeviceptr();
        cuGraphicsMapResources(
            1, new CUgraphicsResource[]{vboGraphicsResource}, null);
        cuGraphicsResourceGetMappedPointer(
            basePointer, new long[1], vboGraphicsResource);
        
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values. One 
        // pointer to the base pointer of the geometry data, 
        // one int for the mesh width, one int for the mesh 
        // height, and one float for the current animation state. 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(basePointer),
            Pointer.to(new int[]{meshWidth}),
            Pointer.to(new int[]{meshHeight}),
            Pointer.to(new float[]{animationState})
        );

        // Call the kernel function.
        int blockX = 8;
        int blockY = 8;
        int gridX = meshWidth / blockX;
        int gridY = meshHeight / blockY;
        cuLaunchKernel(function,
            gridX, gridY, 1,       // Grid dimension
            blockX, blockY, 1,     // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        // Unmap buffer object
        cuGraphicsUnmapResources(
            1, new CUgraphicsResource[]{vboGraphicsResource}, null);
    }

    
    /**
     * Run the Java computation to create new vertex positions
     * inside the vertexBufferObject.
     */
    private void runJava()
    {
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
        mappedBuffer = 
            glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE, mappedBuffer);
        FloatBuffer vertices = 
            mappedBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer();
        for (int x = 0; x < meshWidth; x++)
        {
            for (int y = 0; y < meshHeight; y++)
            {
                // Calculate u/v coordinates
                float u = x / (float) meshWidth;
                float v = y / (float) meshHeight;

                u = u * 2.0f - 1.0f;
                v = v * 2.0f - 1.0f;

                // Calculate simple sine wave pattern
                float freq = 4.0f;
                float w = (float) Math.sin(u * freq + animationState) *
                          (float) Math.cos(v * freq + animationState) * 0.5f;

                // Write output vertex
                int index = 4 * (y * meshWidth + x);
                vertices.put(index + 0, u);
                vertices.put(index + 1, w);
                vertices.put(index + 2, v);
                vertices.put(index + 3, 1);
            }
        }
        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    
}
