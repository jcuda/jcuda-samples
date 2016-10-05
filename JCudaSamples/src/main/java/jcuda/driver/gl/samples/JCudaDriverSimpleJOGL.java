/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.gl.samples;

import static com.jogamp.opengl.GL.GL_ARRAY_BUFFER;
import static com.jogamp.opengl.GL.GL_COLOR_BUFFER_BIT;
import static com.jogamp.opengl.GL.GL_DEPTH_BUFFER_BIT;
import static com.jogamp.opengl.GL.GL_DEPTH_TEST;
import static com.jogamp.opengl.GL.GL_DYNAMIC_DRAW;
import static com.jogamp.opengl.GL.GL_FLOAT;
import static com.jogamp.opengl.GL.GL_POINTS;
import static com.jogamp.opengl.GL2ES2.GL_FRAGMENT_SHADER;
import static com.jogamp.opengl.GL2ES2.GL_VERTEX_SHADER;
import static com.jogamp.opengl.GL2ES3.GL_READ_WRITE;
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

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import com.jogamp.opengl.GL;
import com.jogamp.opengl.GL3;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.GLProfile;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.Animator;

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
 * to interact with JOGL, the Java Bindings for OpenGL. It creates
 * a vertex buffer object (VBO) consisting of a rectangular grid of
 * points, and animates it with a sine wave.<br> 
 * <br>
 * Pressing the 't' key will toggle between the CUDA computation and
 * the Java computation mode.<br>
 * <br>
 * This sample uses the kernel from the "Simple OpenGL" NVIDIA CUDA sample 
 */
public class JCudaDriverSimpleJOGL implements GLEventListener
{
    /**
     * Entry point for this sample.
     * 
     * @param args not used
     */
    public static void main(String args[])
    {
        GLProfile profile = GLProfile.get(GLProfile.GL3);
        final GLCapabilities capabilities = new GLCapabilities(profile);
        SwingUtilities.invokeLater(new Runnable()
        {
            public void run()
            {
                new JCudaDriverSimpleJOGL(capabilities);
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
     * The animator used to animate the mesh.
     */
    private Animator animator;

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
     * Creates a new JCudaDriverSimpleJOGL.
     * 
     * @param The JOGL OpenGL capabilities
     */
    public JCudaDriverSimpleJOGL(GLCapabilities capabilities)
    {
        simpleInteraction = new SimpleInteraction();
        
        // Initialize the GL component and the animator
        GLCanvas glComponent = new GLCanvas(capabilities);
        glComponent.setFocusable(true);
        glComponent.addGLEventListener(this);

        // Initialize the mouse and keyboard controls
        MouseControl mouseControl = simpleInteraction.getMouseControl();
        glComponent.addMouseMotionListener(mouseControl);
        glComponent.addMouseWheelListener(mouseControl);
        KeyboardControl keyboardControl = new KeyboardControl();
        glComponent.addKeyListener(keyboardControl);

        // Create the main frame 
        frame = new JFrame("JCuda / JOGL interaction sample");
        frame.addWindowListener(new WindowAdapter()
        {
            @Override
            public void windowClosing(WindowEvent e)
            {
                runExit();
            }
        });
        frame.setLayout(new BorderLayout());
        glComponent.setPreferredSize(new Dimension(800, 800));
        frame.add(glComponent, BorderLayout.CENTER);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        glComponent.requestFocus();

        // Create and start the animator
        animator = new Animator(glComponent);
        animator.setRunAsFastAsPossible(false);
        animator.start();
        
    }

    @Override
    public void init(GLAutoDrawable drawable)
    {
        // Perform the default GL initialization 
        GL3 gl = drawable.getGL().getGL3();
        gl.setSwapInterval(0);
        gl.glEnable(GL_DEPTH_TEST);
        gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        setupView(drawable);

        // Initialize the shaders
        initShaders(gl);

        // Initialize JCuda
        initJCuda();

        // Create the VBO containing the vertex data
        initVBO(gl);
    }

    /**
     * Set up a default view for the given GLAutoDrawable
     * 
     * @param drawable The GLAutoDrawable to set the view for
     */
    private void setupView(GLAutoDrawable drawable)
    {
        GL3 gl = drawable.getGL().getGL3();
        int w = drawable.getSurfaceWidth();
        int h = drawable.getSurfaceHeight();
        gl.glViewport(0, 0, w, h);
        simpleInteraction.updateProjectionMatrix(w, h);
    }
    
    
    /**
     * Initialize the shaders and the shader program
     * 
     * @param gl The GL context
     */
    private void initShaders(GL3 gl)
    {
        shaderProgramID = gl.glCreateProgram();

        int vertexShaderID = gl.glCreateShader(GL_VERTEX_SHADER);
        gl.glShaderSource(vertexShaderID, 1, 
            new String[]{vertexShaderSource}, null);
        gl.glCompileShader(vertexShaderID);
        gl.glAttachShader(shaderProgramID, vertexShaderID);
        gl.glDeleteShader(vertexShaderID);

        int fragmentShaderID = gl.glCreateShader(GL_FRAGMENT_SHADER);
        gl.glShaderSource(fragmentShaderID, 1, 
            new String[]{fragmentShaderSource}, null);
        gl.glCompileShader(fragmentShaderID);
        gl.glAttachShader(shaderProgramID, fragmentShaderID);
        gl.glDeleteShader(fragmentShaderID);
        
        gl.glLinkProgram(shaderProgramID);
        gl.glValidateProgram(shaderProgramID);
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
     * 
     * @param gl The GL context
     */
    private void initVBO(GL3 gl)
    {
        int buffer[] = new int[1];

        // Create the vertex array object
        gl.glGenVertexArrays(1, IntBuffer.wrap(buffer));
        vertexArrayObject = buffer[0];

        gl.glBindVertexArray(vertexArrayObject);
        
        // Create the vertex buffer object
        gl.glGenBuffers(1, IntBuffer.wrap(buffer));
        vertexBufferObject = buffer[0];

        // Initialize the vertex buffer object
        gl.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
        int size = meshWidth * meshHeight * 4 * Sizeof.FLOAT;
        gl.glBufferData(GL_ARRAY_BUFFER, size, (Buffer) null,
            GL_DYNAMIC_DRAW);

        // Initialize the attribute location of the input
        // vertices for the shader program
        int location = gl.glGetAttribLocation(shaderProgramID, "inVertex");
        gl.glVertexAttribPointer(location, 4, GL_FLOAT, false, 0, 0);
        gl.glEnableVertexAttribArray(location);

        // Register the vertexBufferObject for use with CUDA
        vboGraphicsResource = new CUgraphicsResource();
        cuGraphicsGLRegisterBuffer(
            vboGraphicsResource, vertexBufferObject, 
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
    }
    


    @Override
    public void display(GLAutoDrawable drawable)
    {
        GL3 gl = drawable.getGL().getGL3();

        if (useCUDA)
        {
            // Run the CUDA kernel to generate new vertex positions.
            runCuda(gl);
        }
        else
        {
            // Run the Java method to generate new vertex positions.
            runJava(gl);
        }

        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate the shader program
        gl.glUseProgram(shaderProgramID);
        
        // Set the current projection matrix
        int projectionMatrixLocation = 
            gl.glGetUniformLocation(shaderProgramID, "projectionMatrix");
        gl.glUniformMatrix4fv(
            projectionMatrixLocation, 1, false, 
            simpleInteraction.getProjectionMatrix(), 0);

        // Set the current modelview matrix
        int modelviewMatrixLocation = 
            gl.glGetUniformLocation(shaderProgramID, "modelviewMatrix");
        gl.glUniformMatrix4fv(
            modelviewMatrixLocation, 1, false, 
            simpleInteraction.getModelviewMatrix(), 0);
        
        // Render the VBO
        gl.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
        gl.glDrawArrays(GL_POINTS, 0, meshWidth * meshHeight);
        
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
            String t = "JCuda / JOGL interaction sample - ";
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
     * 
     * @param gl The current GL.
     */
    private void runCuda(GL gl)
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
     * 
     * @param gl The current GL.
     */
    private void runJava(GL gl)
    {
        gl.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
        ByteBuffer byteBuffer = 
            gl.glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
        if (byteBuffer == null)
        {
            throw new RuntimeException("Unable to map buffer");
        }
        FloatBuffer vertices = 
            byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer();
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
        gl.glUnmapBuffer(GL_ARRAY_BUFFER);
        gl.glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    @Override
    public void reshape(
        GLAutoDrawable drawable, int x, int y, int width, int height)
    {
        setupView(drawable);
    }

    @Override
    public void dispose(GLAutoDrawable drawable)
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
