/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.driver.gl.samples;

import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.util.Arrays;

/**
 * A class encapsulating a VERY simple mouse interaction for the GL samples:
 * It offers a {@link #getMouseControl()} that may be attached as a 
 * MouseMotionListener and MouseWheelListener to an arbitrary component,
 * and methods to obtain a {@link #getModelviewMatrix() model-view} and
 * {@link #getProjectionMatrix() projection} matrix.   
 */
class SimpleInteraction
{
    /**
     * The translation in X-direction
     */
    private float translationX = 0;

    /**
     * The translation in Y-direction
     */
    private float translationY = 0;

    /**
     * The translation in Z-direction
     */
    private float translationZ = -4;

    /**
     * The rotation about the X-axis, in degrees
     */
    private float rotationX = 40;

    /**
     * The rotation about the Y-axis, in degrees
     */
    private float rotationY = 30;

    /**
     * The current projection matrix
     */
    float projectionMatrix[] = new float[16];

    /**
     * The current projection matrix
     */
    float modelviewMatrix[] = new float[16];
    
    /**
     * Inner class encapsulating the MouseMotionListener and
     * MouseWheelListener for the interaction
     */
    class MouseControl implements MouseMotionListener, MouseWheelListener
    {
        private Point previousMousePosition = new Point();

        @Override
        public void mouseDragged(MouseEvent e)
        {
            int dx = e.getX() - previousMousePosition.x;
            int dy = e.getY() - previousMousePosition.y;

            // If the left button is held down, move the object
            if ((e.getModifiersEx() & MouseEvent.BUTTON1_DOWN_MASK) == 
                MouseEvent.BUTTON1_DOWN_MASK)
            {
                translationX += dx / 100.0f;
                translationY -= dy / 100.0f;
            }

            // If the right button is held down, rotate the object
            else if ((e.getModifiersEx() & MouseEvent.BUTTON3_DOWN_MASK) == 
                MouseEvent.BUTTON3_DOWN_MASK)
            {
                rotationX += dy;
                rotationY += dx;
            }
            previousMousePosition = e.getPoint();
            updateModelviewMatrix();
        }

        @Override
        public void mouseMoved(MouseEvent e)
        {
            previousMousePosition = e.getPoint();
        }

        @Override
        public void mouseWheelMoved(MouseWheelEvent e)
        {
            // Translate along the Z-axis
            translationZ += e.getWheelRotation() * 0.25f;
            previousMousePosition = e.getPoint();
            updateModelviewMatrix();
        }
    }
    
    /**
     * The mouse control
     */
    private final MouseControl mouseControl;
    
    /**
     * Default constructor
     */
    SimpleInteraction()
    {
        this.mouseControl = new MouseControl();
        updateModelviewMatrix();
    }
    
    /**
     * Returns the mouse control that may be attached to a component
     * as a MouseMotionListener and MouseWheelListener
     * 
     * @return The mouse control
     */
    MouseControl getMouseControl()
    {
        return mouseControl;
    }
    
    /**
     * Update the modelview matrix depending on the
     * current translation and rotation
     */
    private void updateModelviewMatrix()
    {
        float m0[] = translation(translationX, translationY, translationZ);
        float m1[] = rotationX(rotationX);
        float m2[] = rotationY(rotationY);
        modelviewMatrix = multiply(multiply(m1,m2), m0);
    }

    /**
     * Update the projection matrix for the given screen width and height
     * 
     * @param w The width
     * @param h The height
     */
    void updateProjectionMatrix(int w, int h)
    {
        float aspect = (float) w / h;
        projectionMatrix = perspective(50, aspect, 0.1f, 100.0f);
    }
    
    /**
     * Returns a <b>reference</b> to the modelview matrix
     * 
     * @return The matrix
     */
    float[] getModelviewMatrix()
    {
        return modelviewMatrix;
    }
    
    /**
     * Returns a <b>reference</b> to the projection matrix
     * 
     * @return The matrix
     */
    float[] getProjectionMatrix()
    {
        return projectionMatrix;
    }
    
    /**
     * Helper method that creates a perspective matrix
     * @param fovy The fov in y-direction, in degrees
     * 
     * @param aspect The aspect ratio
     * @param zNear The near clipping plane
     * @param zFar The far clipping plane
     * @return A perspective matrix
     */
    private static float[] perspective(
        float fovy, float aspect, float zNear, float zFar)
    {
        float radians = (float)Math.toRadians(fovy / 2);
        float deltaZ = zFar - zNear;
        float sine = (float)Math.sin(radians);
        if ((deltaZ == 0) || (sine == 0) || (aspect == 0)) 
        {
            return identity();
        }
        float cotangent = (float)Math.cos(radians) / sine;
        float m[] = identity();
        m[0*4+0] = cotangent / aspect;
        m[1*4+1] = cotangent;
        m[2*4+2] = -(zFar + zNear) / deltaZ;
        m[2*4+3] = -1;
        m[3*4+2] = -2 * zNear * zFar / deltaZ;
        m[3*4+3] = 0;
        return m;
    }
    
    /**
     * Creates an identity matrix
     * 
     * @return An identity matrix 
     */
    private static float[] identity()
    {
        float m[] = new float[16];
        Arrays.fill(m, 0);
        m[0] = m[5] = m[10] = m[15] = 1.0f;
        return m;
    }
    
    /**
     * Multiplies the given matrices and returns the result
     * 
     * @param m0 The first matrix
     * @param m1 The second matrix
     * @return The product m0*m1
     */
    private static float[] multiply(float m0[], float m1[])
    {
        float m[] = new float[16];
        for (int x=0; x < 4; x++)
        {
            for(int y=0; y < 4; y++)
            {
                m[x*4 + y] = 
                    m0[x*4+0] * m1[y+ 0] +
                    m0[x*4+1] * m1[y+ 4] +
                    m0[x*4+2] * m1[y+ 8] +
                    m0[x*4+3] * m1[y+12];
            }
        }
        return m;
    }
    
    /**
     * Creates a translation matrix
     * 
     * @param x The x translation
     * @param y The y translation
     * @param z The z translation
     * @return A translation matrix
     */
    private static float[] translation(float x, float y, float z)
    {
        float m[] = identity();
        m[12] = x;
        m[13] = y;
        m[14] = z;
        return m;
    }

    /**
     * Creates a matrix describing a rotation around the x-axis
     * 
     * @param angleDeg The rotation angle, in degrees
     * @return The rotation matrix
     */
    private static float[] rotationX(float angleDeg)
    {
        float m[] = identity();
        float angleRad = (float)Math.toRadians(angleDeg);
        float ca = (float)Math.cos(angleRad);
        float sa = (float)Math.sin(angleRad);
        m[ 5] =  ca;
        m[ 6] =  sa;
        m[ 9] = -sa;
        m[10] =  ca;
        return m;
    }

    /**
     * Creates a matrix describing a rotation around the y-axis
     * 
     * @param angleDeg The rotation angle, in degrees
     * @return The rotation matrix
     */
    private static float[] rotationY(float angleDeg)
    {
        float m[] = identity();
        float angleRad = (float)Math.toRadians(angleDeg);
        float ca = (float)Math.cos(angleRad);
        float sa = (float)Math.sin(angleRad);
        m[ 0] =  ca;
        m[ 2] = -sa;
        m[ 8] =  sa;
        m[10] =  ca;
        return m;
    }

    

}
