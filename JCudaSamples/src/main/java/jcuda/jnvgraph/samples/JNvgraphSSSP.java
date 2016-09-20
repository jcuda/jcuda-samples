/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jnvgraph.samples;

import static jcuda.jnvgraph.JNvgraph.nvgraphAllocateEdgeData;
import static jcuda.jnvgraph.JNvgraph.nvgraphAllocateVertexData;
import static jcuda.jnvgraph.JNvgraph.nvgraphCreate;
import static jcuda.jnvgraph.JNvgraph.nvgraphCreateGraphDescr;
import static jcuda.jnvgraph.JNvgraph.nvgraphDestroy;
import static jcuda.jnvgraph.JNvgraph.nvgraphDestroyGraphDescr;
import static jcuda.jnvgraph.JNvgraph.nvgraphGetVertexData;
import static jcuda.jnvgraph.JNvgraph.nvgraphSetEdgeData;
import static jcuda.jnvgraph.JNvgraph.nvgraphSetGraphStructure;
import static jcuda.jnvgraph.JNvgraph.nvgraphSssp;
import static jcuda.jnvgraph.nvgraphTopologyType.NVGRAPH_CSC_32;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.cudaDataType;
import jcuda.jnvgraph.JNvgraph;
import jcuda.jnvgraph.nvgraphCSCTopology32I;
import jcuda.jnvgraph.nvgraphGraphDescr;
import jcuda.jnvgraph.nvgraphHandle;

/**
 * A direct port of the NVIDIA nvGRAPH "Single Source Shortest Path" sample
 */
public class JNvgraphSSSP
{
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        JNvgraph.setExceptionsEnabled(true);
        int n = 6, nnz = 10, vertex_numsets = 2, edge_numsets = 1;

        // nvgraph variables
        nvgraphHandle handle = new nvgraphHandle();
        nvgraphGraphDescr graph = new nvgraphGraphDescr();
        nvgraphCSCTopology32I CSC_input = new nvgraphCSCTopology32I();
        int edge_dimT = cudaDataType.CUDA_R_32F;
        int vertex_dimT[];

        // Init host data
        int[] destination_offsets_h = new int[n+1];
        int[]source_indices_h = new int[nnz];
        float[] weights_h = new float[nnz];
        float[] sssp_1_h = new float[n];
        float[]sssp_2_h = new float[n];
        vertex_dimT = new int[vertex_numsets];
        vertex_dimT[0] = cudaDataType.CUDA_R_32F; 
        vertex_dimT[1] = cudaDataType.CUDA_R_32F;

        weights_h [0] = 0.333333f;
        weights_h [1] = 0.500000f;
        weights_h [2] = 0.333333f;
        weights_h [3] = 0.500000f;
        weights_h [4] = 0.500000f;
        weights_h [5] = 1.000000f;
        weights_h [6] = 0.333333f;
        weights_h [7] = 0.500000f;
        weights_h [8] = 0.500000f;
        weights_h [9] = 0.500000f;

        destination_offsets_h [0] = 0;
        destination_offsets_h [1] = 1;
        destination_offsets_h [2] = 3;
        destination_offsets_h [3] = 4;
        destination_offsets_h [4] = 6;
        destination_offsets_h [5] = 8;
        destination_offsets_h [6] = 10;

        source_indices_h [0] = 2;
        source_indices_h [1] = 0;
        source_indices_h [2] = 2;
        source_indices_h [3] = 0;
        source_indices_h [4] = 4;
        source_indices_h [5] = 5;
        source_indices_h [6] = 2;
        source_indices_h [7] = 3;
        source_indices_h [8] = 3;
        source_indices_h [9] = 4;

        nvgraphCreate(handle);
        nvgraphCreateGraphDescr (handle, graph);

        CSC_input.nvertices = n;
        CSC_input.nedges = nnz;
        CSC_input.destination_offsets = Pointer.to(destination_offsets_h);
        CSC_input.source_indices = Pointer.to(source_indices_h);

        // Set graph connectivity and properties (transfers)
        nvgraphSetGraphStructure(
            handle, graph, CSC_input, NVGRAPH_CSC_32);
        nvgraphAllocateVertexData(
            handle, graph, vertex_numsets, Pointer.to(vertex_dimT));
        nvgraphAllocateEdgeData(
            handle, graph, edge_numsets, Pointer.to(new int[] { edge_dimT }));
        nvgraphSetEdgeData(
            handle, graph, Pointer.to(weights_h), 0, NVGRAPH_CSC_32);

        // Solve
        int source_vert = 0;
        nvgraphSssp(handle, graph, 0,  Pointer.to(new int[]{source_vert}), 0);

        // Solve with another source
        source_vert = 5;
        nvgraphSssp(handle, graph, 0, Pointer.to(new int[]{source_vert}), 1);
        
        // Get and print result

        nvgraphGetVertexData(
            handle, graph, Pointer.to(sssp_1_h), 0, NVGRAPH_CSC_32);
        // expect sssp_1_h = 
        //     (0.000000 0.500000 0.500000 1.333333 0.833333 1.333333)^T
        System.out.printf("sssp_1_h: "+Arrays.toString(sssp_1_h)+"\n");


        nvgraphGetVertexData(
            handle, graph, Pointer.to(sssp_2_h), 1, NVGRAPH_CSC_32);
        // expect sssp_2_h = 
        //     (FLT_MAX FLT_MAX FLT_MAX 1.000000 1.500000 0.000000 )^T
        System.out.printf("sssp_2_h: "+Arrays.toString(sssp_2_h)+"\n");

        System.out.printf("\nDone!\n");

        nvgraphDestroyGraphDescr (handle, graph);
        nvgraphDestroy (handle);
    }
}
