/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2017 Marco Hutter - http://www.jcuda.org
 */
package jcuda.jnvgraph.samples;

import jcuda.jnvgraph.nvgraphCSRTopology32I;
import static jcuda.jnvgraph.JNvgraph.*;

import jcuda.Pointer;
import jcuda.jnvgraph.nvgraphGraphDescr;
import jcuda.jnvgraph.nvgraphHandle;
import static jcuda.jnvgraph.nvgraphTopologyType.*;

/**
 * A direct port of the "nvGRAPH Triangles Counting" sample
 */
public class JNvgraphTrianglesCounting
{
    public static void main(String[] args)
    {
        // nvgraph variables
        nvgraphHandle handle = new nvgraphHandle();
        nvgraphGraphDescr graph = new nvgraphGraphDescr(); 
        nvgraphCSRTopology32I CSR_input = new nvgraphCSRTopology32I();  

        // Undirected graph:
        // 0       2-------4
        //  \     / \     / \
        //   \   /   \   /   \
        //    \ /     \ /     \
        //     1-------3-------5
        // 3 triangles
        // CSR of lower triangular of adjacency matrix:
        int n = 6, nnz = 8;
        int source_offsets[] = {0, 0, 1, 2, 4, 6, 8};
        int destination_indices[] = {0, 1, 1, 2, 2, 3, 3, 4};
        
        nvgraphCreate(handle);
        nvgraphCreateGraphDescr(handle, graph);
        CSR_input.nvertices = n;
        CSR_input.nedges = nnz;
        CSR_input.source_offsets = Pointer.to(source_offsets);
        CSR_input.destination_indices = Pointer.to(destination_indices);

        // Set graph connectivity
        nvgraphSetGraphStructure(handle, graph, CSR_input, NVGRAPH_CSR_32);
        long trcount[] = { 0 };
        nvgraphTriangleCount(handle, graph, trcount);
        System.out.printf("Triangles count: %d\n", trcount[0]);

        nvgraphDestroyGraphDescr(handle, graph);
        nvgraphDestroy(handle);
    }
}
