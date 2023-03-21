#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define WARP_SIZE 32
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line);
        exit(-1);
    }
}

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void warmup() {}

__device__ inline void atomicAdd_F(float *address, float value)
{
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f)
        ;
}

// template <typename scalar_t>
// __global__ void SAG_cuda_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> row_pointers,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
//     torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
//     const int num_nodes,
//     const int dim,
//     const int num_parts,
//     const int partSize,
//     const int dimWorker,
//     const int warpPerBlock);

template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
    const int num_nodes,
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock);

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
    const int num_nodes,
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock);

// ////////////////////////////////////////////
// //
// // Basic Scatter-And-Gather kernel.
// //
// ////////////////////////////////////////////
// torch::Tensor SAG_cuda(
//     torch::Tensor input,
//     torch::Tensor row_pointers,
//     torch::Tensor column_index,
//     torch::Tensor degrees,
//     torch::Tensor part_pointers,
//     torch::Tensor part2Node,
//     int partSize,
//     int dimWorker,
//     int warpPerBlock,
//     long long sprt_cu_function,
//     int A_blocks,
//     int C_blocks,
//     int Block_size,
//     long long pctx_ptr)
// {
//     auto output = torch::zeros_like(input);

//     const int num_nodes = output.size(0);
//     const int dim = output.size(1);
//     const int num_parts = part2Node.size(0);

//     const int block = warpPerBlock * WARP_SIZE;
//     const int grid = (num_parts * WARP_SIZE + block - 1) / block;
//     int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);

//     // printf("grid: %d, block: %d, shared_memory: %d\n", grid, block, shared_memory);
//     // printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
//     // printf("dimWorker: %d\n", dimWorker);
//     // #define PROFILE 200

// #ifdef PROFILE
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     for (int i = 0; i < PROFILE; i++)
//     {
//         warmup<<<1, 1>>>();
//     }
//     cudaEventRecord(start, 0);

//     for (int i = 0; i < PROFILE; i++)
// #endif

//         CUcontext *ctx_ptr = (CUcontext *)pctx_ptr;
//     // std::cout << "*ctx_ptr:" << *ctx_ptr << std::endl;
//     checkCudaErrors(cuCtxSetCurrent(*ctx_ptr));
//     CUfunction *func_ptr = (CUfunction *)sprt_cu_function;

//     float **d_BC, **d_AC;
//     float *tmp;
//     gpuErrchk(cudaMalloc((void **)&d_BC, sizeof(float *)));
//     gpuErrchk(cudaMalloc((void **)&d_AC, sizeof(float *)));
//     tmp = (float *)(input.data_ptr());
//     gpuErrchk(cudaMemcpy(d_BC, &tmp, sizeof(float *), cudaMemcpyHostToDevice));
//     tmp = (float *)(output.data_ptr());
//     gpuErrchk(cudaMemcpy(d_AC, &tmp, sizeof(float *), cudaMemcpyHostToDevice));
//     void *args[2] = {&d_BC, &d_AC};

//     // printf("\nLaunching kernel with A_blocks:%d,C_blocks:%d,Block_size:%d\n", A_blocks, C_blocks, Block_size);

//     checkCudaErrors(cuLaunchKernel(*func_ptr, A_blocks, C_blocks, 1, // Nx1x1 blocks
//                                    Block_size, 1, 1,                 // 1x1x1 threads
//                                    0, 0, args, 0));
//     gpuErrchk(cudaFree((void *)d_BC));
//     gpuErrchk(cudaFree((void *)d_AC));
//     // AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "Scatter_and_Gather", ([&] {
//     //                             SAG_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
//     //                                 output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//     //                                 input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//     //                                 row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//     //                                 column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//     //                                 degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//     //                                 part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//     //                                 part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//     //                                 num_nodes,
//     //                                 dim,
//     //                                 num_parts,
//     //                                 partSize,
//     //                                 dimWorker,
//     //                                 warpPerBlock
//     //                             );
//     //                         }));

// #ifdef PROFILE
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);
//     float gflop = 2 * column_index.size(0) / 1e6 * dim;
//     float milliseconds;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("TC-GNN -- Time (ms): %.3f, GFLOPs: %.3f\n", milliseconds / PROFILE, gflop / (milliseconds / PROFILE));
//     printf("\n================================\n");
// #endif

//     cudaDeviceSynchronize();

//     return output;
// }
// template <typename scalar_t>
// __global__ void SAG_cuda_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> row_pointers,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
//     torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
//     const int num_nodes,
//     const int dim,
//     const int num_parts,
//     const int partSize,
//     const int dimWorker,
//     const int warpPerBlock)
// {

//     int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
//     int warpId = tid / WARP_SIZE;                    // global warp-id
//     int block_warpId = threadIdx.x / WARP_SIZE;      // block warp-id
//     int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

//     extern __shared__ int part_meta[];                                     // part information.
//     int *partial_ids = part_meta;                                          // caching ids
//     float *partial_results = (float *)&part_meta[partSize * warpPerBlock]; // caching partial results.

//     if (warpId < num_parts)
//     {

//         int srcId = part2Node[warpId];           // aggregated source node
//         int partBeg = part_pointers[warpId];     // partitioning pointer start
//         int partEnd = part_pointers[warpId + 1]; // part pointer end

//         // Cache the part neighbors.
//         const int pindex_base = block_warpId * partSize;
//         // #pragma unroll
//         for (int nidx = partBeg + laneid; nidx < partEnd; nidx += dimWorker)
//         {
//             // printf("1--pindex_base: %d, laneid: %d\n", pindex_base, laneid);
//             partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
//             // if(partial_ids[pindex_base + laneid]  >= num_nodes || partial_ids[pindex_base + laneid]  < 0) printf("---- partial_ids: %d\n", partial_ids[pindex_base + laneid] );
//         }

//         __syncwarp();

//         // Neighbor aggregation within each part
//         const int presult_base = block_warpId * dim;
//         for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
//         {
//             // if (laneid == 0) printf("2--pindex_base: %d, nIdx: %d\n", pindex_base, nIdx);
//             int nid = partial_ids[pindex_base + nIdx];
//             // if(nid >= num_nodes || nid < 0) printf("Error nid: %d\n", nid);

//             // Initialize shared memory for partial results
//             if (nIdx == 0)
//                 if (laneid < dimWorker)
// #pragma unroll
//                     for (int d = laneid; d < dim; d += dimWorker)
//                     {
//                         partial_results[presult_base + d] = 0.0f;
//                     }

//             if (laneid < dimWorker)
// #pragma unroll
//                 for (int d = laneid; d < dim; d += dimWorker)
//                 {
//                     partial_results[presult_base + d] += input[nid][d];
//                 }
//         }

//         // output the result to global memory from the shared memory
//         if (laneid < dimWorker)
// #pragma unroll
//             for (int d = laneid; d < dim; d += dimWorker)
//             {
//                 atomicAdd_F((float *)&output[srcId][d], partial_results[presult_base + d]);
//             }
//     }
// }

////////////////////////////////////////////
//
// Foward Pass (GCN)  node update --> neighbor aggregation
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock,
    long long sprt_cu_function,
    int A_blocks,
    int C_blocks,
    int Block_size,
    long long pctx_ptr,
    int modeBarrier)
{
    auto input_weight = torch::mm(input, weight);
    auto output = torch::zeros({input.size(0), weight.size(1)}, torch::kCUDA);

    // Mode 1: SparseRT

    if (modeBarrier > 0) // exists element to deal with
    {
        CUcontext *ctx_ptr = (CUcontext *)pctx_ptr;
        // std::cout << "*ctx_ptr:" << *ctx_ptr << std::endl;
        checkCudaErrors(cuCtxSetCurrent(*ctx_ptr));
        CUfunction *func_ptr = (CUfunction *)sprt_cu_function;

        float **d_BC, **d_AC;
        float *tmp;
        gpuErrchk(cudaMalloc((void **)&d_BC, sizeof(float *)));
        gpuErrchk(cudaMalloc((void **)&d_AC, sizeof(float *)));
        tmp = (float *)(input_weight.data_ptr());
        gpuErrchk(cudaMemcpy(d_BC, &tmp, sizeof(float *), cudaMemcpyHostToDevice));
        tmp = (float *)(output.data_ptr());
        gpuErrchk(cudaMemcpy(d_AC, &tmp, sizeof(float *), cudaMemcpyHostToDevice));
        void *args[2] = {&d_BC, &d_AC};

        printf("\nLaunching SparseRT kernel with A_blocks:%d,C_blocks:%d,Block_size:%d\n", A_blocks, C_blocks, Block_size);

        checkCudaErrors(cuLaunchKernel(*func_ptr, A_blocks, C_blocks, 1, // A_blocks x C_blocks x 1 blocks
                                       Block_size, 1, 1,                 // Block_size x 1 x 1 threads
                                       0, 0, args, 0));
        gpuErrchk(cudaFree((void *)d_BC));
        gpuErrchk(cudaFree((void *)d_AC));
    }

    // Mode 2: GNNA Neighbour Groups
    if (modeBarrier < output.size(0))
    {
        const int dim = output.size(1);
        const int num_nodes = output.size(0);
        const int num_parts = part2Node.size(0);

        const int block = warpPerBlock * WARP_SIZE;
        const int grid = (num_parts * WARP_SIZE + block - 1) / block;
        const int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);

        // printf("grid: %d, block: %d\n", grid, block);
        // printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
        // printf("input: (%d, %d)\n", tmp.size(0), tmp.size(1));
        // printf("dimWorker: %d\n", dimWorker);
        // printf("shared_memory: %d\n", tmp.size(0), tmp.size(1));

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "spmm_cuda_forward", ([&]
                                                                              { spmm_forward_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
                                                                                    output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                    input_weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                    row_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                    column_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                    degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                                    part_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                    part2Node.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                    num_nodes,
                                                                                    dim,
                                                                                    num_parts,
                                                                                    partSize,
                                                                                    dimWorker,
                                                                                    warpPerBlock); }));

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
    }

    cudaDeviceSynchronize();

    return {output};
}

template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
    const int num_nodes,
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
    int warpId = tid / WARP_SIZE;                    // global warp-id
    int block_warpId = threadIdx.x / WARP_SIZE;      // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                           // part information.
    int *const partial_ids = part_meta;                                          // caching ids
    float *const partial_results = (float *)&part_meta[partSize * warpPerBlock]; // caching partial results.

    if (warpId < num_parts)
    {

        const int srcId = part2Node[warpId];           // aggregated source node
        const int partBeg = part_pointers[warpId];     // partitioning pointer start
        const int partEnd = part_pointers[warpId + 1]; // part pointer end
        const float src_norm = degrees[srcId];         // norm of the source node

        // Cache the part neighbors by all threads from a warp.
        const int pindex_base = block_warpId * partSize;
#pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE)
        {
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
        }

        __syncwarp();

        // if (laneid == 0)
        // for (int nIdx = laneid; nIdx < partEnd - partBeg; nIdx++){
        // int nid = partial_ids[pindex_base + nIdx];
        // int nid = partial_ids[nIdx];
        // printf("verify nid - 111111: %d\n", nid);
        // if(nid >= num_nodes || nid < 0) printf("verify nid: %d\n", nid);
        // }

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // int nid = partial_ids[nIdx];
            // if (laneid == 0)
            //     printf("verify nid - 222222: %d\n", nid);
            float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
#pragma unroll
                    for (int d = laneid; d < dim; d += dimWorker)
                    {
                        partial_results[presult_base + d] = 0.0f;
                    }

            if (laneid < dimWorker)
#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {
                    partial_results[presult_base + d] += __fmaf_rn(degree_norm_inv, input[nid][d], 0);
                    // partial_results[presult_base + d] += input[nid][d];
                }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
#pragma unroll
            for (int d = laneid; d < dim; d += dimWorker)
            {
                atomicAdd_F((float *)&output[srcId][d], partial_results[presult_base + d]);
            }
    }
}

////////////////////////////////////////////
//
// backward pass (GCN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_backward_cuda(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock,
    long long sprt_cu_function,
    int A_blocks,
    int C_blocks,
    int Block_size,
    long long pctx_ptr,
    int modeBarrier)
{

    auto d_input_prime = torch::zeros_like(d_output);

    // Mode 1: SparseRT

    if (modeBarrier > 0) // exists element to deal with
    {
        CUcontext *ctx_ptr = (CUcontext *)pctx_ptr;
        // std::cout << "*ctx_ptr:" << *ctx_ptr << std::endl;
        checkCudaErrors(cuCtxSetCurrent(*ctx_ptr));
        CUfunction *func_ptr = (CUfunction *)sprt_cu_function;

        float **d_BC, **d_AC;
        float *tmp;
        gpuErrchk(cudaMalloc((void **)&d_BC, sizeof(float *)));
        gpuErrchk(cudaMalloc((void **)&d_AC, sizeof(float *)));
        tmp = (float *)(d_output.data_ptr());
        gpuErrchk(cudaMemcpy(d_BC, &tmp, sizeof(float *), cudaMemcpyHostToDevice));
        tmp = (float *)(d_input_prime.data_ptr());
        gpuErrchk(cudaMemcpy(d_AC, &tmp, sizeof(float *), cudaMemcpyHostToDevice));
        void *args[2] = {&d_BC, &d_AC};

        // printf("\nLaunching SparseRT kernel with A_blocks:%d,C_blocks:%d,Block_size:%d\n", A_blocks, C_blocks, Block_size);

        checkCudaErrors(cuLaunchKernel(*func_ptr, A_blocks, C_blocks, 1, // A_blocks x C_blocks x 1 blocks
                                       Block_size, 1, 1,                 // Block_size x 1 x 1 threads
                                       0, 0, args, 0));
        gpuErrchk(cudaFree((void *)d_BC));
        gpuErrchk(cudaFree((void *)d_AC));
    }

    // Mode 2: GNNA Neighbour Groups
    if (modeBarrier < d_output.size(0))
    {
        const int dim = d_input_prime.size(1);
        // printf("GNNA backward dim:%d\n", dim);
        const int num_nodes = d_input_prime.size(0);
        const int num_parts = part2Node.size(0);

        const int block = warpPerBlock * WARP_SIZE;
        const int grid = (num_parts * WARP_SIZE + block - 1) / block;
        // const int shared_memory = warpPerBlock * partSize * sizeof(int) + warpPerBlock * dim * sizeof(float);
        const int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES(d_output.scalar_type(), "spmm_cuda_backward", ([&]
                                                                                  { spmm_backward_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
                                                                                        d_input_prime.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                        d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                        row_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        column_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                                        part_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        part2Node.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        num_nodes,
                                                                                        dim,
                                                                                        num_parts,
                                                                                        partSize,
                                                                                        dimWorker,
                                                                                        warpPerBlock); }));

        // check for error
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
    }
    cudaDeviceSynchronize();

    auto d_input = torch::mm(d_input_prime, W.transpose(0, 1));
    auto d_weight = torch::mm(X.transpose(0, 1), d_input_prime);

    return {d_input, d_weight};
}

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
    const int num_nodes,
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int block_warpId = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x % WARP_SIZE;

    extern __shared__ int part_meta[];                                     // part information.
    int *partial_ids = part_meta;                                          // caching ids
    float *partial_results = (float *)&part_meta[partSize * warpPerBlock]; // caching partial results.

    if (warpId < num_parts)
    {

        const int srcId = part2Node[warpId];
        const int partBeg = part_pointers[warpId];
        const int partEnd = part_pointers[warpId + 1];
        float src_norm = degrees[srcId];

        const int pindex_base = block_warpId * partSize;
#pragma unroll
        for (int nid = partBeg + laneid; nid < partEnd; nid += WARP_SIZE)
        {
            partial_ids[pindex_base + nid - partBeg] = column_index[nid];
        }

        // #pragma unroll
        // for (int nidx = partBeg; nidx < partEnd; nidx++){
        // //     if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
        //     partial_ids[nidx - partBeg] = column_index[nidx];
        // }

        __syncwarp();

        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // int nid = partial_ids[nIdx];
            float degree_norm = __fmaf_rn(src_norm, degrees[nid], 0);

            if (nIdx == 0)
                if (laneid < dimWorker)
#pragma unroll
                    for (int d = laneid; d < dim; d += dimWorker)
                    {
                        partial_results[presult_base + d] = 0;
                    }

            if (laneid < dimWorker)
#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {
                    partial_results[presult_base + d] += __fmaf_rn(degree_norm, d_output[nid][d], 0);
                }
        }

        if (laneid < dimWorker)
#pragma unroll
            for (int d = laneid; d < dim; d += dimWorker)
            {
                atomicAdd_F((float *)&d_input[srcId][d], partial_results[presult_base + d]);
            }
    }
}
