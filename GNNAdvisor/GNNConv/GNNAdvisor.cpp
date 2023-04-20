#include <torch/extension.h>
#include <vector>

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
    long long pctx_ptr);

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
    long long pctx_ptr);

// #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// torch::Tensor SAG(
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
//   CHECK_INPUT(input);
//   CHECK_INPUT(row_pointers);
//   CHECK_INPUT(column_index);
//   CHECK_INPUT(degrees);
//   CHECK_INPUT(part_pointers);
//   CHECK_INPUT(part2Node);

//   return SAG_cuda(input, row_pointers, column_index,
//                   degrees, part_pointers, part2Node,
//                   partSize, dimWorker, warpPerBlock,
//                   sprt_cu_function, A_blocks, C_blocks, Block_size, pctx_ptr);
// }

std::vector<torch::Tensor> spmm_forward(
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
    long long pctx_ptr)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_forward_cuda(input, weight, row_pointers, column_index,
                           degrees, part_pointers, part2Node,
                           partSize, dimWorker, warpPerBlock,
                           sprt_cu_function, A_blocks, C_blocks, Block_size,
                           pctx_ptr);
}

std::vector<torch::Tensor> spmm_backward(
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
    long long pctx_ptr)
{

  CHECK_INPUT(d_output);
  CHECK_INPUT(X);
  CHECK_INPUT(W);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_backward_cuda(d_output, X, W, row_pointers, column_index,
                            degrees, part_pointers, part2Node,
                            partSize, dimWorker, warpPerBlock,
                            sprt_cu_function, A_blocks, C_blocks, Block_size,
                            pctx_ptr);
}

std::vector<torch::Tensor> build_part(
    int partSize,
    torch::Tensor indptr)
{
  auto indptr_acc = indptr.accessor<int, 1>();
  int num_nodes = indptr.size(0) - 1;
  int degree, thisNumParts, numParts = 0;

  for (int i = 0; i < num_nodes; i++)
  {
    int degree = indptr_acc[i + 1] - indptr_acc[i];
    if (degree % partSize == 0)
      thisNumParts = degree / partSize;
    else
      thisNumParts = degree / partSize + 1;
    numParts += thisNumParts;
  }

  auto partPtr = torch::zeros(numParts + 1);
  auto part2Node = torch::zeros(numParts);

  int part_counter = 0;
  for (int i = 0; i < num_nodes; i++)
  {
    int degree = indptr_acc[i + 1] - indptr_acc[i];
    if (degree % partSize == 0)
      thisNumParts = degree / partSize;
    else
      thisNumParts = degree / partSize + 1;

    for (int pid = 0; pid < thisNumParts; pid++)
    {
      int partBeg = indptr_acc[i] + pid * partSize;
      int partEnd = partBeg + partSize < indptr_acc[i + 1] ? partBeg + partSize : indptr_acc[i + 1];
      partPtr[part_counter] = partBeg;
      part2Node[part_counter++] = i;
      if (i == num_nodes - 1 && partEnd == indptr_acc[i + 1])
        partPtr[part_counter] = partEnd;
    }
  }
  return {partPtr, part2Node};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // m.def("SAG", &SAG, "GNNAdvisor base Scatter-and-Gather Kernel (CUDA)");

  m.def("forward", &spmm_forward, "GNNAdvisor forward (CUDA)");
  m.def("backward", &spmm_backward, "GNNAdvisor backward (CUDA)");

  m.def("build_part", &build_part, "GNNAdvisor backward (CPU)");
}