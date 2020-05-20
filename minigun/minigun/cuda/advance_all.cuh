#ifndef MINIGUN_CUDA_ADVANCE_ALL_CUH_
#define MINIGUN_CUDA_ADVANCE_ALL_CUH_

#include "./cuda_common.cuh"
#include <algorithm>
#include <cstdio>

namespace minigun {
namespace advance {

#define MAX_NTHREADS 1024
#define PER_THREAD_WORKLOAD 1
#define MAX_NBLOCKS 65535

// Binary search the row_offsets to find the source node of the edge id.
template <typename Idx>
__device__ __forceinline__ Idx BinarySearchSrc(const IntArray1D<Idx>& array, Idx eid) {
  Idx lo = 0, hi = array.length - 1;
  while (lo < hi) {
    Idx mid = (lo + hi) >> 1;
    if (_ldg(array.data + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (_ldg(array.data + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CudaAdvanceAllEdgeParallelCSRKernel(
    Csr<Idx> csr,
    GData gdata) {
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx stride_y = blockDim.y * gridDim.y;
  Idx eid = ty;
  while (eid < csr.column_indices.length) {
    // TODO(minjie): this is pretty inefficient; binary search is needed only
    //   when the thread is processing the neighbor list of a new node.
    Idx src = BinarySearchSrc(csr.row_offsets, eid);
    Idx dst = _ldg(csr.column_indices.data + eid);
    Functor::ApplyEdge(src, dst, eid, &gdata);
    eid += stride_y;
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllEdgeParallelCSR(
    const RuntimeConfig& rtcfg,
    const Csr<Idx>& csr,
    GData* gdata,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx M = coo.column.length;
  const int ty = MAX_NTHREADS / rtcfg.data_num_threads;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((M + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  /*
  LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
    << nthrs.x << "," << nthrs.y << ")";
  */
  CudaAdvanceAllEdgeParallelCSRKernel<Idx, DType, Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(csr, *gdata);
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CudaAdvanceAllEdgeParallelKernel(
    Coo<Idx> coo,
    GData gdata) {
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx stride_y = blockDim.y * gridDim.y;
  Idx eid = ty;
  while (eid < coo.column.length) {
    Idx src = _ldg(coo.row.data + eid);
    Idx dst = _ldg(coo.column.data + eid);
    Functor::ApplyEdge(src, dst, eid, &gdata);
    eid += stride_y;
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllEdgeParallel(
    const RuntimeConfig& rtcfg,
    const Coo<Idx>& coo,
    GData* gdata,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx M = coo.column.length;
  const int ty = MAX_NTHREADS / rtcfg.data_num_threads;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((M + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  /*
  LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
    << nthrs.x << "," << nthrs.y << ")";
  */
  CudaAdvanceAllEdgeParallelKernel<Idx, DType, Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(coo, *gdata);
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor>
__global__ void CudaAdvanceAllNodeParallelKernel(
    Csr<Idx> csr,
    GData gdata) {
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx stride_y = blockDim.y * gridDim.y;
  Idx stride_x = blockDim.x * gridDim.x;
  Idx feat_size = Functor::GetFeatSize(&gdata);
  DType *outbuf = Functor::GetOutBuf(&gdata);
  DType val;
  if (Config::kParallel == kDst) {
    Idx dst = ty;
    while (dst < csr.row_offsets.length - 1) {
      Idx start = _ldg(csr.row_offsets.data + dst);
      Idx end = _ldg(csr.row_offsets.data + dst + 1);
      Idx feat_idx = tx;
      while (feat_idx < feat_size) {
        Idx outoff = dst * feat_size + feat_idx;
        if (outbuf != nullptr)
          val = _ldg(outbuf + outoff);
        for (Idx eid = start; eid < end; ++eid) {
          Idx src = _ldg(csr.column_indices.data + eid);
          Functor::ApplyEdgeReduce(src, dst, eid, feat_idx, val, &gdata);
        }
        if (outbuf != nullptr)
          outbuf[outoff] = val;
        feat_idx += stride_x;
      }
      dst += stride_y;
    }
  } else {
    Idx src = ty;
    while (src < csr.row_offsets.length - 1) {
      Idx start = _ldg(csr.row_offsets.data + src);
      Idx end = _ldg(csr.row_offsets.data + src + 1);
      Idx feat_idx = tx;
      while (feat_idx < feat_size) {
        Idx outoff = src * feat_size + feat_idx;
        if (outbuf != nullptr)
          val = _ldg(outbuf + outoff);
        for (Idx eid = start; eid < end; ++eid) {
          Idx dst = _ldg(csr.column_indices.data + eid);
          Functor::ApplyEdgeReduce(src, dst, eid, feat_idx, val, &gdata);
        }
        if (outbuf != nullptr)
          outbuf[outoff] = val;
        feat_idx += stride_x;
      }
      src += stride_y;
    }
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAllNodeParallel(
    const RuntimeConfig& rtcfg,
    const Csr<Idx>& csr,
    GData* gdata,
    Alloc* alloc) {
  CHECK_GT(rtcfg.data_num_blocks, 0);
  CHECK_GT(rtcfg.data_num_threads, 0);
  const Idx N = csr.row_offsets.length - 1;
  const int ty = 1;
  const int ny = ty * PER_THREAD_WORKLOAD;
  const int by = std::min((N + ny - 1) / ny, static_cast<Idx>(MAX_NBLOCKS));
  const dim3 nblks(rtcfg.data_num_blocks, by);
  const dim3 nthrs(rtcfg.data_num_threads, ty);
  /*
  LOG(INFO) << "Blocks: (" << nblks.x << "," << nblks.y << ") Threads: ("
    << nthrs.x << "," << nthrs.y << ")";
  */
  CudaAdvanceAllNodeParallelKernel<Idx, DType, Config, GData, Functor>
    <<<nblks, nthrs, 0, rtcfg.stream>>>(csr, *gdata);
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CudaAdvanceAll(
    const RuntimeConfig& rtcfg,
    const SpMat<Idx> &spmat,
    GData* gdata,
    Alloc* alloc) {
  switch (Config::kParallel) {
    case kSrc:
      if (spmat.out_csr != nullptr)
        CudaAdvanceAllNodeParallel<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.out_csr, gdata, alloc);
      else
        LOG(FATAL) << "out_csr must be created in source parallel mode";
      break;
    case kEdge:
      if (spmat.coo != nullptr)
        CudaAdvanceAllEdgeParallel<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.coo, gdata, alloc);
      else if (spmat.out_csr != nullptr)
        CudaAdvanceAllEdgeParallelCSR<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.out_csr, gdata, alloc);
      else if (spmat.in_csr != nullptr)
        CudaAdvanceAllEdgeParallelCSR<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.in_csr, gdata, alloc);
      else
        LOG(FATAL) << "At least one sparse format should be created.";
      break;
    case kDst:
      if (spmat.in_csr != nullptr)
        CudaAdvanceAllNodeParallel<Idx, DType, Config, GData, Functor, Alloc>(
            rtcfg, *spmat.in_csr, gdata, alloc);
      else
        LOG(FATAL) << "in_csr must be created in destination parallel mode.";
      break;
  }
}

#undef MAX_NTHREADS
#undef PER_THREAD_WORKLOAD
#undef MAX_NBLOCKS

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CUDA_ADVANCE_ALL_CUH_
