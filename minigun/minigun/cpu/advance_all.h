//
// Created by Ye, Zihao on 2019-11-23.
//

#ifndef MINIGUN_ADVANCE_ALL_H
#define MINIGUN_ADVANCE_ALL_H

#include "../advance.h"
#include <algorithm>
#include <dmlc/omp.h>

namespace minigun {
namespace advance {

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvanceAllEdgeParallelCSR(
    const Csr<Idx>& csr,
    GData *gdata) {
  Idx E = csr.column_indices.length;
#pragma omp parallel for
  for (Idx eid = 0; eid < E; ++eid) {
    const Idx src = std::lower_bound(csr.row_offsets, csr.row_offsets + csr.row_offsets.length, eid);
    const Idx dst = csr.column_indices.data[eid];
    Functor::ApplyEdge(src, dst, eid, gdata);
  }  
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvanceAllEdgeParallel(
    const Coo<Idx>& coo,
    GData *gdata) {
  Idx E = coo.column.length;
#pragma omp parallel for
  for (Idx eid = 0; eid < E; ++eid) {
    const Idx src = coo.row.data[eid];
    const Idx dst = coo.col.data[eid];
    Functor::ApplyEdge(src, dst, eid, gdata);
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvanceAllNodeParallel(
    const Csr<Idx>& csr,
    GData *gdata) {
  Idx N = csr.row_offsets.length - 1;
  if (Config::kParallel == kDst) {
#pragma omp parallel for
    for (Idx vid = 0; vid < N; ++vid) {
      const Idx dst = vid;
      const Idx start = csr.row_offsets.data[dst];
      const Idx end = csr.row_offsets.data[dst + 1];
      for (Idx eid = start; eid < end; ++eid) {
        const Idx src = csr.column_indices.data[eid];
        Functor::ApplyEdge(src, dst, eid, gdata);
      }
    }
  } else {
#pragma omp parallel for
    for (Idx vid = 0; vid < N; ++vid) {
      const Idx src = vid;
      const Idx start = csr.row_offsets.data[src];
      const Idx end = csr.row_offsets.data[src + 1];
      for (Idx eid = start; eid < end; ++eid) {
        const Idx dst = csr.column_indices.data[eid];
        Functor::ApplyEdge(src, dst, eid, gdata);
      }
    }
  }
}

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
void CPUAdvanceAll(
      const SpMat<Idx>& spmat,
      GData* gdata) {
  switch (Config::kParallel) {
    case kSrc:
      if (spmat.out_csr != nullptr)
        CPUAdvanceAllNodeParallel(*spmat.out_csr, gdata);
      else
        LOG(FATAL) << "out_csr need to be created in source parallel mode.";
      break;
    case kEdge:
      if (spmat.coo != nullptr)
        CPUAdvanceAllEdgeParallel(*spmat.coo, gdata);
      else if (spmat.out_csr != nullptr)
        CPUAdvanceAllEdgeParallelCSR(*spmat.out_csr, gdata);
      else if (spmat.in_csr != nullptr)
        CPUAdvanceAllEdgeParallelCRS(*spmat.in_csr, gdata);
      else
        LOG(FATAL) << "At least one sparse format should be created.";
      break;
    case kDst:
      if (spmat.in_csr != nullptr)
        CPUAdvanceAllNodeParallel(*spmat.in_csr, gdata);
      else
        LOG(FATAL) << "in_csr need to be created in destination parallel mode."; 
      break;
  }
}


} //namespace advance
} //namespace minigun

#endif //MINIGUN_ADVANCE_ALL_H
