#ifndef MINIGUN_CPU_ADVANCE_H_
#define MINIGUN_CPU_ADVANCE_H_

#include "../advance.h"
#include "./advance_all.h"

namespace minigun {
namespace advance {

template <typename Idx,
          typename DType,
          typename Config,
          typename GData,
          typename Functor,
          typename Alloc>
struct DispatchXPU<kDLCPU, Idx, DType, Config, GData, Functor, Alloc> {
  static void Advance(
      const RuntimeConfig& rtcfg,
      const SpMat<Idx>& spmat,
      GData* gdata,
      Alloc* alloc) {
    CPUAdvanceAll<Idx, DType, Config, GData, Functor, Alloc>(
        csr, gdata, alloc);
  }
};

}  // namespace advance
}  // namespace minigun

#endif  // MINIGUN_CPU_ADVANCE_H_
