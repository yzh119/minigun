#ifndef SAMPLES_BENCHMARK_MINIGUN_ESOFTMAX_BACK_CUH_
#define SAMPLES_BENCHMARK_MINIGUN_ESOFTMAX_BACK_CUH_

#include <cuda_runtime.h>
#include "../../samples_io.h"
#include "../../samples_utils.h"

namespace esoftmax_back {

struct GData {
  int H = 0;  // num heads
  float* score{nullptr};
  float* grad_score{nullptr};
  float* accum{nullptr};
  float* out{nullptr};
  int* eid_mapping{nullptr};
};

// backward softmax phase 0
struct BackSoftmaxAccum {
  static __device__ __forceinline__ void ApplyEdge(
    int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  // accum: (N, H)
  static __device__ __forceinline__ void ApplyEdgeReduce(
    int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {
    const int H = gdata->H;
    // each thread handles one attention head
    float* score_off = gdata->score + eid * H;
    float* grad_score_off = gdata->grad_score + gdata->eid_mapping[eid] * H;
    float* ret_off = gdata->out + gdata->eid_mapping[eid] * H;
    float sds = __ldg(score_off + feat_idx) * __ldg(grad_score_off + feat_idx);
    val += sds;
    *(ret_off + feat_idx) = sds;
  }
  static __device__ __forceinline__ int32_t GetFeatSize(GData *gdata) {
    return gdata->H;
  }
  static __device__ __forceinline__ float* GetOutBuf(GData* gdata) {
    return gdata->accum;
  }
};

struct BackSoftmaxMinus {
  static __device__ __forceinline__ void ApplyEdge(
    int32_t src, int32_t dst, int32_t eid, GData* gdata) {}
  // accum: (N, H)
  static __device__ __forceinline__ void ApplyEdgeReduce(
    int32_t src, int32_t dst, int32_t eid, int32_t feat_idx, float& val, GData* gdata) {
    const int H = gdata->H;
    // each thread handles one attention head
    float* score_off = gdata->score + gdata->eid_mapping[eid] * H;
    float* accum_off = gdata->accum + dst * H;
    float* ret_off = gdata->out + gdata->eid_mapping[eid] * H;
    *(ret_off + feat_idx) -= __ldg(score_off + feat_idx) * __ldg(accum_off + feat_idx);
  }
  static __device__ __forceinline__ int32_t GetFeatSize(GData* gdata) {
    return gdata->H;
  }
  static __device__ __forceinline__ float* GetOutBuf(GData* gdata) {
    return nullptr;
  }
};


void InitGData(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const int32_t N = csr.row_offsets.size() - 1;
  const int32_t M = csr.column_indices.size();
  const int H = gdata->H;
  std::vector<float> accum(N * gdata->H, 0.);
  std::vector<float> score(M * gdata->H, 0.), grad_score(M * gdata->H, 0.), out(M * gdata->H, 0.);
  for (size_t i = 0; i < score.size(); ++i) {
    score[i] = (float)rand() / RAND_MAX;
    grad_score[i] = (float)rand() / RAND_MAX;
  }
  CUDA_CALL(cudaMalloc(&(gdata->accum), sizeof(float) * accum.size()));
  CUDA_CALL(cudaMemcpy(gdata->accum, &accum[0],
        sizeof(float) * accum.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->out), sizeof(float) * out.size()));
  CUDA_CALL(cudaMemcpy(gdata->out, &out[0],
        sizeof(float) * out.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->score), sizeof(float) * score.size()));
  CUDA_CALL(cudaMemcpy(gdata->score, &score[0],
        sizeof(float) * score.size(), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&(gdata->grad_score), sizeof(float) * grad_score.size()));
  CUDA_CALL(cudaMemcpy(gdata->grad_score, &grad_score[0],
        sizeof(float) * grad_score.size(), cudaMemcpyHostToDevice));
  // compute truth
  truth->out = new float[M * H];
  for (size_t u = 0; u < csr.row_offsets.size() - 1; u++) {
    for (int32_t i = csr.row_offsets[u]; i < csr.row_offsets[u + 1]; i++) {
      int32_t v = csr.column_indices[i];
      for (int32_t h = 0; h < H; h++) {
        accum[v * H + h] -= grad_score[i * H + h] * score[i * H + h];
      }
    }
  }
  for (size_t u = 0; u < csr.row_offsets.size() - 1; u++) {
    for (int32_t i = csr.row_offsets[u]; i < csr.row_offsets[u + 1]; i++) {
      int32_t v = csr.column_indices[i];
      for (int32_t h = 0; h < H; h++) {
        truth->out[i * H + h] = accum[v * H + h] * score[i * H + h] + grad_score[i * H + h] * score[i * H + h];
      }
    }
  }
}

void FreeGData(GData* gdata, GData* truth) {
  CUDA_CALL(cudaFree(gdata->score));
  CUDA_CALL(cudaFree(gdata->grad_score));
  CUDA_CALL(cudaFree(gdata->accum));
  CUDA_CALL(cudaFree(gdata->out));
  delete [] truth->out;
}

void CheckResult(const utils::SampleCsr& csr, GData* gdata, GData* truth) {
  const int32_t M = csr.column_indices.size();
  const int H = gdata->H;
  float* h_ret = new float[M * H];
  CUDA_CALL(cudaMemcpy(h_ret, gdata->out, sizeof(float) * M * H, cudaMemcpyDeviceToHost));
  bool equal = utils::IterEqual(h_ret, truth->out, M * H);
  assert(equal);
  delete [] h_ret;
}

}  // masked_mm

#endif
