/* Sample code for edge softmax.*/
#include <iostream>
#include <cstdlib>
#include <limits>
#include <dmlc/omp.h>
#include <chrono>

#include <minigun/minigun.h>
#include "../samples_utils.h"
#include "../samples_io.h"

struct GData {
  int32_t dim = 0;
  float* sum{nullptr};  // ndata
  float* max{nullptr};  // ndata
  float* score{nullptr};
};

// Max
struct EdgeMax {
  static inline void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    const int32_t dim = gdata->dim;
    for (int32_t fid = 0; fid < dim; ++fid) {
#pragma omp critical
      gdata->max[dst * dim + fid] = std::max(gdata->max[dst * dim + fid],
          gdata->score[eid * dim + fid]);
    }
  }
};

// minus max, exp and sum
struct MinuxMaxExpSum {
  static inline void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    const int32_t dim = gdata->dim;
    for (int32_t fid = 0; fid < dim; ++fid) {
      gdata->score[eid * dim + fid] = expf(gdata->score[eid * dim + fid] - gdata->max[dst * dim + fid]);
#pragma omp atomic
      gdata->sum[dst * dim + fid] += gdata->score[eid * dim + fid];
    }
  }
};

// norm
struct Norm {
  static inline void ApplyEdge(
      int32_t src, int32_t dst, int32_t eid, GData* gdata) {
    const int32_t dim = gdata->dim;
    for (int32_t fid = 0; fid < dim; ++fid) {
      gdata->score[eid * dim + fid] /= gdata->sum[dst * dim + fid];
    }
  }
};

const int32_t D = 8;  // number of heads

std::vector<float> GroundTruth(
    const std::vector<int32_t>& row_offsets,
    const std::vector<int32_t>& column_indices,
    std::vector<float> score) {
  const size_t N = row_offsets.size() - 1;
  std::vector<float> tmp(N * D, 0.);
  for (size_t i = 0; i < score.size(); ++i) {
    score[i] = std::exp(score[i]);
  }
  for (size_t u = 0; u < row_offsets.size() - 1; ++u) {
    for (int32_t eid = row_offsets[u]; eid < row_offsets[u+1]; ++eid) {
      int32_t v = column_indices[eid];
      for (int32_t idx = 0; idx < D; ++idx) {
        tmp[v * D + idx] += score[eid * D + idx];
      }
    }
  }
  for (size_t eid = 0; eid < column_indices.size(); ++eid) {
    for (int32_t i = 0; i < D; ++i) {
      score[eid * D + i] /= tmp[column_indices[eid] * D + i];
    }
  }
  return score;
}

int main(int argc, char** argv) {
  srand(42);

  // create graph
  std::vector<int32_t> row_offsets, column_indices;
  utils::CreateNPGraph(1000, 0.01, row_offsets, column_indices);
  const int32_t N = row_offsets.size() - 1;
  const int32_t M = column_indices.size();
  std::cout << "#nodes: " << N << " #edges: " << M
    << " #feats: " << D << std::endl;

  // copy graph to gpu
  minigun::IntCsr csr;
  csr.row_offsets.length = row_offsets.size();
  csr.row_offsets.data = &row_offsets[0];
  csr.column_indices.length = column_indices.size();
  csr.column_indices.data = &column_indices[0];

  // Create raw eid_mapping
  minigun::IntArray csr_mapping = utils::arange(0, M, kDLCPU);

  // Create csr_t and coo
  minigun::IntCsr csr_t;
  auto pack = utils::ToReverseCsr(csr, csr_mapping, kDLCPU);
  csr_t = pack.first;
  minigun::IntArray csr_t_mapping = pack.second;
  minigun::IntCoo coo;
  coo = utils::ToCoo(csr, kDLCPU);
  minigun::IntSpMat spmat = {&csr, &coo, &csr_t};

  // Create Runtime Config, not used for cpu
  minigun::advance::RuntimeConfig config;
  config.ctx = {kDLCPU, 0};

  // Create feature data
  std::vector<float> vvec(N * D), evec(M * D);
  for (int32_t i = 0; i < N * D; ++i) {
    vvec[i] = std::numeric_limits<float>::lowest();
  }
  for (int32_t i = 0; i < M * D; ++i) {
    evec[i] = (float)rand() / RAND_MAX - 0.5;
  }
  //utils::VecPrint(evec);

  // Copy feature data to gpu
  GData gdata;
  gdata.dim = D;
  std::vector<float> sum(N * D, 0);
  gdata.score = &evec[0];
  gdata.max = &vvec[0];
  gdata.sum = &sum[0];

  // Compute ground truth
  std::vector<float> truth = GroundTruth(row_offsets, column_indices, evec);
  //utils::VecPrint(truth);

  typedef minigun::advance::Config<minigun::advance::kEdge> Config;
  minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, EdgeMax>(
      config, spmat, &gdata);
  minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, MinuxMaxExpSum>(
      config, spmat, &gdata);
  minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, Norm>(
      config, spmat, &gdata);

  // verify output
  std::cout << "Correct? " << utils::VecEqual(truth, evec) << std::endl;

  const int K = 10;
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, EdgeMax>(
        config, spmat, &gdata);
    minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, MinuxMaxExpSum>(
        config, spmat, &gdata);
    minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, Norm>(
        config, spmat, &gdata);
  }

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < K; ++i) {
    minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, EdgeMax>(
        config, spmat, &gdata);
    minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, MinuxMaxExpSum>(
        config, spmat, &gdata);
    minigun::advance::Advance<kDLCPU, int32_t, float, Config, GData, Norm>(
        config, spmat, &gdata);
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time(ms): " << elapsed_seconds.count() * 1e3 / K << std::endl;
  return 0;
}
