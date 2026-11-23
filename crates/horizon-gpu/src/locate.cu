// Biome searches on the GPU (LocateGpu, locate.rs): the climate and the climate tree search of a
// list of quarts, which the CPU then replays in the order and with the rule of its own caller
// (BiomeSource.findClosestBiome3d's spiral, a map preview's pixels): the GPU cannot apply the
// previous-leaf rule of ClimateSearch itself, as each caller keeps that leaf from one quart to the
// next of its own order (ClimateSearch::find_entry_from_first, index.rs).
//
// 1. kbl_climate, one thread per column: the six climate fields (temperature, humidity,
//    continentalness, erosion, depth, weirdness) in point mode at the column's Y quarts, through
//    cl_layers (generated, its Y-free subgraphs computed once per column). Column c is the quart
//    (XZ[2c], XZ[2c + 1]); its Y quarts are QY[0..nqy) shared by every column or, with `own`, the
//    single QY[c] (nqy is then 1). QF[(c * nqy + j) * 6 + k]: field k at the column's j-th quart.
// 2. kbl_search, one thread per quart: climate::quantize, then the ClimateIndex search without a
//    starting point (depth first, children in order, a node visited only if its distance is
//    strictly below the best one, a leaf kept only if strictly closer). BEST: the first leaf at
//    the smallest distance, in traversal order.
//
// Tree: E, 13 integers per node (6 minimums, 6 maximums, squared offset distance); L, 2 per node
// (first child and number of children; for a leaf: its entry and 0), as in biomes.cu.
//
// The quantization, the distance and the traversal are those of biomes.cu (bi_quantize,
// bi_distance_row, kbi_search), copied so that this module needs neither cl_column nor the region
// kernels: keep them identical. The CPU quantizes the fields again with its own code and only
// takes BEST from here; it checks BEST against its previous leaf and searches itself where a
// distance could wrap.
#define BL_STACK 32
#define BL_E_STACK 1
#define BL_E_EMPTY 2

// climate::quantize: (v * 10000f) as i64, saturating, NaN -> 0
__device__ __forceinline__ long long bl_quantize(float v) {
  float p = __fmul_rn(v, 10000.0f);
  if (isnan(p)) return 0;
  if (p >= 9.2233720368547758e18f) return 0x7fffffffffffffffLL;
  if (p <= -9.2233720368547758e18f) return (long long)0x8000000000000000ULL;
  return __float2ll_rz(p);
}
// SearchSpace::distance (Parameter::distance), in modular 64-bit arithmetic
__device__ __forceinline__ long long bl_distance(const long long *__restrict__ E, int n,
                                                 const long long *v) {
  const long long *e = E + 13 * (size_t)n;
  unsigned long long s = (unsigned long long)e[12];
#pragma unroll
  for (int a = 0; a < 6; ++a) {
    long long above = (long long)((unsigned long long)v[a] - (unsigned long long)e[6 + a]);
    long long d;
    if (above > 0) {
      d = above;
    } else {
      d = (long long)((unsigned long long)e[a] - (unsigned long long)v[a]);
      if (d < 0) d = 0;
    }
    s += (unsigned long long)d * (unsigned long long)d;
  }
  return (long long)s;
}

extern "C" __global__ void kbl_climate(const int *__restrict__ XZ, int ncol,
                                       const int *__restrict__ QY, int nqy, int own, float *QF) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= ncol) return;
  int qx = XZ[2 * col], qz = XZ[2 * col + 1];
  const int *qy = own ? QY + col : QY;
  // ClimateSampler::sample: the position is the quart << 2, wrapping
  cl_layers((int)((unsigned)qx << 2), (int)((unsigned)qz << 2), qy, nqy, QF + 6L * nqy * col, 6L);
}

extern "C" __global__ void kbl_search(int n, const float *__restrict__ QF, int *BEST,
                                      const long long *__restrict__ E,
                                      const unsigned *__restrict__ L, int *ERR) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long v[6];
#pragma unroll
  for (int a = 0; a < 6; ++a) v[a] = bl_quantize(QF[6 * i + a]);
  int best = -1;
  long long fit = 0x7fffffffffffffffLL;
  if (L[1] == 0) {
    best = 0;
  } else {
    unsigned pi[BL_STACK], pf[BL_STACK];
    int sp = 1;
    pi[0] = L[0];
    pf[0] = L[0] + L[1];
    while (sp > 0) {
      int t = sp - 1;
      if (pi[t] >= pf[t]) {
        --sp;
        continue;
      }
      int child = (int)pi[t]++;
      long long low = bl_distance(E, child, v);
      if (fit > low) {
        unsigned c = L[2 * child + 1];
        if (c == 0) {
          fit = low;
          best = child;
        } else if (sp < BL_STACK) {
          pi[sp] = L[2 * child];
          pf[sp] = L[2 * child] + c;
          ++sp;
        } else {
          atomicOr(ERR, BL_E_STACK);
          break;
        }
      }
    }
  }
  if (best < 0) {
    atomicOr(ERR, BL_E_EMPTY);
    best = 0;
  }
  BEST[i] = best;
}
