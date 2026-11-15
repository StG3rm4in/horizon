// Biomes of a region of R x R chunks on the GPU: Horizon's BiomeChunk::fill (biome_chunk_or_busy)
// with a multi-noise source.
//
// 1. kbi_climate, one thread per column of quarts: the six climate fields (temperature, humidity,
//    continentalness, erosion, depth, weirdness) in point mode, through cl_column (generated, its
//    Y-free subgraphs computed once per column).
// 2. kbi_search, one thread per quart: the quantization of climate::quantize, then the ClimateIndex
//    search without a starting point.
// 3. kbi_order, one block per chunk: a CPU search starts from the chunk's previous result (`last`);
//    it returns `last` if its distance equals the minimum of the quart, otherwise the first leaf,
//    in traversal order, that reaches this minimum: the one the search without a starting point
//    finds. The block applies `last` in fill order (sections bottom to top, then x, y, z), by
//    tiles.
//
// Quart i of the region: chunk c = z * R + x, then section s, x, y, z (fill order); column of a
// quart: c * 16 + x * 4 + z; QF[(j * columns + column) * 6 + k]: field k at Y quart j.
// Tree: E, 13 integers per node (6 minimums, 6 maximums, squared offset distance); L, 2 per node
// (first child and number of children; for a leaf: its entry and 0).
#define BI_STACK 32
#define BI_TILE 128
#define BI_E_STACK 1
#define BI_E_EMPTY 2
#define BI_E_ID 4

// climate::quantize: (v * 10000f) as i64, saturating, NaN -> 0
__device__ __forceinline__ long long bi_quantize(float v) {
  float p = __fmul_rn(v, 10000.0f);
  if (isnan(p)) return 0;
  if (p >= 9.2233720368547758e18f) return 0x7fffffffffffffffLL;
  if (p <= -9.2233720368547758e18f) return (long long)0x8000000000000000ULL;
  return __float2ll_rz(p);
}
// SearchSpace::distance (Parameter::distance), in modular 64-bit arithmetic
__device__ __forceinline__ long long bi_distance_row(const long long *e, const long long *v) {
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
__device__ __forceinline__ long long bi_distance(const long long *__restrict__ E, int n, const long long *v) {
  return bi_distance_row(E + 13 * (size_t)n, v);
}

extern "C" __global__ void kbi_climate(int cx0, int cz0, int r, int qy0, int nqy, float *QF) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int columns = r * r * 16;
  if (col >= columns) return;
  int c = col >> 4, dx = (col >> 2) & 3, dz = col & 3;
  int qx = (int)((unsigned)(cx0 + c % r) * 4u + (unsigned)dx);
  int qz = (int)((unsigned)(cz0 + c / r) * 4u + (unsigned)dz);
  cl_column((int)((unsigned)qx << 2), (int)((unsigned)qz << 2), qy0, nqy, QF + 6 * (long)col, 6L * columns);
}

// Q: the six quantized values; BEST: the first leaf at the minimum; FMIN: that minimum.
extern "C" __global__ void kbi_search(int r, int nqy, const float *__restrict__ QF, long long *Q, int *BEST,
                                         long long *FMIN, const long long *__restrict__ E,
                                         const unsigned *__restrict__ L, int *ERR) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int par = 16 * nqy;
  if (i >= (long)r * r * par) return;
  int c = (int)(i / par), k = (int)(i % par);
  int s = k >> 6, rest = k & 63;
  int dx = rest >> 4, dy = (rest >> 2) & 3, dz = rest & 3;
  const float *f = QF + ((long)(s * 4 + dy) * (r * r * 16) + c * 16 + dx * 4 + dz) * 6;
  long long v[6];
#pragma unroll
  for (int a = 0; a < 6; ++a) {
    v[a] = bi_quantize(f[a]);
    Q[6 * i + a] = v[a];
  }
  int best = -1;
  long long fit = 0x7fffffffffffffffLL;
  if (L[1] == 0) {
    best = 0;
    fit = bi_distance(E, 0, v);
  } else {
    // SubTree.search, depth first: children in order, a node is visited only if its distance is
    // strictly below the best one, a leaf replaces the best one only if it is strictly closer.
    unsigned pi[BI_STACK], pf[BI_STACK];
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
      long long low = bi_distance(E, child, v);
      if (fit > low) {
        unsigned n = L[2 * child + 1];
        if (n == 0) {
          fit = low;
          best = child;
        } else if (sp < BI_STACK) {
          pi[sp] = L[2 * child];
          pf[sp] = L[2 * child] + n;
          ++sp;
        } else {
          atomicOr(ERR, BI_E_STACK);
          break;
        }
      }
    }
  }
  if (best < 0) {
    atomicOr(ERR, BI_E_EMPTY);
    best = 0;
  }
  BEST[i] = best;
  FMIN[i] = fit;
}

// One block of BI_TILE threads per chunk: `last` in fill order, then the biome id of each quart, in
// the same order (IDS: id of each entry, 0xffffffff if its name has none). Per tile: the block
// loads the quarts, thread 0 runs the chain, the block writes.
extern "C" __global__ void kbi_order(int r, int nqy, const long long *__restrict__ Q,
                                     const int *__restrict__ BEST, const long long *__restrict__ FMIN,
                                     const long long *__restrict__ E, const unsigned *__restrict__ L,
                                     const unsigned *__restrict__ IDS, unsigned *OUT, int *ERR) {
  __shared__ long long sq[6 * BI_TILE];
  __shared__ long long sf[BI_TILE];
  __shared__ int sb[BI_TILE];
  int c = blockIdx.x, t = threadIdx.x;
  int par = 16 * nqy;
  long base = (long)c * par;
  int last = -1;
  long long e[13];
  for (int k0 = 0; k0 < par; k0 += BI_TILE) {
    int n = min(BI_TILE, par - k0);
    if (t < n) {
      sb[t] = BEST[base + k0 + t];
      sf[t] = FMIN[base + k0 + t];
    }
    for (int a = t; a < 6 * n; a += blockDim.x) sq[a] = Q[6 * (base + k0) + a];
    __syncthreads();
    if (t == 0) {
      for (int k = 0; k < n; ++k) {
        int res = sb[k];
        if (last >= 0 && last != res && bi_distance_row(e, sq + 6 * k) == sf[k]) res = last;
        if (res != last) {
          last = res;
#pragma unroll
          for (int a = 0; a < 13; ++a) e[a] = E[13 * (size_t)res + a];
        }
        sb[k] = res;
      }
    }
    __syncthreads();
    if (t < n) {
      unsigned id = IDS[L[2 * sb[t]]];
      if (id == 0xffffffffu) atomicOr(ERR, BI_E_ID);
      OUT[base + k0 + t] = id;
    }
    __syncthreads();
  }
}
