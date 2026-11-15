// Light of a chunk on the GPU: Horizon's light_chunk (horizon-client-render/src/light.rs), rules of
// vanilla's LightEngine. Region of 3 x 3 chunks, 48 x 48 x h; p = y * 48 * 48 + z * 48 + x.
// State of a cell (LightTables info): emission << 16 | type << 4 | attenuation.
// The final light is the fixed point (maximum over all paths); a fill by waves, in any order,
// reaches it too.
#define LT_W 48
#define LT_WW (48 * 48)
// Error flags
#define LT_E_FRONT 1
#define LT_E_SEEDS 2
#define LT_E_SECTION 4
// A cache section differs from its inline copy (cache test mode)
#define LT_E_CACHE 8

// Input (64-bit words): [0] h | nsec << 16 | sky << 31; [1] present chunks (9 bits); [2] counters
// (seeds, errors: zero when sent);
// then 2 words per (chunk c = dz * 3 + dx, section s < h / 16): word 0 = kind | bits << 8 | palette
// size << 16 | info << 32
// (kind 0 air or absent, 1 uniform, 2 palette, 3 global), word 1 = start of the words | start of
// the palette << 32 (in 64-bit words from the start of the input; the palette is in u32 infos).
// Kinds 4 (palette) and 5 (global): the section is in the section cache (CACHE), its starts counted
// from the start of the cache. Cache test mode (check != 0): from IN + check, 2 words per (chunk,
// section) give the starts in the input of an inline copy (0: none), also read and compared cell by
// cell.
struct LtDesc {
  unsigned kind, bits, npal, info, words, palette;
};
__device__ __forceinline__ LtDesc lt_desc(const unsigned long long *IN, int k) {
  unsigned long long a = IN[3 + 2 * k], b = IN[4 + 2 * k];
  LtDesc d;
  d.kind = (unsigned)(a & 0xFF);
  d.bits = (unsigned)((a >> 8) & 0xFF);
  d.npal = (unsigned)((a >> 16) & 0xFFFF);
  d.info = (unsigned)(a >> 32);
  d.words = (unsigned)(b & 0xFFFFFFFFULL);
  d.palette = (unsigned)(b >> 32);
  return d;
}

// 1. States of the region and emission (states_of); the emitting cells go into the seeds.
// One block of 256 threads per (chunk, section) below h.
extern "C" __global__ void klt_states(const unsigned long long *IN, const unsigned long long *CACHE,
                                     const unsigned *INFO, unsigned short *S, unsigned char *BLK, unsigned *SEEDS,
                                     unsigned *C, unsigned cap_seeds, int nsh, unsigned ninfo, unsigned check) {
  unsigned *NSEEDS = C, *ERR = C + 1;
  int k = blockIdx.x;
  int c = k / nsh, s = k % nsh;
  int cx = c % 3, cz = c / 3;
  LtDesc d = lt_desc(IN, k);
  unsigned kind = d.kind == 4 || d.kind == 5 ? d.kind - 2 : d.kind;
  const unsigned long long *B = d.kind == 4 || d.kind == 5 ? CACHE : IN;
  const unsigned *PAL = reinterpret_cast<const unsigned *>(B + d.palette);
  const unsigned long long *M = B + d.words;
  const unsigned long long *VM = 0;
  const unsigned *VPAL = 0;
  if (check != 0 && (d.kind == 4 || d.kind == 5)) {
    unsigned long long vm = IN[check + 2 * k], vp = IN[check + 2 * k + 1];
    if (vm != 0) {
      VM = IN + vm;
      VPAL = reinterpret_cast<const unsigned *>(IN + vp);
    }
  }
  unsigned bits = d.bits;
  unsigned par = bits ? 64 / bits : 0;
  unsigned long long mask = bits ? ((1ULL << bits) - 1ULL) : 0;
  if ((kind == 2 || kind == 3) && (bits == 0 || bits > 32)) {
    if (threadIdx.x == 0) atomicOr(ERR, LT_E_SECTION);
    return;
  }
  unsigned limit = kind == 2 ? d.npal : ninfo;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    unsigned info = 0;
    if (kind == 1) {
      info = d.info;
    } else if (kind == 2 || kind == 3) {
      unsigned w = i / par, sh = (i % par) * bits;
      unsigned id = (unsigned)((M[w] >> sh) & mask);
      if (id >= limit) {
        atomicOr(ERR, LT_E_SECTION);
        id = 0;
      }
      info = kind == 2 ? PAL[id] : INFO[id];
      if (VM) {
        unsigned vid = (unsigned)((VM[w] >> sh) & mask);
        unsigned vinfo = vid < limit ? (kind == 2 ? VPAL[vid] : INFO[vid]) : 0xFFFFFFFFu;
        if (vinfo != info) atomicOr(ERR, LT_E_CACHE);
      }
    }
    int lx = i & 15, lz = (i >> 4) & 15, ly = i >> 8;
    unsigned p = (unsigned)((s * 16 + ly) * LT_WW + (cz * 16 + lz) * LT_W + cx * 16 + lx);
    S[p] = (unsigned short)(info & 0xFFFF);
    unsigned char e = (unsigned char)(info >> 16);
    BLK[p] = e;
    if (info > 0xFFFF) {
      unsigned n = atomicAdd(NSEEDS, 1u);
      if (n < cap_seeds) SEEDS[n] = p;
      else atomicOr(ERR, LT_E_SEEDS);
    }
  }
}

__device__ __forceinline__ bool lt_present(unsigned presents, int col) {
  int x = col % LT_W, z = col / LT_W;
  return (presents >> ((z / 16) * 3 + x / 16)) & 1;
}
// packed_edge_occluded(high, low)
__device__ __forceinline__ bool lt_edge(const unsigned char *MASKS, int types, unsigned high, unsigned low) {
  return (low & 0xF) != 0 || ((high | low) != 0 && (MASKS[(high >> 4) * types + (low >> 4)] & 1) != 0);
}

// 2. Sky sources: each column goes down to the first occluded edge (LOWEST = last cell at 15), and
//    the whole sky column is written.
extern "C" __global__ void klt_sky(const unsigned long long *IN, const unsigned short *S, unsigned char *SKY,
                                    int *LOWEST, const unsigned char *MASKS, int types) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= LT_WW) return;
  unsigned h = (unsigned)(IN[0] & 0xFFFF);
  bool sky = (IN[0] >> 31) & 1;
  unsigned presents = (unsigned)IN[1];
  int low = (int)h;
  if (sky && lt_present(presents, col)) {
    int y = (int)h - 1;
    unsigned p = (unsigned)((h - 1) * LT_WW + col);
    while (y > 0) {
      unsigned q = p - LT_WW;
      if (lt_edge(MASKS, types, S[p], S[q])) break;
      y--;
      p = q;
    }
    low = y;
  }
  LOWEST[col] = low;
  for (int y = 0; y < (int)h; ++y) SKY[(unsigned)y * LT_WW + col] = (low < (int)h && y >= low) ? 15 : 0;
}

// Raises the light of a cell (bytes, by CAS on the word that holds it); true if it went up.
__device__ __forceinline__ bool lt_raise(unsigned char *L, unsigned p, unsigned v) {
  unsigned *word = reinterpret_cast<unsigned *>(L + (p & ~3u));
  unsigned sh = (p & 3u) * 8;
  unsigned previous = *(volatile unsigned *)word;
  for (;;) {
    unsigned cur = (previous >> sh) & 0xFF;
    if (cur >= v) return false;
    unsigned fresh = (previous & ~(0xFFu << sh)) | (v << sh);
    unsigned seen = atomicCAS(word, previous, fresh);
    if (seen == previous) return true;
    previous = seen;
  }
}

// One wave: each cell of F passes its light to its neighbours (spread of light_chunk).
__device__ void lt_wave(unsigned char *L, const unsigned short *S, const unsigned char *MASKS, int types,
                         unsigned h, unsigned presents, const unsigned *F, unsigned n, unsigned *G, unsigned *NG,
                         unsigned cap, unsigned *ERR) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned p = F[i];
    unsigned level = ((volatile unsigned char *)L)[p];
    if (level <= 1) continue;
    unsigned y = p / LT_WW, col = p % LT_WW;
    unsigned z = col / LT_W, x = col % LT_W;
    unsigned de = S[p];
    const unsigned char *rank = MASKS + (size_t)(de >> 4) * types;
    for (int dir = 0; dir < 6; ++dir) {
      unsigned q;
      if (dir == 0) {
        if (y == 0) continue;
        q = p - LT_WW;
      } else if (dir == 1) {
        if (y + 1 >= h) continue;
        q = p + LT_WW;
      } else if (dir == 2) {
        if (z == 0 || !lt_present(presents, col - LT_W)) continue;
        q = p - LT_W;
      } else if (dir == 3) {
        if (z + 1 >= LT_W || !lt_present(presents, col + LT_W)) continue;
        q = p + LT_W;
      } else if (dir == 4) {
        if (x == 0 || !lt_present(presents, col - 1)) continue;
        q = p - 1;
      } else {
        if (x + 1 >= LT_W || !lt_present(presents, col + 1)) continue;
        q = p + 1;
      }
      unsigned lq = ((volatile unsigned char *)L)[q];
      if (lq + 1 >= level) continue;
      unsigned toward = S[q];
      unsigned att = toward & 0xF;
      if (att < 1) att = 1;
      if (level <= lq + att) continue;
      if (rank[toward >> 4] & (1u << dir)) continue;
      if (lt_raise(L, q, level - att)) {
        unsigned k = atomicAdd(NG, 1u);
        if (k < cap) G[k] = q;
        else atomicOr(ERR, LT_E_FRONT);
      }
    }
  }
}

// Fill of one light up to the fixed point, by waves, within a block.
__device__ void lt_fill(unsigned char *L, const unsigned short *S, const unsigned char *MASKS, int types,
                           unsigned h, unsigned presents, unsigned *F0, unsigned *F1, unsigned cap, unsigned n0,
                           unsigned *ERR, unsigned *NV) {
  unsigned *F = F0, *G = F1;
  unsigned n = n0;
  while (n > 0) {
    if (threadIdx.x == 0) *NV = 0;
    __syncthreads();
    lt_wave(L, S, MASKS, types, h, presents, F, n, G, NV, cap, ERR);
    __threadfence_block();
    __syncthreads();
    n = *NV;
    if (n > cap) n = cap;
    unsigned *t = F;
    F = G;
    G = t;
    __syncthreads();
  }
}

// 3. The sky then the blocks, in a block of 1024 threads. The sky seeds are the source cells of the
//    columns between their own floor and the deepest one of their present neighbours.
extern "C" __global__ void __launch_bounds__(1024, 1) klt_fill(const unsigned long long *IN, const unsigned short *S, unsigned char *SKY,
                                       unsigned char *BLK, const int *LOWEST, const unsigned char *MASKS, int types,
                                       unsigned *F0, unsigned *F1, unsigned cap, const unsigned *SEEDS,
                                       unsigned *C) {
  __shared__ unsigned NV;
  const unsigned *NSEEDS = C;
  unsigned *ERR = C + 1;
  unsigned h = (unsigned)(IN[0] & 0xFFFF);
  bool sky = (IN[0] >> 31) & 1;
  unsigned presents = (unsigned)IN[1];
  if (sky) {
    if (threadIdx.x == 0) NV = 0;
    __syncthreads();
    for (int col = threadIdx.x; col < LT_WW; col += blockDim.x) {
      if (!lt_present(presents, col)) continue;
      int x = col % LT_W, z = col / LT_W;
      int own = LOWEST[col], deep = own;
      if (x > 0 && lt_present(presents, col - 1)) deep = max(deep, LOWEST[col - 1]);
      if (x + 1 < LT_W && lt_present(presents, col + 1)) deep = max(deep, LOWEST[col + 1]);
      if (z > 0 && lt_present(presents, col - LT_W)) deep = max(deep, LOWEST[col - LT_W]);
      if (z + 1 < LT_W && lt_present(presents, col + LT_W)) deep = max(deep, LOWEST[col + LT_W]);
      int de = own > 0 ? own : 0;
      int a = min(deep, (int)h - 1);
      for (int y = de; y <= a; ++y) {
        unsigned p = (unsigned)y * LT_WW + col;
        if (SKY[p] == 15) {
          unsigned k = atomicAdd(&NV, 1u);
          if (k < cap) F0[k] = p;
          else atomicOr(ERR, LT_E_FRONT);
        }
      }
    }
    __syncthreads();
    unsigned n = NV;
    __syncthreads();
    lt_fill(SKY, S, MASKS, types, h, presents, F0, F1, cap, n > cap ? cap : n, ERR, &NV);
  }
  __syncthreads();
  unsigned ns = *NSEEDS;
  for (unsigned i = threadIdx.x; i < ns && i < cap; i += blockDim.x) F0[i] = SEEDS[i];
  __threadfence_block();
  __syncthreads();
  lt_fill(BLK, S, MASKS, types, h, presents, F0, F1, cap, ns > cap ? cap : ns, ERR, &NV);
}

// 4. The sections of the central chunk like pack_lighting then Light::from_slice. Output: 8 bytes
//    (errors, seeds), then per section 8 header bytes (kind 0 uniform, 1 variable sky, 2 variable
//    block, 3 both; first byte) then 4096 bytes (kind 3), or the 2048-byte nibble layer (kinds 1
//    and 2).
#define LT_STEP 4104
extern "C" __global__ void klt_output(const unsigned long long *IN, const unsigned char *SKY, const unsigned char *BLK,
                                      unsigned char *OUT, const unsigned *COUNTERS) {
  __shared__ unsigned char V[4096];
  __shared__ int VS, VB;
  int s = blockIdx.x;
  unsigned h = (unsigned)(IN[0] & 0xFFFF);
  bool sky = (IN[0] >> 31) & 1;
  unsigned char open = sky ? 0xF0 : 0x00;
  if (s == 0 && threadIdx.x == 0) {
    reinterpret_cast<unsigned *>(OUT)[0] = COUNTERS[1];
    reinterpret_cast<unsigned *>(OUT)[1] = COUNTERS[0];
  }
  unsigned char *o = OUT + 8 + (size_t)s * LT_STEP;
  if ((unsigned)s * 16 >= h) {
    if (threadIdx.x == 0) {
      o[0] = 0;
      o[1] = open;
    }
    return;
  }
  if (threadIdx.x == 0) {
    VS = 0;
    VB = 0;
  }
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    int lx = i & 15, lz = (i >> 4) & 15, ly = i >> 8;
    unsigned p = (unsigned)((s * 16 + ly) * LT_WW + (16 + lz) * LT_W + 16 + lx);
    V[i] = (unsigned char)((SKY[p] << 4) | BLK[p]);
  }
  __syncthreads();
  unsigned char first = V[0];
  int vs = 0, vb = 0;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    unsigned char d = V[i] ^ first;
    vs |= (d & 0xF0) != 0;
    vb |= (d & 0x0F) != 0;
  }
  if (vs) atomicOr(&VS, 1);
  if (vb) atomicOr(&VB, 1);
  __syncthreads();
  int kind = (VS ? 1 : 0) | (VB ? 2 : 0);
  if (threadIdx.x == 0) {
    o[0] = (unsigned char)kind;
    o[1] = first;
  }
  unsigned char *d = o + 8;
  if (kind == 3) {
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) d[i] = V[i];
  } else if (kind != 0) {
    int sh = kind == 1 ? 4 : 0;
    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
      d[i] = (unsigned char)(((V[2 * i] >> sh) & 15) | (((V[2 * i + 1] >> sh) & 15) << 4));
  }
}

// Filling the section cache: n copies, each 2 words from IN + at (start in the input | length <<
// 32, then start in the cache); one block per copy.
extern "C" __global__ void klt_cache(const unsigned long long *IN, unsigned long long *CACHE, unsigned at, unsigned n) {
  unsigned k = blockIdx.x;
  if (k >= n) return;
  unsigned long long a = IN[at + 2 * k];
  unsigned long long dst = IN[at + 2 * k + 1];
  unsigned src = (unsigned)(a & 0xFFFFFFFFULL), len = (unsigned)(a >> 32);
  for (unsigned i = threadIdx.x; i < len; i += blockDim.x) CACHE[dst + i] = IN[src + i];
}
