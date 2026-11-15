// Compact sections of the fill on the GPU: what Horizon's fill_noise_states (terrain.rs) leaves in
// a fresh TerrainChunk, from the GPU's final states (END: state on 30 bits, bits 30 and 31 the
// post-processing marks; (x + z * 16) * h + y).
//
// fill_noise_states writes each non-air cell once, over air, column after column (z then x), from
// top to bottom; BlockSection::set (and fill_run, its equivalent) then evolves the palette this
// way:
// - some air is left: palette [air, states in the order of their first write], counters;
// - no air left, a single state: uniform section;
// - no air left, the state of the last write already seen: the air's entry is free (counter 0);
// - no air left, the state of the last write is new: it takes the air's entry.
// The width is that of the final palette (bits_for_palette(length, 4)), which has only grown.
#define SC_K 128
#define SC_EMPTY 0xFFFFFFFFu
#define SC_F_AIR 1
#define SC_F_FLUID 2
#define SC_F_MOTION 4
// u32 output words per section: [0] kind | bits << 8 | npal << 16, [1] non_air, [2] uniform state,
// [3] u64 data words, [4] 1 if entry 0 is free; then SC_K states, SC_K counters, 1024 u32 data
// words (at most 512 u64 words: 129 entries, 8 bits).
#define SC_STRIDE (5 + 2 * SC_K + 1024)
// Errors
#define SC_E_PALETTE 1
#define SC_E_STATE 2
#define SC_E_MARKS 4

__device__ __forceinline__ int sc_bits(int n) {
  if (n <= 1) return 0;
  int b = 32 - __clz(n - 1);
  return b < 4 ? 4 : b;
}

extern "C" __global__ void ksec_sections(const unsigned *END, const unsigned char *FLAGS, unsigned nflags,
                                         unsigned air, int h, unsigned *OUT, int *ERR) {
  __shared__ unsigned TS[SC_K];
  __shared__ int TMIN[SC_K];
  __shared__ unsigned TC[SC_K];
  __shared__ int RANK[SC_K];
  __shared__ unsigned short ID[4096];
  __shared__ int N, NONAIR, PMAX, D;
  int s = blockIdx.x, t = threadIdx.x;
  for (int k = t; k < SC_K; k += blockDim.x) {
    TS[k] = SC_EMPTY;
    TMIN[k] = 0x7fffffff;
    TC[k] = 0;
  }
  if (t == 0) {
    N = 0;
    NONAIR = 0;
    PMAX = -1;
    D = 0;
  }
  __syncthreads();
  int err = 0;
  for (int c = t; c < 4096; c += blockDim.x) {
    // Neighbouring threads read neighbouring cells of a column
    int y = c & 15, column = c >> 4;
    int x = column & 15, z = column >> 4;
    int i = (y << 8) | column;
    unsigned v = END[column * h + s * 16 + y];
    unsigned st = v & 0x3fffffffu;
    if (st == air) {
      ID[i] = 0xFFFF;
      continue;
    }
    if (st >= nflags) {
      err |= SC_E_STATE;
      ID[i] = 0;
      continue;
    }
    int pos = (z * 16 + x) * 16 + (15 - y);
    unsigned hh = (st * 2654435761u) >> 25;
    int slot = -1;
    for (int p = 0; p < SC_K; ++p) {
      unsigned k = (hh + p) & (SC_K - 1);
      unsigned seen = atomicCAS(&TS[k], SC_EMPTY, st);
      if (seen == SC_EMPTY || seen == st) {
        slot = (int)k;
        break;
      }
    }
    if (slot < 0) {
      err |= SC_E_PALETTE;
      ID[i] = 0;
      continue;
    }
    atomicMin(&TMIN[slot], pos);
    atomicAdd(&TC[slot], 1u);
    atomicMax(&PMAX, (pos << 8) | slot);
    ID[i] = (unsigned short)slot;
    unsigned f = FLAGS[st];
    bool in_run = !(f & SC_F_FLUID) && (v & 0xC0000000u) == 0;
    atomicAdd(&N, 1);
    if (in_run || !(f & SC_F_AIR)) atomicAdd(&NONAIR, 1);
  }
  if (err) atomicOr(ERR, err);
  __syncthreads();
  for (int k = t; k < SC_K; k += blockDim.x) {
    if (TS[k] == SC_EMPTY) continue;
    int r = 0;
    for (int j = 0; j < SC_K; ++j)
      if (TS[j] != SC_EMPTY && TMIN[j] < TMIN[k]) ++r;
    RANK[k] = r;
    atomicAdd(&D, 1);
  }
  __syncthreads();
  int n = N, d = D;
  unsigned *o = OUT + (size_t)s * SC_STRIDE;
  // kind 0: uniform air; 1: uniform; 2: palette
  if (n == 0 || (n == 4096 && d == 1)) {
    if (t == 0) {
      unsigned u = air;
      if (n) {
        for (int k = 0; k < SC_K; ++k)
          if (TS[k] != SC_EMPTY) u = TS[k];
      }
      o[0] = n ? 1u : 0u;
      o[1] = (unsigned)NONAIR;
      o[2] = u;
      o[3] = 0;
      o[4] = 0;
    }
    return;
  }
  // Rank in the palette of each entry of the table, and entry 0
  int last = PMAX & 0xFF, last_pos = PMAX >> 8;
  bool fresh = n == 4096 && TMIN[last] == last_pos;
  int len = fresh ? d : 1 + d;
  int bits = sc_bits(len);
  unsigned *pal = o + 5, *cpt = o + 5 + SC_K;
  // id in the palette of an entry of the table
  auto id_of = [&](int k) -> int {
    if (!fresh) return 1 + RANK[k];
    if (k == last) return 0;
    return 1 + RANK[k] - (RANK[k] > RANK[last] ? 1 : 0);
  };
  for (int k = t; k < SC_K; k += blockDim.x) {
    if (TS[k] == SC_EMPTY) continue;
    int id = id_of(k);
    pal[id] = TS[k];
    cpt[id] = TC[k];
  }
  if (t == 0) {
    if (!fresh) {
      pal[0] = air;
      cpt[0] = (unsigned)(4096 - n);
    }
    o[0] = 2u | ((unsigned)bits << 8) | ((unsigned)len << 16);
    o[1] = (unsigned)NONAIR;
    o[2] = 0;
    o[4] = (n == 4096 && !fresh) ? 1u : 0u;
  }
  __syncthreads();
  // The ids of the cells (air: 0, unless its entry was taken)
  for (int i = t; i < 4096; i += blockDim.x) {
    unsigned short k = ID[i];
    ID[i] = (unsigned short)(k == 0xFFFF ? 0 : id_of(k));
  }
  __syncthreads();
  int par = 64 / bits;
  int nwords = (4096 + par - 1) / par;
  if (t == 0) o[3] = (unsigned)nwords;
  unsigned *words = o + 5 + 2 * SC_K;
  for (int w = t; w < nwords; w += blockDim.x) {
    unsigned long long m = 0;
    for (int k = 0; k < par; ++k) {
      int i = w * par + k;
      if (i < 4096) m |= (unsigned long long)ID[i] << (k * bits);
    }
    words[2 * w] = (unsigned)m;
    words[2 * w + 1] = (unsigned)(m >> 32);
  }
}

// Heightmaps of the fill (OCEAN_FLOOR_WG, WORLD_SURFACE_WG: 1 + the highest opaque cell, 0
// otherwise, from the bottom of the chunk) and number of marks of each column. One block of 128
// threads per column (z * 16 + x), which read the column at once. HM: 512 integers (ocean,
// surface); NM: 512 counters (bit 31 of the written cells, bit 30 of all of them).
extern "C" __global__ void ksec_heights(const unsigned *END, const unsigned char *FLAGS, unsigned nflags,
                                         unsigned air, int h, int *HM, unsigned *NM) {
  __shared__ int OC, SU;
  __shared__ unsigned NA, NB;
  int c = blockIdx.x, t = threadIdx.x;
  if (t == 0) {
    OC = 0;
    SU = 0;
    NA = 0;
    NB = 0;
  }
  __syncthreads();
  const unsigned *col = END + c * h;
  int oc = 0, su = 0;
  unsigned na = 0, nb = 0;
  for (int y = t; y < h; y += blockDim.x) {
    unsigned v = col[y];
    unsigned st = v & 0x3fffffffu;
    if (v & 0x40000000u) ++nb;
    if (st == air) continue;
    if (v & 0x80000000u) ++na;
    unsigned f = st < nflags ? FLAGS[st] : 0;
    if (!(f & SC_F_AIR)) su = y + 1;
    if (f & SC_F_MOTION) oc = y + 1;
  }
  if (oc) atomicMax(&OC, oc);
  if (su) atomicMax(&SU, su);
  if (na) atomicAdd(&NA, na);
  if (nb) atomicAdd(&NB, nb);
  __syncthreads();
  if (t == 0) {
    HM[c] = OC;
    HM[256 + c] = SU;
    NM[c] = NA;
    NM[256 + c] = NB;
  }
}

// Post-processing marks in fill_noise_states order: first bit 31 of the written cells, columns by z
// then x, y going down; then bit 30 of all the cells, columns by x then z, y going down. A mark:
// section << 16 | x | y << 4 | z << 8. One block of 256 threads, one per column; only the columns
// that have marks reread them. MQ: [na, nb, marks a, marks b], cap each.
extern "C" __global__ void ksec_marks(const unsigned *END, unsigned air, int h, const unsigned *NM, unsigned *MQ,
                                        int cap, int *ERR) {
  __shared__ int CA[256], CB[256];
  int t = threadIdx.x;
  int x = t & 15, z = t >> 4;
  int tb = x * 16 + z;
  int na = (int)NM[t], nb = (int)NM[256 + t];
  CA[t] = na;
  CB[tb] = nb;
  __syncthreads();
  if (t == 0) {
    int sa = 0, sb = 0;
    for (int k = 0; k < 256; ++k) {
      int a = CA[k], b = CB[k];
      CA[k] = sa;
      CB[k] = sb;
      sa += a;
      sb += b;
    }
    MQ[0] = (unsigned)sa;
    MQ[1] = (unsigned)sb;
    if (sa > cap || sb > cap) atomicOr(ERR, SC_E_MARKS);
  }
  __syncthreads();
  if (na == 0 && nb == 0) return;
  int oa = CA[t], ob = CB[tb];
  unsigned *A = MQ + 2, *B = MQ + 2 + cap;
  const unsigned *col = END + t * h;
  for (int y = h - 1; y >= 0; --y) {
    unsigned v = col[y];
    unsigned m = (unsigned)(y >> 4) << 16 | (unsigned)(x | (y & 15) << 4 | z << 8);
    if ((v & 0x3fffffffu) != air && (v & 0x80000000u) && oa < cap) A[oa++] = m;
    if ((v & 0x40000000u) && ob < cap) B[ob++] = m;
  }
}
