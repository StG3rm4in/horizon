// Lumiere d'un chunk sur GPU : light_chunk de Horizon (horizon-client-render/src/light.rs), regles du
// LightEngine de vanilla. Region de 3 x 3 chunks, 48 x 48 x h ; p = y * 48 * 48 + z * 48 + x.
// Etat d'une cellule (info de LightTables) : emission << 16 | type << 4 | attenuation.
// La lumiere finale est le point fixe (maximum sur tous les chemins) ; un remplissage par vagues,
// dans n'importe quel ordre, l'atteint aussi.
#define LU_W 48
#define LU_WW (48 * 48)
// Drapeaux d'erreur
#define LU_E_FRONT 1
#define LU_E_SEMIS 2
#define LU_E_SECTION 4
// Une section du cache differe de sa copie en clair (mode test du cache)
#define LU_E_CACHE 8

// Entree (mots de 64 bits) : [0] h | nsec << 16 | ciel << 31 ; [1] presents (9 bits) ; [2] compteurs
// (semis, erreurs : a zero a l'envoi) ;
// puis 2 mots par (chunk c = dz * 3 + dx, section s < h / 16) : mot 0 = genre | bits << 8 |
// taille de la palette << 16 | info << 32
// (genre 0 air ou absent, 1 uniforme, 2 palette, 3 globale), mot 1 = debut des mots | debut de la
// palette << 32 (en mots de 64 bits depuis le debut de l'entree ; la palette est en infos u32).
// Genres 4 (palette) et 5 (globale) : la section est dans le cache des sections (CACHE), ses debuts
// comptes depuis le debut du cache. Mode test du cache (verif != 0) : a partir de IN + verif, 2 mots
// par (chunk, section) donnent les debuts dans l'entree d'une copie en clair (0 : aucune), lue aussi
// et comparee cellule par cellule.
struct LuDesc {
  unsigned genre, bits, npal, info, mots, palette;
};
__device__ __forceinline__ LuDesc lu_desc(const unsigned long long *IN, int k) {
  unsigned long long a = IN[3 + 2 * k], b = IN[4 + 2 * k];
  LuDesc d;
  d.genre = (unsigned)(a & 0xFF);
  d.bits = (unsigned)((a >> 8) & 0xFF);
  d.npal = (unsigned)((a >> 16) & 0xFFFF);
  d.info = (unsigned)(a >> 32);
  d.mots = (unsigned)(b & 0xFFFFFFFFULL);
  d.palette = (unsigned)(b >> 32);
  return d;
}

// 1. Etats de la region et emission (states_of) ; les cellules qui emettent vont dans les semis.
// Un bloc de 256 threads par (chunk, section) sous h.
extern "C" __global__ void klu_etats(const unsigned long long *IN, const unsigned long long *CACHE,
                                     const unsigned *INFO, unsigned short *S, unsigned char *BLK, unsigned *SEMIS,
                                     unsigned *C, unsigned cap_semis, int nsh, unsigned ninfo, unsigned verif) {
  unsigned *NSEMIS = C, *ERR = C + 1;
  int k = blockIdx.x;
  int c = k / nsh, s = k % nsh;
  int cx = c % 3, cz = c / 3;
  LuDesc d = lu_desc(IN, k);
  unsigned genre = d.genre == 4 || d.genre == 5 ? d.genre - 2 : d.genre;
  const unsigned long long *B = d.genre == 4 || d.genre == 5 ? CACHE : IN;
  const unsigned *PAL = reinterpret_cast<const unsigned *>(B + d.palette);
  const unsigned long long *M = B + d.mots;
  const unsigned long long *VM = 0;
  const unsigned *VPAL = 0;
  if (verif != 0 && (d.genre == 4 || d.genre == 5)) {
    unsigned long long vm = IN[verif + 2 * k], vp = IN[verif + 2 * k + 1];
    if (vm != 0) {
      VM = IN + vm;
      VPAL = reinterpret_cast<const unsigned *>(IN + vp);
    }
  }
  unsigned bits = d.bits;
  unsigned par = bits ? 64 / bits : 0;
  unsigned long long masque = bits ? ((1ULL << bits) - 1ULL) : 0;
  if ((genre == 2 || genre == 3) && (bits == 0 || bits > 32)) {
    if (threadIdx.x == 0) atomicOr(ERR, LU_E_SECTION);
    return;
  }
  unsigned limite = genre == 2 ? d.npal : ninfo;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    unsigned info = 0;
    if (genre == 1) {
      info = d.info;
    } else if (genre == 2 || genre == 3) {
      unsigned w = i / par, sh = (i % par) * bits;
      unsigned id = (unsigned)((M[w] >> sh) & masque);
      if (id >= limite) {
        atomicOr(ERR, LU_E_SECTION);
        id = 0;
      }
      info = genre == 2 ? PAL[id] : INFO[id];
      if (VM) {
        unsigned vid = (unsigned)((VM[w] >> sh) & masque);
        unsigned vinfo = vid < limite ? (genre == 2 ? VPAL[vid] : INFO[vid]) : 0xFFFFFFFFu;
        if (vinfo != info) atomicOr(ERR, LU_E_CACHE);
      }
    }
    int lx = i & 15, lz = (i >> 4) & 15, ly = i >> 8;
    unsigned p = (unsigned)((s * 16 + ly) * LU_WW + (cz * 16 + lz) * LU_W + cx * 16 + lx);
    S[p] = (unsigned short)(info & 0xFFFF);
    unsigned char e = (unsigned char)(info >> 16);
    BLK[p] = e;
    if (info > 0xFFFF) {
      unsigned n = atomicAdd(NSEMIS, 1u);
      if (n < cap_semis) SEMIS[n] = p;
      else atomicOr(ERR, LU_E_SEMIS);
    }
  }
}

__device__ __forceinline__ bool lu_present(unsigned presents, int col) {
  int x = col % LU_W, z = col / LU_W;
  return (presents >> ((z / 16) * 3 + x / 16)) & 1;
}
// packed_edge_occluded(haut, bas)
__device__ __forceinline__ bool lu_arete(const unsigned char *MASQUES, int types, unsigned haut, unsigned bas) {
  return (bas & 0xF) != 0 || ((haut | bas) != 0 && (MASQUES[(haut >> 4) * types + (bas >> 4)] & 1) != 0);
}

// 2. Sources du ciel : chaque colonne descend jusqu'a la premiere arete occultee (LOWEST = derniere
// cellule a 15), et toute la colonne du ciel est ecrite.
extern "C" __global__ void klu_ciel(const unsigned long long *IN, const unsigned short *S, unsigned char *SKY,
                                    int *LOWEST, const unsigned char *MASQUES, int types) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= LU_WW) return;
  unsigned h = (unsigned)(IN[0] & 0xFFFF);
  bool ciel = (IN[0] >> 31) & 1;
  unsigned presents = (unsigned)IN[1];
  int bas = (int)h;
  if (ciel && lu_present(presents, col)) {
    int y = (int)h - 1;
    unsigned p = (unsigned)((h - 1) * LU_WW + col);
    while (y > 0) {
      unsigned q = p - LU_WW;
      if (lu_arete(MASQUES, types, S[p], S[q])) break;
      y--;
      p = q;
    }
    bas = y;
  }
  LOWEST[col] = bas;
  for (int y = 0; y < (int)h; ++y) SKY[(unsigned)y * LU_WW + col] = (bas < (int)h && y >= bas) ? 15 : 0;
}

// Releve la lumiere d'une cellule (octets, par CAS sur le mot qui la contient) ; vrai si elle a monte.
__device__ __forceinline__ bool lu_monter(unsigned char *L, unsigned p, unsigned v) {
  unsigned *mot = reinterpret_cast<unsigned *>(L + (p & ~3u));
  unsigned sh = (p & 3u) * 8;
  unsigned ancien = *(volatile unsigned *)mot;
  for (;;) {
    unsigned cur = (ancien >> sh) & 0xFF;
    if (cur >= v) return false;
    unsigned nouveau = (ancien & ~(0xFFu << sh)) | (v << sh);
    unsigned vu = atomicCAS(mot, ancien, nouveau);
    if (vu == ancien) return true;
    ancien = vu;
  }
}

// Une vague : chaque cellule de F passe sa lumiere a ses voisines (spread de light_chunk).
__device__ void lu_vague(unsigned char *L, const unsigned short *S, const unsigned char *MASQUES, int types,
                         unsigned h, unsigned presents, const unsigned *F, unsigned n, unsigned *G, unsigned *NG,
                         unsigned cap, unsigned *ERR) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned p = F[i];
    unsigned niveau = ((volatile unsigned char *)L)[p];
    if (niveau <= 1) continue;
    unsigned y = p / LU_WW, col = p % LU_WW;
    unsigned z = col / LU_W, x = col % LU_W;
    unsigned de = S[p];
    const unsigned char *rang = MASQUES + (size_t)(de >> 4) * types;
    for (int dir = 0; dir < 6; ++dir) {
      unsigned q;
      if (dir == 0) {
        if (y == 0) continue;
        q = p - LU_WW;
      } else if (dir == 1) {
        if (y + 1 >= h) continue;
        q = p + LU_WW;
      } else if (dir == 2) {
        if (z == 0 || !lu_present(presents, col - LU_W)) continue;
        q = p - LU_W;
      } else if (dir == 3) {
        if (z + 1 >= LU_W || !lu_present(presents, col + LU_W)) continue;
        q = p + LU_W;
      } else if (dir == 4) {
        if (x == 0 || !lu_present(presents, col - 1)) continue;
        q = p - 1;
      } else {
        if (x + 1 >= LU_W || !lu_present(presents, col + 1)) continue;
        q = p + 1;
      }
      unsigned lq = ((volatile unsigned char *)L)[q];
      if (lq + 1 >= niveau) continue;
      unsigned vers = S[q];
      unsigned att = vers & 0xF;
      if (att < 1) att = 1;
      if (niveau <= lq + att) continue;
      if (rang[vers >> 4] & (1u << dir)) continue;
      if (lu_monter(L, q, niveau - att)) {
        unsigned k = atomicAdd(NG, 1u);
        if (k < cap) G[k] = q;
        else atomicOr(ERR, LU_E_FRONT);
      }
    }
  }
}

// Remplissage d'une lumiere jusqu'au point fixe, par vagues, dans un bloc.
__device__ void lu_remplir(unsigned char *L, const unsigned short *S, const unsigned char *MASQUES, int types,
                           unsigned h, unsigned presents, unsigned *F0, unsigned *F1, unsigned cap, unsigned n0,
                           unsigned *ERR, unsigned *NV) {
  unsigned *F = F0, *G = F1;
  unsigned n = n0;
  while (n > 0) {
    if (threadIdx.x == 0) *NV = 0;
    __syncthreads();
    lu_vague(L, S, MASQUES, types, h, presents, F, n, G, NV, cap, ERR);
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

// 3. Le ciel puis les blocs, dans un bloc de 1024 threads. Les semis du ciel sont les cellules sources
// des colonnes entre leur propre fond et le plus profond de leurs voisines presentes.
extern "C" __global__ void __launch_bounds__(1024, 1) klu_remplir(const unsigned long long *IN, const unsigned short *S, unsigned char *SKY,
                                       unsigned char *BLK, const int *LOWEST, const unsigned char *MASQUES, int types,
                                       unsigned *F0, unsigned *F1, unsigned cap, const unsigned *SEMIS,
                                       unsigned *C) {
  __shared__ unsigned NV;
  const unsigned *NSEMIS = C;
  unsigned *ERR = C + 1;
  unsigned h = (unsigned)(IN[0] & 0xFFFF);
  bool ciel = (IN[0] >> 31) & 1;
  unsigned presents = (unsigned)IN[1];
  if (ciel) {
    if (threadIdx.x == 0) NV = 0;
    __syncthreads();
    for (int col = threadIdx.x; col < LU_WW; col += blockDim.x) {
      if (!lu_present(presents, col)) continue;
      int x = col % LU_W, z = col / LU_W;
      int propre = LOWEST[col], profond = propre;
      if (x > 0 && lu_present(presents, col - 1)) profond = max(profond, LOWEST[col - 1]);
      if (x + 1 < LU_W && lu_present(presents, col + 1)) profond = max(profond, LOWEST[col + 1]);
      if (z > 0 && lu_present(presents, col - LU_W)) profond = max(profond, LOWEST[col - LU_W]);
      if (z + 1 < LU_W && lu_present(presents, col + LU_W)) profond = max(profond, LOWEST[col + LU_W]);
      int de = propre > 0 ? propre : 0;
      int a = min(profond, (int)h - 1);
      for (int y = de; y <= a; ++y) {
        unsigned p = (unsigned)y * LU_WW + col;
        if (SKY[p] == 15) {
          unsigned k = atomicAdd(&NV, 1u);
          if (k < cap) F0[k] = p;
          else atomicOr(ERR, LU_E_FRONT);
        }
      }
    }
    __syncthreads();
    unsigned n = NV;
    __syncthreads();
    lu_remplir(SKY, S, MASQUES, types, h, presents, F0, F1, cap, n > cap ? cap : n, ERR, &NV);
  }
  __syncthreads();
  unsigned ns = *NSEMIS;
  for (unsigned i = threadIdx.x; i < ns && i < cap; i += blockDim.x) F0[i] = SEMIS[i];
  __threadfence_block();
  __syncthreads();
  lu_remplir(BLK, S, MASQUES, types, h, presents, F0, F1, cap, ns > cap ? cap : ns, ERR, &NV);
}

// 4. Les sections du chunk central comme pack_lighting puis Light::from_slice. Sortie : 8 octets
// (erreurs, semis), puis par section 8 octets
// d'en-tete (genre 0 uniforme, 1 ciel variable, 2 bloc variable, 3 les deux ; premier octet) puis
// 4096 octets (genre 3), ou la couche de nibbles de 2048 octets (genres 1 et 2).
#define LU_PAS 4104
extern "C" __global__ void klu_sortie(const unsigned long long *IN, const unsigned char *SKY, const unsigned char *BLK,
                                      unsigned char *OUT, const unsigned *COMPTEURS) {
  __shared__ unsigned char V[4096];
  __shared__ int VS, VB;
  int s = blockIdx.x;
  unsigned h = (unsigned)(IN[0] & 0xFFFF);
  bool ciel = (IN[0] >> 31) & 1;
  unsigned char ouvert = ciel ? 0xF0 : 0x00;
  if (s == 0 && threadIdx.x == 0) {
    reinterpret_cast<unsigned *>(OUT)[0] = COMPTEURS[1];
    reinterpret_cast<unsigned *>(OUT)[1] = COMPTEURS[0];
  }
  unsigned char *o = OUT + 8 + (size_t)s * LU_PAS;
  if ((unsigned)s * 16 >= h) {
    if (threadIdx.x == 0) {
      o[0] = 0;
      o[1] = ouvert;
    }
    return;
  }
  if (threadIdx.x == 0) {
    VS = 0;
    VB = 0;
  }
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    int lx = i & 15, lz = (i >> 4) & 15, ly = i >> 8;
    unsigned p = (unsigned)((s * 16 + ly) * LU_WW + (16 + lz) * LU_W + 16 + lx);
    V[i] = (unsigned char)((SKY[p] << 4) | BLK[p]);
  }
  __syncthreads();
  unsigned char premier = V[0];
  int vs = 0, vb = 0;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    unsigned char d = V[i] ^ premier;
    vs |= (d & 0xF0) != 0;
    vb |= (d & 0x0F) != 0;
  }
  if (vs) atomicOr(&VS, 1);
  if (vb) atomicOr(&VB, 1);
  __syncthreads();
  int genre = (VS ? 1 : 0) | (VB ? 2 : 0);
  if (threadIdx.x == 0) {
    o[0] = (unsigned char)genre;
    o[1] = premier;
  }
  unsigned char *d = o + 8;
  if (genre == 3) {
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) d[i] = V[i];
  } else if (genre != 0) {
    int sh = genre == 1 ? 4 : 0;
    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
      d[i] = (unsigned char)(((V[2 * i] >> sh) & 15) | (((V[2 * i + 1] >> sh) & 15) << 4));
  }
}

// Le remplissage du cache des sections : n copies, chacune 2 mots a partir de IN + at (debut dans
// l'entree | longueur << 32, puis debut dans le cache) ; un bloc par copie.
extern "C" __global__ void klu_cache(const unsigned long long *IN, unsigned long long *CACHE, unsigned at, unsigned n) {
  unsigned k = blockIdx.x;
  if (k >= n) return;
  unsigned long long a = IN[at + 2 * k];
  unsigned long long dst = IN[at + 2 * k + 1];
  unsigned src = (unsigned)(a & 0xFFFFFFFFULL), len = (unsigned)(a >> 32);
  for (unsigned i = threadIdx.x; i < len; i += blockDim.x) CACHE[dst + i] = IN[src + i];
}
