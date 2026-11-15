// Biomes d'une region de R x R chunks sur GPU : BiomeChunk::fill de Horizon (biome_chunk_or_busy) avec
// une source multi-noise.
//
// 1. kbi_climat, un thread par colonne de quarts : les six champs du climat (temperature, humidite,
//    continentalite, erosion, profondeur, bizarrerie) en mode point, par cl_colonne (genere, ses
//    sous-graphes sans Y calcules une fois par colonne).
// 2. kbi_recherche, un thread par quart : la quantification de climate::quantize, puis la recherche
//    de ClimateIndex sans point de depart.
// 3. kbi_ordre, un bloc par chunk : une recherche du CPU part du resultat precedent du chunk (`last`) ;
//    elle rend `last` si sa distance vaut le minimum du quart, sinon la premiere feuille, dans l'ordre
//    du parcours, qui atteint ce minimum : celle de la recherche sans point de depart. Le bloc applique
//    `last` dans l'ordre du remplissage (sections de bas en haut, puis x, y, z), par tuiles.
//
// Quart i de la region : chunk c = z * R + x, puis section s, x, y, z (ordre du remplissage) ; colonne
// d'un quart : c * 16 + x * 4 + z ; QF[(j * colonnes + colonne) * 6 + k] : champ k au quart Y j.
// Arbre : E, 13 entiers par noeud (6 minimums, 6 maximums, distance du decalage au carre) ; L, 2 par
// noeud (premier enfant et nombre d'enfants ; une feuille : son entree et 0).
#define BI_PILE 32
#define BI_TUILE 128
#define BI_E_PILE 1
#define BI_E_VIDE 2
#define BI_E_ID 4

// climate::quantize : (v * 10000f) as i64, sature, NaN -> 0
__device__ __forceinline__ long long bi_quantifier(float v) {
  float p = __fmul_rn(v, 10000.0f);
  if (isnan(p)) return 0;
  if (p >= 9.2233720368547758e18f) return 0x7fffffffffffffffLL;
  if (p <= -9.2233720368547758e18f) return (long long)0x8000000000000000ULL;
  return __float2ll_rz(p);
}
// SearchSpace::distance (Parameter::distance), en arithmetique 64 bits modulaire
__device__ __forceinline__ long long bi_distance_ligne(const long long *e, const long long *v) {
  unsigned long long s = (unsigned long long)e[12];
#pragma unroll
  for (int a = 0; a < 6; ++a) {
    long long au_dessus = (long long)((unsigned long long)v[a] - (unsigned long long)e[6 + a]);
    long long d;
    if (au_dessus > 0) {
      d = au_dessus;
    } else {
      d = (long long)((unsigned long long)e[a] - (unsigned long long)v[a]);
      if (d < 0) d = 0;
    }
    s += (unsigned long long)d * (unsigned long long)d;
  }
  return (long long)s;
}
__device__ __forceinline__ long long bi_distance(const long long *__restrict__ E, int n, const long long *v) {
  return bi_distance_ligne(E + 13 * (size_t)n, v);
}

extern "C" __global__ void kbi_climat(int cx0, int cz0, int r, int qy0, int nqy, float *QF) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int colonnes = r * r * 16;
  if (col >= colonnes) return;
  int c = col >> 4, dx = (col >> 2) & 3, dz = col & 3;
  int qx = (int)((unsigned)(cx0 + c % r) * 4u + (unsigned)dx);
  int qz = (int)((unsigned)(cz0 + c / r) * 4u + (unsigned)dz);
  cl_colonne((int)((unsigned)qx << 2), (int)((unsigned)qz << 2), qy0, nqy, QF + 6 * (long)col, 6L * colonnes);
}

// Q : les six valeurs quantifiees ; BEST : la premiere feuille au minimum ; FMIN : ce minimum.
extern "C" __global__ void kbi_recherche(int r, int nqy, const float *__restrict__ QF, long long *Q, int *BEST,
                                         long long *FMIN, const long long *__restrict__ E,
                                         const unsigned *__restrict__ L, int *ERR) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int par = 16 * nqy;
  if (i >= (long)r * r * par) return;
  int c = (int)(i / par), k = (int)(i % par);
  int s = k >> 6, reste = k & 63;
  int dx = reste >> 4, dy = (reste >> 2) & 3, dz = reste & 3;
  const float *f = QF + ((long)(s * 4 + dy) * (r * r * 16) + c * 16 + dx * 4 + dz) * 6;
  long long v[6];
#pragma unroll
  for (int a = 0; a < 6; ++a) {
    v[a] = bi_quantifier(f[a]);
    Q[6 * i + a] = v[a];
  }
  int meilleur = -1;
  long long fit = 0x7fffffffffffffffLL;
  if (L[1] == 0) {
    meilleur = 0;
    fit = bi_distance(E, 0, v);
  } else {
    // SubTree.search en profondeur : enfants dans l'ordre, un noeud n'est visite que si sa distance est
    // strictement sous la meilleure, une feuille ne remplace la meilleure que si elle est strictement
    // plus proche.
    unsigned pi[BI_PILE], pf[BI_PILE];
    int sp = 1;
    pi[0] = L[0];
    pf[0] = L[0] + L[1];
    while (sp > 0) {
      int t = sp - 1;
      if (pi[t] >= pf[t]) {
        --sp;
        continue;
      }
      int enfant = (int)pi[t]++;
      long long bas = bi_distance(E, enfant, v);
      if (fit > bas) {
        unsigned n = L[2 * enfant + 1];
        if (n == 0) {
          fit = bas;
          meilleur = enfant;
        } else if (sp < BI_PILE) {
          pi[sp] = L[2 * enfant];
          pf[sp] = L[2 * enfant] + n;
          ++sp;
        } else {
          atomicOr(ERR, BI_E_PILE);
          break;
        }
      }
    }
  }
  if (meilleur < 0) {
    atomicOr(ERR, BI_E_VIDE);
    meilleur = 0;
  }
  BEST[i] = meilleur;
  FMIN[i] = fit;
}

// Un bloc de BI_TUILE threads par chunk : `last` dans l'ordre du remplissage, puis l'identifiant du
// biome de chaque quart, dans ce meme ordre (IDS : identifiant de chaque entree, 0xffffffff si le nom
// n'en a pas). Par tuile : le bloc charge les quarts, le thread 0 fait la chaine, le bloc ecrit.
extern "C" __global__ void kbi_ordre(int r, int nqy, const long long *__restrict__ Q,
                                     const int *__restrict__ BEST, const long long *__restrict__ FMIN,
                                     const long long *__restrict__ E, const unsigned *__restrict__ L,
                                     const unsigned *__restrict__ IDS, unsigned *OUT, int *ERR) {
  __shared__ long long sq[6 * BI_TUILE];
  __shared__ long long sf[BI_TUILE];
  __shared__ int sb[BI_TUILE];
  int c = blockIdx.x, t = threadIdx.x;
  int par = 16 * nqy;
  long base = (long)c * par;
  int dernier = -1;
  long long e[13];
  for (int k0 = 0; k0 < par; k0 += BI_TUILE) {
    int n = min(BI_TUILE, par - k0);
    if (t < n) {
      sb[t] = BEST[base + k0 + t];
      sf[t] = FMIN[base + k0 + t];
    }
    for (int a = t; a < 6 * n; a += blockDim.x) sq[a] = Q[6 * (base + k0) + a];
    __syncthreads();
    if (t == 0) {
      for (int k = 0; k < n; ++k) {
        int res = sb[k];
        if (dernier >= 0 && dernier != res && bi_distance_ligne(e, sq + 6 * k) == sf[k]) res = dernier;
        if (res != dernier) {
          dernier = res;
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
