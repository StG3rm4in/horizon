// Sections compactes du remplissage sur GPU : ce que fill_noise_states de Horizon (terrain.rs) laisse
// dans un TerrainChunk neuf, a partir des etats finaux du GPU (FIN : etat sur 30 bits, bit 30 et bit 31
// les marques de post-traitement ; (x + z * 16) * h + y).
//
// fill_noise_states ecrit chaque cellule non air une fois, sur de l'air, colonne apres colonne (z puis x),
// de haut en bas ; BlockSection::set (et fill_run, equivalent) fait alors evoluer la palette ainsi :
// - il reste de l'air : palette [air, etats dans l'ordre de leur premiere ecriture], compteurs ;
// - plus d'air, un seul etat : section uniforme ;
// - plus d'air, l'etat de la derniere ecriture deja vu : la case de l'air est libre (compteur 0) ;
// - plus d'air, l'etat de la derniere ecriture nouveau : il prend la case de l'air.
// La largeur est celle de la palette finale (bits_for_palette(longueur, 4)), qui n'a fait que grandir.
#define SC_K 128
#define SC_VIDE 0xFFFFFFFFu
#define SC_F_AIR 1
#define SC_F_FLUIDE 2
#define SC_F_MOUVEMENT 4
// Mots u32 de sortie par section : [0] genre | bits << 8 | npal << 16, [1] non_air, [2] etat uniforme,
// [3] mots u64 de donnees, [4] 1 si la case 0 est libre ; puis SC_K etats, SC_K compteurs, 1024 mots u32
// de donnees (au plus 512 mots u64 : 129 entrees, 8 bits).
#define SC_PAS (5 + 2 * SC_K + 1024)
// Erreurs
#define SC_E_PALETTE 1
#define SC_E_ETAT 2
#define SC_E_MARQUES 4

__device__ __forceinline__ int sc_bits(int n) {
  if (n <= 1) return 0;
  int b = 32 - __clz(n - 1);
  return b < 4 ? 4 : b;
}

extern "C" __global__ void ksec_sections(const unsigned *FIN, const unsigned char *FLAGS, unsigned nflags,
                                         unsigned air, int h, unsigned *OUT, int *ERR) {
  __shared__ unsigned TS[SC_K];
  __shared__ int TMIN[SC_K];
  __shared__ unsigned TC[SC_K];
  __shared__ int RANG[SC_K];
  __shared__ unsigned short ID[4096];
  __shared__ int N, NONAIR, PMAX, D;
  int s = blockIdx.x, t = threadIdx.x;
  for (int k = t; k < SC_K; k += blockDim.x) {
    TS[k] = SC_VIDE;
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
    // Les threads voisins lisent des cellules voisines d'une colonne
    int y = c & 15, colonne = c >> 4;
    int x = colonne & 15, z = colonne >> 4;
    int i = (y << 8) | colonne;
    unsigned v = FIN[colonne * h + s * 16 + y];
    unsigned st = v & 0x3fffffffu;
    if (st == air) {
      ID[i] = 0xFFFF;
      continue;
    }
    if (st >= nflags) {
      err |= SC_E_ETAT;
      ID[i] = 0;
      continue;
    }
    int pos = (z * 16 + x) * 16 + (15 - y);
    unsigned hh = (st * 2654435761u) >> 25;
    int slot = -1;
    for (int p = 0; p < SC_K; ++p) {
      unsigned k = (hh + p) & (SC_K - 1);
      unsigned vu = atomicCAS(&TS[k], SC_VIDE, st);
      if (vu == SC_VIDE || vu == st) {
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
    bool suite = !(f & SC_F_FLUIDE) && (v & 0xC0000000u) == 0;
    atomicAdd(&N, 1);
    if (suite || !(f & SC_F_AIR)) atomicAdd(&NONAIR, 1);
  }
  if (err) atomicOr(ERR, err);
  __syncthreads();
  for (int k = t; k < SC_K; k += blockDim.x) {
    if (TS[k] == SC_VIDE) continue;
    int r = 0;
    for (int j = 0; j < SC_K; ++j)
      if (TS[j] != SC_VIDE && TMIN[j] < TMIN[k]) ++r;
    RANG[k] = r;
    atomicAdd(&D, 1);
  }
  __syncthreads();
  int n = N, d = D;
  unsigned *o = OUT + (size_t)s * SC_PAS;
  // genre 0 : air uniforme ; 1 : uniforme ; 2 : palette
  if (n == 0 || (n == 4096 && d == 1)) {
    if (t == 0) {
      unsigned u = air;
      if (n) {
        for (int k = 0; k < SC_K; ++k)
          if (TS[k] != SC_VIDE) u = TS[k];
      }
      o[0] = n ? 1u : 0u;
      o[1] = (unsigned)NONAIR;
      o[2] = u;
      o[3] = 0;
      o[4] = 0;
    }
    return;
  }
  // Rang dans la palette de chaque case de la table, et la case 0
  int dernier = PMAX & 0xFF, pos_dernier = PMAX >> 8;
  bool nouveau = n == 4096 && TMIN[dernier] == pos_dernier;
  int len = nouveau ? d : 1 + d;
  int bits = sc_bits(len);
  unsigned *pal = o + 5, *cpt = o + 5 + SC_K;
  // id dans la palette d'une case de la table
  auto id_de = [&](int k) -> int {
    if (!nouveau) return 1 + RANG[k];
    if (k == dernier) return 0;
    return 1 + RANG[k] - (RANG[k] > RANG[dernier] ? 1 : 0);
  };
  for (int k = t; k < SC_K; k += blockDim.x) {
    if (TS[k] == SC_VIDE) continue;
    int id = id_de(k);
    pal[id] = TS[k];
    cpt[id] = TC[k];
  }
  if (t == 0) {
    if (!nouveau) {
      pal[0] = air;
      cpt[0] = (unsigned)(4096 - n);
    }
    o[0] = 2u | ((unsigned)bits << 8) | ((unsigned)len << 16);
    o[1] = (unsigned)NONAIR;
    o[2] = 0;
    o[4] = (n == 4096 && !nouveau) ? 1u : 0u;
  }
  __syncthreads();
  // Les ids des cellules (l'air : 0, sauf si sa case a ete prise)
  for (int i = t; i < 4096; i += blockDim.x) {
    unsigned short k = ID[i];
    ID[i] = (unsigned short)(k == 0xFFFF ? 0 : id_de(k));
  }
  __syncthreads();
  int par = 64 / bits;
  int nmots = (4096 + par - 1) / par;
  if (t == 0) o[3] = (unsigned)nmots;
  unsigned *mots = o + 5 + 2 * SC_K;
  for (int w = t; w < nmots; w += blockDim.x) {
    unsigned long long m = 0;
    for (int k = 0; k < par; ++k) {
      int i = w * par + k;
      if (i < 4096) m |= (unsigned long long)ID[i] << (k * bits);
    }
    mots[2 * w] = (unsigned)m;
    mots[2 * w + 1] = (unsigned)(m >> 32);
  }
}

// Cartes de hauteur du remplissage (OCEAN_FLOOR_WG, WORLD_SURFACE_WG : 1 + la plus haute cellule opaque,
// 0 sinon, depuis le bas du chunk) et nombre de marques de chaque colonne. Un bloc de 128 threads par
// colonne (z * 16 + x), qui lisent la colonne d'un coup. HM : 512 entiers (ocean, surface) ; NM : 512
// compteurs (bit 31 des cellules ecrites, bit 30 de toutes).
extern "C" __global__ void ksec_hauteurs(const unsigned *FIN, const unsigned char *FLAGS, unsigned nflags,
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
  const unsigned *col = FIN + c * h;
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
    if (f & SC_F_MOUVEMENT) oc = y + 1;
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

// Marques de post-traitement dans l'ordre de fill_noise_states : d'abord le bit 31 des cellules ecrites,
// colonnes en z puis x, y descendant ; puis le bit 30 de toutes les cellules, colonnes en x puis z,
// y descendant. Une marque : section << 16 | x | y << 4 | z << 8. Un bloc de 256 threads, un par colonne ;
// seules les colonnes qui en ont les relisent. MQ : [na, nb, marques a, marques b], cap chacune.
extern "C" __global__ void ksec_marques(const unsigned *FIN, unsigned air, int h, const unsigned *NM, unsigned *MQ,
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
    if (sa > cap || sb > cap) atomicOr(ERR, SC_E_MARQUES);
  }
  __syncthreads();
  if (na == 0 && nb == 0) return;
  int oa = CA[t], ob = CB[tb];
  unsigned *A = MQ + 2, *B = MQ + 2 + cap;
  const unsigned *col = FIN + t * h;
  for (int y = h - 1; y >= 0; --y) {
    unsigned v = col[y];
    unsigned m = (unsigned)(y >> 4) << 16 | (unsigned)(x | (y & 15) << 4 | z << 8);
    if ((v & 0x3fffffffu) != air && (v & 0x80000000u) && oa < cap) A[oa++] = m;
    if ((v & 0x40000000u) && ob < cap) B[ob++] = m;
  }
}
