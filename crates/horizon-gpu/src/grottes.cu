// Masque des carvers d'un chunk (carver.rs, carver/canyon.rs, horizon_core::carving de Horizon, 26.3).
// Un bloc par element planifie (salle, tunnel de grotte, canyon) : le thread 0 trace le chemin et garde
// les ellipsoides qui atteignent le chunk ; tout le bloc les rasterise dans un masque partage, reuni au
// masque global par atomicOr. Le masque est une union : l'ordre des elements ne compte pas.
// Calculs exacts : float et double avec les intrinsics _rn, sans contraction (--fmad=false), table de
// sinus de Horizon (sine_table) en memoire globale.
//
// Sortie (mots de 64 bits) : NW mots de masque, NC compteurs de cellules testees par appel de carver
// (budget de 2^24), puis un mot de drapeaux d'erreur (le chunk repasse alors par le CPU).
#define GR_SALLE 0
#define GR_TUNNEL 1
#define GR_CANYON 2
#define GR_EV_MAX 512
#define GR_NY_MAX 512
#define GR_MOTS_MAX 1536
#define GR_PILE 8
// Drapeaux
#define GR_E_EVENEMENTS 1
#define GR_E_HAUTEUR 2
#define GR_E_BORNE 4
#define GR_E_PILE 8
#define GR_E_MASQUE 16
#define GR_E_LARGEUR 32
#define GR_E_FOURNISSEUR 64

#define GR_SCALE 10430.378350470453
#define GR_PI_F 3.14159265358979323846f
#define GR_FRAC_PI_2_F 1.57079632679489661923f

// Element planifie, 16 mots (le meme #[repr(C)] que CaveItem cote Rust)
struct GrItem {
  unsigned long long seed;  // tunnel : graine de LegacyRandom::new ; canyon : etat apres la table des largeurs
  double x, y, z;
  double echelle;     // tunnel : vertical_scale ; canyon : y_scale
  double horizontal;  // tunnel : multiplicateur du run ; salle : rayon horizontal
  double vertical;    // tunnel : multiplicateur du run ; salle : rayon vertical
  double plancher;    // grotte : floor_level
  unsigned genre, appel;
  int fin;
  unsigned largeur;  // canyon : debut de sa table des largeurs
  float epaisseur, lacet, tangage;
  unsigned fournisseur;  // canyon, horizontal_radius_factor : 0 constant (fmin), 1 uniforme [fmin, fmax)
  float fmin, fmax, vdefaut, vcentre;
  unsigned bourrage[4];
};

struct Lr {
  unsigned long long s;
};
__device__ __forceinline__ void lr_graine(Lr &r, long long g) {
  r.s = ((unsigned long long)g ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
}
__device__ __forceinline__ int lr_bits(Lr &r, int n) {
  r.s = (r.s * 0x5DEECE66DULL + 11ULL) & ((1ULL << 48) - 1);
  return (int)(unsigned)(r.s >> (48 - n));
}
__device__ __forceinline__ float lr_float(Lr &r) { return __fmul_rn((float)lr_bits(r, 24), 1.0f / 16777216.0f); }
__device__ __forceinline__ int lr_int(Lr &r, int borne, unsigned *err) {
  if (borne <= 0) {
    *err |= GR_E_BORNE;
    return 0;
  }
  if ((borne & (borne - 1)) == 0) return (int)(((long long)borne * (long long)lr_bits(r, 31)) >> 31);
  for (;;) {
    int bits = lr_bits(r, 31);
    int v = bits % borne;
    if ((int)((unsigned)bits - (unsigned)v + (unsigned)(borne - 1)) >= 0) return v;
  }
}
__device__ __forceinline__ long long lr_long(Lr &r) {
  long long haut = (long long)lr_bits(r, 32) << 32;
  return haut + (long long)lr_bits(r, 32);
}

__device__ __forceinline__ float gr_sin(const float *T, double a) {
  long long i = __double2ll_rz(__dmul_rn(a, GR_SCALE));
  return T[i & 65535];
}
__device__ __forceinline__ float gr_cos(const float *T, double a) {
  long long i = __double2ll_rz(__dadd_rn(__dmul_rn(a, GR_SCALE), 16384.0));
  return T[i & 65535];
}
// can_reach : epaisseur arrondie en float, distances en double
__device__ __forceinline__ bool gr_atteint(int cx, int cz, double x, double z, int pas, int fin, float ep) {
  double dx = __dsub_rn(x, (double)(cx * 16 + 8));
  double dz = __dsub_rn(z, (double)(cz * 16 + 8));
  double reste = (double)(fin - pas);
  double r = (double)__fadd_rn(__fadd_rn(ep, 2.0f), 16.0f);
  return __dsub_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dz, dz)), __dmul_rn(reste, reste)) <= __dmul_rn(r, r);
}

struct Ev {
  double x, y, z, h, v;
};

struct Tun {
  long long graine;
  double x, y, z;
  float ep, lacet, tangage;
  int debut, fin;
  double echelle;
};

// Garde un ellipsoide s'il touche le chunk (test de portee de carve_ellipsoid_with_budget)
__device__ __forceinline__ void gr_garder(Ev *E, int *n, unsigned *err, int cx, int cz, double x, double y, double z,
                                          double h, double v) {
  double portee = __dadd_rn(16.0, __dmul_rn(h, 2.0));
  if (fabs(__dsub_rn(x, (double)(cx * 16 + 8))) > portee || fabs(__dsub_rn(z, (double)(cz * 16 + 8))) > portee) return;
  if (*n >= GR_EV_MAX) {
    *err |= GR_E_EVENEMENTS;
    return;
  }
  Ev e;
  e.x = x;
  e.y = y;
  e.z = z;
  e.h = h;
  e.v = v;
  E[(*n)++] = e;
}

// CaveRun::tunnel, avec une pile pour les branches (la premiere tracee avant la seconde)
__device__ void gr_tunnels(const GrItem &it, const float *T, int cx, int cz, Ev *E, int *n, unsigned *err) {
  Tun pile[GR_PILE];
  int np = 1;
  pile[0].graine = (long long)it.seed;
  pile[0].x = it.x;
  pile[0].y = it.y;
  pile[0].z = it.z;
  pile[0].ep = it.epaisseur;
  pile[0].lacet = it.lacet;
  pile[0].tangage = it.tangage;
  pile[0].debut = 0;
  pile[0].fin = it.fin;
  pile[0].echelle = it.echelle;
  while (np > 0) {
    Tun t = pile[--np];
    Lr r;
    lr_graine(r, t.graine);
    int partage = lr_int(r, t.fin / 2, err) + t.fin / 4;
    bool doux = lr_int(r, 6, err) == 0;
    if (*err) return;
    float dlacet = 0.0f, dtangage = 0.0f;
    for (int pas = t.debut; pas < t.fin; ++pas) {
      double rayon = __dadd_rn(1.5, (double)__fmul_rn(gr_sin(T, (double)__fdiv_rn(__fmul_rn(GR_PI_F, (float)pas), (float)t.fin)), t.ep));
      double vert = __dmul_rn(rayon, t.echelle);
      float ct = gr_cos(T, (double)t.tangage);
      t.x = __dadd_rn(t.x, (double)__fmul_rn(gr_cos(T, (double)t.lacet), ct));
      t.y = __dadd_rn(t.y, (double)gr_sin(T, (double)t.tangage));
      t.z = __dadd_rn(t.z, (double)__fmul_rn(gr_sin(T, (double)t.lacet), ct));
      t.tangage = __fmul_rn(t.tangage, doux ? 0.92f : 0.7f);
      t.tangage = __fadd_rn(t.tangage, __fmul_rn(dtangage, 0.1f));
      t.lacet = __fadd_rn(t.lacet, __fmul_rn(dlacet, 0.1f));
      dtangage = __fmul_rn(dtangage, 0.9f);
      dlacet = __fmul_rn(dlacet, 0.75f);
      {
        float a = lr_float(r), b = lr_float(r), c = lr_float(r);
        dtangage = __fadd_rn(dtangage, __fmul_rn(__fmul_rn(__fsub_rn(a, b), c), 2.0f));
      }
      {
        float a = lr_float(r), b = lr_float(r), c = lr_float(r);
        dlacet = __fadd_rn(dlacet, __fmul_rn(__fmul_rn(__fsub_rn(a, b), c), 4.0f));
      }
      if (pas == partage && t.ep > 1.0f) {
        if (np + 2 > GR_PILE) {
          *err |= GR_E_PILE;
          return;
        }
        Tun f[2];
        for (int k = 0; k < 2; ++k) {
          f[k] = t;
          f[k].graine = lr_long(r);
          f[k].ep = __fadd_rn(__fmul_rn(lr_float(r), 0.5f), 0.5f);
          f[k].lacet = k == 0 ? __fsub_rn(t.lacet, GR_FRAC_PI_2_F) : __fadd_rn(t.lacet, GR_FRAC_PI_2_F);
          f[k].tangage = __fdiv_rn(t.tangage, 3.0f);
          f[k].debut = pas;
          f[k].echelle = 1.0;
        }
        pile[np++] = f[1];
        pile[np++] = f[0];
        break;
      }
      if (lr_int(r, 4, err) == 0) continue;
      if (!gr_atteint(cx, cz, t.x, t.z, pas, t.fin, t.ep)) break;
      gr_garder(E, n, err, cx, cz, t.x, t.y, t.z, __dmul_rn(rayon, it.horizontal), __dmul_rn(vert, it.vertical));
    }
  }
}

// CanyonCarver::trace
__device__ void gr_canyon(const GrItem &it, const float *T, int cx, int cz, Ev *E, int *n, unsigned *err) {
  Lr r;
  r.s = it.seed;
  double x = it.x, y = it.y, z = it.z;
  float lacet = it.lacet, tangage = it.tangage, dlacet = 0.0f, dtangage = 0.0f;
  int fin = it.fin;
  for (int pas = 0; pas < fin; ++pas) {
    double h = __dadd_rn(1.5, (double)__fmul_rn(gr_sin(T, (double)__fdiv_rn(__fmul_rn((float)pas, GR_PI_F), (float)fin)), it.epaisseur));
    double v = __dmul_rn(h, it.echelle);
    float f = it.fmin;
    if (it.fournisseur == 1) {
      f = __fadd_rn(__fmul_rn(lr_float(r), __fsub_rn(it.fmax, it.fmin)), it.fmin);
    } else if (it.fournisseur != 0) {
      *err |= GR_E_FOURNISSEUR;
      return;
    }
    h = __dmul_rn(h, (double)f);
    float centre = __fsub_rn(1.0f, __fmul_rn(fabsf(__fsub_rn(0.5f, __fdiv_rn((float)pas, (float)fin))), 2.0f));
    float mult = __fadd_rn(it.vdefaut, __fmul_rn(it.vcentre, centre));
    v = __dmul_rn(__dmul_rn((double)mult, v), (double)__fadd_rn(__fmul_rn(lr_float(r), 0.25f), 0.75f));
    float ct = gr_cos(T, (double)tangage);
    float st = gr_sin(T, (double)tangage);
    x = __dadd_rn(x, (double)__fmul_rn(gr_cos(T, (double)lacet), ct));
    y = __dadd_rn(y, (double)st);
    z = __dadd_rn(z, (double)__fmul_rn(gr_sin(T, (double)lacet), ct));
    tangage = __fmul_rn(tangage, 0.7f);
    tangage = __fadd_rn(tangage, __fmul_rn(dtangage, 0.05f));
    lacet = __fadd_rn(lacet, __fmul_rn(dlacet, 0.05f));
    dtangage = __fmul_rn(dtangage, 0.8f);
    dlacet = __fmul_rn(dlacet, 0.5f);
    {
      float a = lr_float(r), b = lr_float(r), c = lr_float(r);
      dtangage = __fadd_rn(dtangage, __fmul_rn(__fmul_rn(__fsub_rn(a, b), c), 2.0f));
    }
    {
      float a = lr_float(r), b = lr_float(r), c = lr_float(r);
      dlacet = __fadd_rn(dlacet, __fmul_rn(__fmul_rn(__fsub_rn(a, b), c), 4.0f));
    }
    if (lr_int(r, 4, err) == 0) continue;
    if (!gr_atteint(cx, cz, x, z, pas, fin, it.epaisseur)) return;
    gr_garder(E, n, err, cx, cz, x, y, z, h, v);
  }
}

// IN : elements (16 mots chacun) ; W : tables des largeurs des canyons (nlarg chacune) ; T : table de sinus.
// OUT : NW mots de masque, NC compteurs, 1 mot de drapeaux (a zero avant le lancement).
// Masque : bit (y - ymin) + (z + x * 16) * hauteur ; cellules testees de y = ymax(e) a ymin(e) + 1.
extern "C" __global__ void kgr_trace(const unsigned long long *IN, const float *W, const float *T,
                                     unsigned long long *OUT, int nw, int nc, int cx, int cz, int ymin,
                                     int ymax, int hauteur, int gen_ymin, int nlarg) {
  __shared__ Ev E[GR_EV_MAX];
  __shared__ unsigned long long M[GR_MOTS_MAX];
  __shared__ double NX[16], NZ[16], NX2[16], NZ2[16];
  __shared__ double NY[GR_NY_MAX], NY2[GR_NY_MAX];
  __shared__ int NE, BORNES[6];
  __shared__ unsigned ERR, TOUCHE;
  __shared__ unsigned long long COMPTE;
  const GrItem &it = *reinterpret_cast<const GrItem *>(IN + 16 * (size_t)blockIdx.x);
  int t = threadIdx.x;
  if (t == 0) {
    int n = 0;
    unsigned err = 0;
    if (nw > GR_MOTS_MAX) err |= GR_E_MASQUE;
    else if (it.genre == GR_SALLE) gr_garder(E, &n, &err, cx, cz, it.x, it.y, it.z, it.horizontal, it.vertical);
    else if (it.genre == GR_TUNNEL) gr_tunnels(it, T, cx, cz, E, &n, &err);
    else if (it.genre == GR_CANYON) gr_canyon(it, T, cx, cz, E, &n, &err);
    else err |= GR_E_FOURNISSEUR;
    NE = err ? 0 : n;
    ERR = err;
    TOUCHE = 0;
    COMPTE = 0;
  }
  __syncthreads();
  int ne = NE;
  if (ne > 0)
    for (int i = t; i < nw; i += blockDim.x) M[i] = 0;
  bool canyon = it.genre == GR_CANYON;
  double plancher = it.plancher;
  unsigned long long compte = 0;
  unsigned err = 0, touche = 0;
  int x0 = cx * 16, z0 = cz * 16;
  for (int k = 0; k < ne; ++k) {
    Ev e = E[k];
    __syncthreads();
    if (t == 0) {
      int mx = __double2int_rd(__dsub_rn(e.x, e.h)) - x0 - 1;
      int Mx = __double2int_rd(__dadd_rn(e.x, e.h)) - x0;
      int my = __double2int_rd(__dsub_rn(e.y, e.v)) - 1;
      int My = __double2int_rd(__dadd_rn(e.y, e.v)) + 1;
      int mz = __double2int_rd(__dsub_rn(e.z, e.h)) - z0 - 1;
      int Mz = __double2int_rd(__dadd_rn(e.z, e.h)) - z0;
      BORNES[0] = mx < 0 ? 0 : mx;
      BORNES[1] = Mx > 15 ? 15 : Mx;
      BORNES[2] = my < ymin ? ymin : my;
      BORNES[3] = My > ymax ? ymax : My;
      BORNES[4] = mz < 0 ? 0 : mz;
      BORNES[5] = Mz > 15 ? 15 : Mz;
    }
    __syncthreads();
    int bx = BORNES[0], Bx = BORNES[1], by = BORNES[2], By = BORNES[3], bz = BORNES[4], Bz = BORNES[5];
    int nx = Bx - bx + 1, nz = Bz - bz + 1, ny = By - by;
    if (nx <= 0 || nz <= 0 || ny <= 0) continue;
    if (ny > GR_NY_MAX) {
      err |= GR_E_HAUTEUR;
      continue;
    }
    if (t < nx) {
      double v = __ddiv_rn(__dsub_rn(__dadd_rn((double)(x0 + bx + t), 0.5), e.x), e.h);
      NX[t] = v;
      NX2[t] = __dmul_rn(v, v);
    }
    if (t >= 32 && t < 32 + nz) {
      double v = __ddiv_rn(__dsub_rn(__dadd_rn((double)(z0 + bz + t - 32), 0.5), e.z), e.h);
      NZ[t - 32] = v;
      NZ2[t - 32] = __dmul_rn(v, v);
    }
    for (int j = t; j < ny; j += blockDim.x) {
      double v = __ddiv_rn(__dsub_rn(__dsub_rn((double)(By - j), 0.5), e.y), e.v);
      NY[j] = v;
      NY2[j] = __dmul_rn(v, v);
    }
    __syncthreads();
    int total = nx * nz * ny;
    for (int c = t; c < total; c += blockDim.x) {
      int j = c % ny, col = c / ny;
      int ix = col / nz, iz = col % nz;
      if (__dadd_rn(NX2[ix], NZ2[iz]) >= 1.0) continue;
      ++compte;
      int y = By - j;
      bool saute;
      if (canyon) {
        int li = y - gen_ymin - 1;
        if (li < 0 || li >= nlarg) {
          err |= GR_E_LARGEUR;
          continue;
        }
        double f = (double)W[it.largeur + li];
        saute = __dadd_rn(__dmul_rn(__dadd_rn(NX2[ix], NZ2[iz]), f), __ddiv_rn(NY2[j], 6.0)) >= 1.0;
      } else {
        saute = NY[j] <= plancher || __dadd_rn(__dadd_rn(NX2[ix], NY2[j]), NZ2[iz]) >= 1.0;
      }
      if (!saute) {
        int i = (y - ymin) + ((bz + iz) + (bx + ix) * 16) * hauteur;
        atomicOr(&M[i >> 6], 1ULL << (i & 63));
        touche = 1;
      }
    }
  }
  if (compte) atomicAdd(&COMPTE, compte);
  if (err) atomicOr(&ERR, err);
  if (touche) atomicOr(&TOUCHE, 1u);
  __syncthreads();
  if (TOUCHE)
    for (int i = t; i < nw; i += blockDim.x)
      if (M[i]) atomicOr(&OUT[i], M[i]);
  if (t == 0) {
    if (COMPTE && it.appel < (unsigned)nc) atomicAdd(&OUT[nw + it.appel], COMPTE);
    if (ERR) atomicOr(&OUT[nw + nc], (unsigned long long)ERR);
  }
}
