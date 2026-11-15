// Horizon's Xoroshiro128++ generator, operation by operation.
// Xoroshiro128++ and XoroshiroPositionalRandom::at
__device__ __forceinline__ unsigned long long aq_rotl(unsigned long long x, int k) { return (x << k) | (x >> (64 - k)); }
__device__ __forceinline__ unsigned long long aq_next_long(unsigned long long &lo, unsigned long long &hi) {
  unsigned long long l = lo, h = hi ^ l;
  unsigned long long r = aq_rotl(l + hi, 17) + l;
  lo = aq_rotl(l, 49) ^ h ^ (h << 21);
  hi = aq_rotl(h, 28);
  return r;
}
__device__ __forceinline__ int aq_next_int_bound(unsigned long long &lo, unsigned long long &hi, int bound) {
  unsigned int b = (unsigned int)bound;
  unsigned int threshold = (0u - b) % b;
  while (true) {
    unsigned long long p = (unsigned long long)(unsigned int)aq_next_long(lo, hi) * (unsigned long long)b;
    if ((unsigned int)p >= threshold) return (int)(p >> 32);
  }
}
// Mth.getSeed (Horizon's position_seed), signed Java arithmetic
__device__ __forceinline__ long long aq_position_seed(int x, int y, int z) {
  long long m = (long long)(int)((unsigned)x * 3129871u) ^ (long long)((unsigned long long)(long long)z * 116129781ull) ^ (long long)y;
  unsigned long long u = (unsigned long long)m;
  unsigned long long r = u * u * 42317861ull + u * 11ull;
  return ((long long)r) >> 16;
}
// Xoroshiro draws (Horizon's RandomSource) and positional random factory: state of at(x, y, z)
__device__ __forceinline__ float aq_next_float(unsigned long long &lo, unsigned long long &hi) {
  return (float)(aq_next_long(lo, hi) >> 40) * 5.9604644775390625e-8f;
}
__device__ __forceinline__ double aq_next_double(unsigned long long &lo, unsigned long long &hi) {
  return (double)(aq_next_long(lo, hi) >> 11) * 1.1102230246251565e-16;
}
__device__ __forceinline__ void aq_at(unsigned long long LO, unsigned long long HI, int x, int y, int z, unsigned long long &lo, unsigned long long &hi) {
  lo = (unsigned long long)aq_position_seed(x, y, z) ^ LO;
  hi = HI;
  if ((lo | hi) == 0ull) { lo = 0x9e3779b97f4a7c15ull; hi = 0x6a09e667f3bcc909ull; }
}
