//! Packed column heights and the six Java 26.3 heightmap predicates.
//!
//! Block behavior supplies air/fluid facts and resolved block-tag membership.
//! Heightmaps do not infer those facts from collision shapes or block names.
use horizon_core::storage::{PackedStorage, StorageError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HeightmapType {
    WorldSurfaceWg,
    WorldSurface,
    OceanFloorWg,
    OceanFloor,
    MotionBlocking,
    MotionBlockingNoLeaves,
}

impl HeightmapType {
    pub const ALL: [Self; 6] = [
        Self::WorldSurfaceWg,
        Self::WorldSurface,
        Self::OceanFloorWg,
        Self::OceanFloor,
        Self::MotionBlocking,
        Self::MotionBlockingNoLeaves,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::WorldSurfaceWg => "WORLD_SURFACE_WG",
            Self::WorldSurface => "WORLD_SURFACE",
            Self::OceanFloorWg => "OCEAN_FLOOR_WG",
            Self::OceanFloor => "OCEAN_FLOOR",
            Self::MotionBlocking => "MOTION_BLOCKING",
            Self::MotionBlockingNoLeaves => "MOTION_BLOCKING_NO_LEAVES",
        }
    }

    pub fn send_to_client(self) -> bool {
        matches!(
            self,
            Self::WorldSurface | Self::MotionBlocking | Self::MotionBlockingNoLeaves
        )
    }

    pub fn keep_after_worldgen(self) -> bool {
        !matches!(self, Self::WorldSurfaceWg | Self::OceanFloorWg)
    }

    pub fn is_opaque(self, state: impl HeightmapState) -> bool {
        match self {
            Self::WorldSurfaceWg | Self::WorldSurface => !state.is_air(),
            Self::OceanFloorWg | Self::OceanFloor => state.blocks_motion_in_heightmap(),
            Self::MotionBlocking => state.blocks_motion_in_heightmap() || state.has_fluid(),
            Self::MotionBlockingNoLeaves => {
                state.blocks_motion_in_heightmap_no_leaves() || state.has_fluid()
            }
        }
    }
}

pub trait HeightmapState: Copy {
    fn is_air(self) -> bool;
    /// Exact AIR block identity, distinct from cave/void air.
    fn is_air_block(self) -> bool;
    fn has_fluid(self) -> bool;
    fn blocks_motion_in_heightmap(self) -> bool;
    fn blocks_motion_in_heightmap_no_leaves(self) -> bool;
}

pub trait HeightmapChunk {
    type State: HeightmapState;
    fn min_y(&self) -> i32;
    fn height(&self) -> i32;
    fn highest_section_y(&self) -> i32;
    fn block_state(&self, x: i32, y: i32, z: i32) -> Self::State;
}

/// What [`Heightmap::prime_sections`] reads of a column: a section at a time, each cell as the maps
/// it tops.
pub trait HeightmapSections: HeightmapChunk {
    /// The lowest Y of the sections, one every 16 blocks up from it.
    fn sections_min_y(&self) -> i32;
    /// The cells of the section whose lowest Y is `bottom`, in `(y << 8) | (z << 4) | x` order: bit `i`
    /// where the state `block_state` reads is not exactly air and `kinds[i].is_opaque` holds for it,
    /// as [`Heightmap::prime`] tests them. False (and `out` unspecified) when no cell has a bit, or
    /// there is no section there.
    fn section_masks(&self, bottom: i32, kinds: &[HeightmapType], out: &mut [u8; 4096]) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeightmapError {
    InvalidBounds,
    MismatchedChunk,
    Storage(StorageError),
}
impl From<StorageError> for HeightmapError {
    fn from(value: StorageError) -> Self {
        Self::Storage(value)
    }
}
impl std::fmt::Display for HeightmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for HeightmapError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Heightmap {
    kind: HeightmapType,
    min_y: i32,
    height: i32,
    data: PackedStorage,
}

impl Heightmap {
    /// The heap bytes of its packed heights.
    pub fn heap_bytes(&self) -> usize {
        self.data.heap_bytes()
    }
    /// Bounded to the vanilla codec's positive, aligned vertical ranges.
    pub fn new(kind: HeightmapType, min_y: i32, height: i32) -> Result<Self, HeightmapError> {
        if !(-2032..=2031).contains(&min_y)
            || !(16..=4064).contains(&height)
            || min_y % 16 != 0
            || height % 16 != 0
            || min_y + height > 2032
        {
            return Err(HeightmapError::InvalidBounds);
        }
        let bits = (32 - (height as u32).leading_zeros()) as u8;
        Ok(Self {
            kind,
            min_y,
            height,
            data: PackedStorage::new(bits, 256)?,
        })
    }

    pub fn kind(&self) -> HeightmapType {
        self.kind
    }
    pub fn raw_data(&self) -> &[u64] {
        self.data.words()
    }

    fn index(x: i32, z: i32) -> usize {
        x.wrapping_add(z.wrapping_mul(16)) as usize
    }

    /// Whether `update` at (`x`, `y`, `z`) returns at once, whatever the
    /// state: the block is two or more below the first available one. `x`
    /// and `z` are within 0..16. Most generation writes are (the surface
    /// rule turns every stone under y 8 to deepslate), and this skips the
    /// checks `update` does first.
    #[inline]
    pub fn below_update(&self, chunk: &impl HeightmapChunk, x: i32, y: i32, z: i32) -> bool {
        debug_assert!((0..16).contains(&x) && (0..16).contains(&z));
        let first = (self.data.read(Self::index(x, z)) as i32).wrapping_add(self.min_y);
        y <= first.wrapping_sub(2) && self.check_chunk(chunk).is_ok()
    }

    pub fn first_available(&self, x: i32, z: i32) -> Result<i32, HeightmapError> {
        Ok((self.data.get(Self::index(x, z))? as i32).wrapping_add(self.min_y))
    }

    /// Whether any column of the local square `x0..=x1` by `z0..=z1` (each
    /// within 0..16) has its first available y at or above `y`.
    pub fn any_first_available_at_least(
        &self,
        [x0, x1]: [i32; 2],
        [z0, z1]: [i32; 2],
        y: i32,
    ) -> bool {
        debug_assert!(0 <= x0 && x1 < 16 && 0 <= z0 && z1 < 16);
        (z0..=z1).any(|z| {
            (x0..=x1)
                .any(|x| (self.data.read(Self::index(x, z)) as i32).wrapping_add(self.min_y) >= y)
        })
    }

    /// Set a column's first available y (`x` and `z` within 0..16), as `update` would have.
    pub fn set_first_available(&mut self, x: i32, z: i32, y: i32) -> Result<(), HeightmapError> {
        self.set_height(x, z, y)
    }
    pub fn highest_taken(&self, x: i32, z: i32) -> Result<i32, HeightmapError> {
        Ok(self.first_available(x, z)?.wrapping_sub(1))
    }

    fn set_height(&mut self, x: i32, z: i32, y: i32) -> Result<(), HeightmapError> {
        self.data
            .set(Self::index(x, z), y.wrapping_sub(self.min_y) as u32)?;
        Ok(())
    }

    fn check_chunk(&self, chunk: &impl HeightmapChunk) -> Result<(), HeightmapError> {
        if chunk.min_y() != self.min_y || chunk.height() != self.height {
            return Err(HeightmapError::MismatchedChunk);
        }
        Ok(())
    }

    /// The caller writes the new state to the chunk before updating its maps.
    pub fn update<C: HeightmapChunk>(
        &mut self,
        chunk: &C,
        x: i32,
        y: i32,
        z: i32,
        state: C::State,
    ) -> Result<bool, HeightmapError> {
        self.check_chunk(chunk)?;
        let first = self.first_available(x, z)?;
        if y <= first.wrapping_sub(2) {
            return Ok(false);
        }
        if self.kind.is_opaque(state) {
            if y >= first {
                self.set_height(x, z, y.wrapping_add(1))?;
                return Ok(true);
            }
        } else if y == first.wrapping_sub(1) {
            let mut scan_y = y.wrapping_sub(1);
            while scan_y >= self.min_y {
                if self.kind.is_opaque(chunk.block_state(x, scan_y, z)) {
                    self.set_height(x, z, scan_y.wrapping_add(1))?;
                    return Ok(true);
                }
                scan_y -= 1;
            }
            self.set_height(x, z, self.min_y)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Matching lengths preserve all bits, including padding and unused slots.
    /// A length mismatch primes this registered map from the chunk instead.
    pub fn set_raw_data(
        &mut self,
        chunk: &impl HeightmapChunk,
        words: &[u64],
    ) -> Result<(), HeightmapError> {
        self.check_chunk(chunk)?;
        if words.len() == self.data.words().len() {
            self.data = PackedStorage::from_words(self.data.bits(), 256, words.to_vec())?;
            Ok(())
        } else {
            Self::prime(chunk, &mut [self])
        }
    }

    /// Preserves caller map order and scans X, then Z, then descending Y.
    /// An unfound column retains its previous value, including during re-priming.
    /// [`prime`](Self::prime) a section at a time: the same heights (each map at the highest cell its
    /// kind tops, under the highest section that is not all air), from the masks of the sections
    /// instead of a state per cell.
    pub fn prime_sections(
        chunk: &impl HeightmapSections,
        maps: &mut [&mut Self],
    ) -> Result<(), HeightmapError> {
        if maps.is_empty() {
            return Ok(());
        }
        if maps.len() > 8 {
            return Self::prime(chunk, maps);
        }
        for map in maps.iter() {
            map.check_chunk(chunk)?;
        }
        let kinds: Vec<HeightmapType> = maps.iter().map(|m| m.kind).collect();
        let full = ((1u16 << maps.len()) - 1) as u8;
        let mut remaining = [full; 256];
        let mut open = 256usize;
        let top = i64::from(chunk.highest_section_y().wrapping_add(16).wrapping_sub(1));
        let bottom = i64::from(chunk.min_y());
        let base = i64::from(chunk.sections_min_y());
        let mut masks = [0u8; 4096];
        let mut section = base + (top - base).div_euclid(16) * 16;
        while open > 0 && section + 15 >= bottom && top >= bottom {
            if let Ok(start) = i32::try_from(section)
                && chunk.section_masks(start, &kinds, &mut masks)
            {
                let (hi, lo) = ((section + 15).min(top), section.max(bottom));
                for z in 0..16 {
                    for x in 0..16 {
                        let column = (z * 16 + x) as usize;
                        if remaining[column] == 0 {
                            continue;
                        }
                        let mut y = hi;
                        while y >= lo {
                            let cell = (((y - section) as usize) << 8) | ((z as usize) << 4) | x as usize;
                            let m = masks[cell] & remaining[column];
                            if m != 0 {
                                for (i, map) in maps.iter_mut().enumerate() {
                                    if m >> i & 1 != 0 {
                                        map.set_height(x, z, (y as i32).wrapping_add(1))?;
                                    }
                                }
                                remaining[column] &= !m;
                                if remaining[column] == 0 {
                                    open -= 1;
                                    break;
                                }
                            }
                            y -= 1;
                        }
                    }
                }
            }
            section -= 16;
        }
        Ok(())
    }
    pub fn prime(
        chunk: &impl HeightmapChunk,
        maps: &mut [&mut Self],
    ) -> Result<(), HeightmapError> {
        if maps.is_empty() {
            return Ok(());
        }
        for map in maps.iter() {
            map.check_chunk(chunk)?;
        }
        let upper = chunk.highest_section_y().wrapping_add(16);
        let mut remaining = Vec::with_capacity(maps.len());
        for x in 0..16 {
            for z in 0..16 {
                remaining.clear();
                remaining.extend(0..maps.len());
                let mut y = upper.wrapping_sub(1);
                while y >= chunk.min_y() {
                    let state = chunk.block_state(x, y, z);
                    if !state.is_air_block() {
                        let mut i = 0;
                        while i < remaining.len() {
                            let map = &mut maps[remaining[i]];
                            if map.kind.is_opaque(state) {
                                map.set_height(x, z, y.wrapping_add(1))?;
                                remaining.remove(i);
                            } else {
                                i += 1;
                            }
                        }
                        if remaining.is_empty() {
                            break;
                        }
                    }
                    y -= 1;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod prime_sections_tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct S(u8);
    impl HeightmapState for S {
        fn is_air(self) -> bool {
            self.0 < 3
        }
        fn is_air_block(self) -> bool {
            self.0 == 0
        }
        fn has_fluid(self) -> bool {
            self.0 % 7 == 3
        }
        fn blocks_motion_in_heightmap(self) -> bool {
            self.0 % 3 == 1
        }
        fn blocks_motion_in_heightmap_no_leaves(self) -> bool {
            self.0 % 5 == 1
        }
    }
    struct Column {
        cells: Vec<u8>,
    }
    impl HeightmapChunk for Column {
        type State = S;
        fn min_y(&self) -> i32 {
            -64
        }
        fn height(&self) -> i32 {
            384
        }
        fn highest_section_y(&self) -> i32 {
            let top = (0..24).rev().find(|&s| self.cells[s * 4096..(s + 1) * 4096].iter().any(|&c| c != 0));
            -64 + top.unwrap_or(0) as i32 * 16
        }
        fn block_state(&self, x: i32, y: i32, z: i32) -> S {
            S(self.cells[((y + 64) as usize) * 256 + (z as usize) * 16 + x as usize])
        }
    }
    impl HeightmapSections for Column {
        fn sections_min_y(&self) -> i32 {
            -64
        }
        fn section_masks(&self, bottom: i32, kinds: &[HeightmapType], out: &mut [u8; 4096]) -> bool {
            let mut any = false;
            for (i, cell) in out.iter_mut().enumerate() {
                let s = self.block_state((i & 15) as i32, bottom + (i >> 8) as i32, ((i >> 4) & 15) as i32);
                *cell = kinds.iter().enumerate().fold(0, |m, (k, kind)| {
                    m | u8::from(!s.is_air_block() && kind.is_opaque(s)) << k
                });
                any |= *cell != 0;
            }
            any
        }
    }

    /// Section by section, the heights are those of the cell-by-cell prime.
    #[test]
    fn prime_sections_matches_prime() {
        let kinds = [
            HeightmapType::WorldSurface,
            HeightmapType::OceanFloor,
            HeightmapType::MotionBlocking,
            HeightmapType::MotionBlockingNoLeaves,
        ];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for round in 0..40 {
            // Ground up to a random height, then sparse blocks, air above
            let ground = next(384) as usize;
            let cells: Vec<u8> = (0..384 * 256)
                .map(|i| {
                    let y = i / 256;
                    if y < ground {
                        (next(30) + 1) as u8
                    } else if y < ground + 40 && next(9) == 0 {
                        next(30) as u8
                    } else {
                        0
                    }
                })
                .collect();
            let column = Column { cells };
            let mut a: Vec<Heightmap> = kinds.iter().map(|k| Heightmap::new(*k, -64, 384).unwrap()).collect();
            let mut b: Vec<Heightmap> = kinds.iter().map(|k| Heightmap::new(*k, -64, 384).unwrap()).collect();
            Heightmap::prime(&column, &mut a.iter_mut().collect::<Vec<_>>()).unwrap();
            Heightmap::prime_sections(&column, &mut b.iter_mut().collect::<Vec<_>>()).unwrap();
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.raw_data(), y.raw_data(), "round {round}, {:?}", x.kind);
            }
        }
    }
}
