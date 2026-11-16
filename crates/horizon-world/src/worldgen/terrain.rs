//! Noise-stage chunk filling. Material rules, structures and chunk lifecycle are
//! separate stages; this module does not produce a complete generated chunk.
use super::aquifer::{
    AquiferBlockStates, AquiferError, DisabledAquifer, FluidPicker, NoiseBasedAquifer,
};
use crate::{
    chunk::BlockSection,
    heightmap::{Heightmap, HeightmapChunk, HeightmapError, HeightmapState, HeightmapType},
    registry::{BlockRegistry, BlockStateId},
};
use horizon_core::{
    density::{DensitySampler, SampleError, SamplerContext},
    noise::DensityVolume,
    position::ChunkPos,
    random::PositionalRandom,
    ticks::ProtoChunkTicks,
};
use std::{cell::RefCell, io};

thread_local! {
    static SCRATCH_DENSITY: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Behavior and resolved tags for every state in the chunk's immutable registry.
pub trait TerrainBlockStates {
    type State: HeightmapState;
    fn heightmap_state(&self, id: BlockStateId) -> Self::State;
}

pub trait TerrainAquifer {
    fn compute_substance(
        &mut self,
        context: &mut SamplerContext,
        position: [i32; 3],
        density: f64,
    ) -> Result<Option<BlockStateId>, AquiferError>;
    fn should_schedule_fluid_update(&self) -> bool;
    /// The y above which every non-positive density is the aquifer's air. The
    /// fill skips those blocks without asking the aquifer; `None` keeps the
    /// per-block call for aquifers whose global fluid can vary with x/z.
    fn air_above_y(&self) -> Option<i32> {
        None
    }
    /// The fill decided this block without the aquifer (positive density or air
    /// above its skip height). Both old paths cleared the pending fluid update.
    fn note_skipped(&mut self) {}
}
impl<P: FluidPicker> TerrainAquifer for DisabledAquifer<P> {
    fn compute_substance(
        &mut self,
        _: &mut SamplerContext,
        position: [i32; 3],
        density: f64,
    ) -> Result<Option<BlockStateId>, AquiferError> {
        Ok(DisabledAquifer::compute_substance(self, position, density))
    }
    fn should_schedule_fluid_update(&self) -> bool {
        false
    }
}
impl<P: FluidPicker, F: PositionalRandom, B: AquiferBlockStates> TerrainAquifer
    for NoiseBasedAquifer<P, F, B>
{
    fn compute_substance(
        &mut self,
        context: &mut SamplerContext,
        position: [i32; 3],
        density: f64,
    ) -> Result<Option<BlockStateId>, AquiferError> {
        NoiseBasedAquifer::compute_substance(self, context, position, density)
    }
    fn should_schedule_fluid_update(&self) -> bool {
        NoiseBasedAquifer::should_schedule_fluid_update(self)
    }
    fn air_above_y(&self) -> Option<i32> {
        NoiseBasedAquifer::air_above_y(self)
    }
    fn note_skipped(&mut self) {
        NoiseBasedAquifer::note_skipped(self);
    }
}

#[derive(Debug)]
pub enum TerrainError {
    Storage(io::Error),
    Heightmap(HeightmapError),
    Sample(SampleError),
    Aquifer(AquiferError),
    MissingAir,
    InvalidState,
    InvalidVolume,
    OutsideBuildHeight,
}
impl std::fmt::Display for TerrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for TerrainError {}
impl From<io::Error> for TerrainError {
    fn from(value: io::Error) -> Self {
        Self::Storage(value)
    }
}
impl From<HeightmapError> for TerrainError {
    fn from(value: HeightmapError) -> Self {
        Self::Heightmap(value)
    }
}
impl From<SampleError> for TerrainError {
    fn from(value: SampleError) -> Self {
        Self::Sample(value)
    }
}
impl From<AquiferError> for TerrainError {
    fn from(value: AquiferError) -> Self {
        Self::Aquifer(value)
    }
}

/// Resident registry-backed sections, optional heightmaps and ordered per-section
/// fluid postprocessing offsets for the noise and surface generation stages.
pub struct TerrainChunk<'r, B: TerrainBlockStates> {
    stage: PreLightChunkStage,
    below_zero_retrogen: Option<crate::retrogen::BelowZeroRetrogen>,
    position: ChunkPos,
    blocks: TerrainBlocks<'r, B>,
    heightmaps: [Option<Heightmap>; 6],
    postprocessing: Vec<Vec<u16>>,
    block_ticks: ProtoChunkTicks,
    fluid_ticks: ProtoChunkTicks,
}
/// The persisted generation statuses before lighting initialization. This models
/// status-dependent storage and upgrade cleanup, not scheduling, lighting or
/// save-dirty notifications.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreLightChunkStage {
    #[default]
    Empty,
    StructureStarts,
    StructureReferences,
    Biomes,
    Terrain,
    Features,
}
impl PreLightChunkStage {
    pub const ALL: [Self; 6] = [
        Self::Empty,
        Self::StructureStarts,
        Self::StructureReferences,
        Self::Biomes,
        Self::Terrain,
        Self::Features,
    ];
    pub fn status(self) -> crate::chunk_status::ChunkStatus {
        crate::chunk_status::ChunkStatus::ALL[self as usize]
    }
    pub fn name(self) -> &'static str {
        self.status().name().strip_prefix("minecraft:").unwrap()
    }
    pub fn heightmaps(self) -> &'static [HeightmapType] {
        self.status().heightmaps()
    }
}
struct TerrainBlocks<'r, B: TerrainBlockStates> {
    min_y: i32,
    height: i32,
    sections: Vec<BlockSection<'r>>,
    non_air: Vec<u16>,
    registry: &'r BlockRegistry,
    properties: B,
    air: BlockStateId,
    void_air: BlockStateId,
}
impl<B: TerrainBlockStates> HeightmapChunk for TerrainBlocks<'_, B> {
    type State = B::State;
    fn min_y(&self) -> i32 {
        self.min_y
    }
    fn height(&self) -> i32 {
        self.height
    }
    fn highest_section_y(&self) -> i32 {
        self.non_air
            .iter()
            .rposition(|&count| count != 0)
            .map_or(self.min_y, |i| self.min_y + i as i32 * 16)
    }
    fn block_state(&self, x: i32, y: i32, z: i32) -> B::State {
        self.properties.heightmap_state(self.block(x, y, z))
    }
}
impl<B: TerrainBlockStates> TerrainBlocks<'_, B> {
    /// Section `index`'s count of non-air blocks after a write of `new` over `old`.
    #[inline]
    fn count_write(&mut self, index: usize, old: BlockStateId, new: BlockStateId) {
        let old_air = self.properties.heightmap_state(old).is_air();
        let new_air = self.properties.heightmap_state(new).is_air();
        if old_air && !new_air {
            self.non_air[index] += 1;
        } else if !old_air && new_air {
            self.non_air[index] -= 1;
        }
    }
    #[inline(always)]
    fn section_index(&self, y: i32) -> Option<usize> {
        let dy = i64::from(y) - i64::from(self.min_y);
        (dy >= 0 && dy < i64::from(self.height)).then_some((dy >> 4) as usize)
    }
    #[inline(always)]
    fn block(&self, x: i32, y: i32, z: i32) -> BlockStateId {
        let Some(section) = self.section_index(y) else {
            return self.void_air;
        };
        if self.non_air[section] == 0 {
            return self.air;
        }
        self.sections[section].state(local_index(x, y, z))
    }
    /// Every state of section `index` in `local_index` order, as `block`
    /// reads them one by one.
    fn section_states(&self, index: usize, out: &mut [u32; 4096]) {
        if self.non_air[index] == 0 {
            out.fill(self.air.0);
        } else {
            self.sections[index].fill_states(out);
        }
    }
}
#[inline(always)]
fn local_index(x: i32, y: i32, z: i32) -> usize {
    (((y & 15) << 8) | ((z & 15) << 4) | (x & 15)) as usize
}

/// The answers of a [`TerrainChunk::replace_row_states`] decision for the states met in one row:
/// the state written (if any) and the change it makes to the section's non-air count, as
/// `count_write` counts it.
struct StateMemo {
    entries: [(BlockStateId, Option<BlockStateId>, i16); 6],
    len: usize,
    next: usize,
}
impl Default for StateMemo {
    fn default() -> Self {
        Self {
            entries: [(BlockStateId(0), None, 0); 6],
            len: 0,
            next: 0,
        }
    }
}
impl StateMemo {
    #[inline]
    fn get<B: TerrainBlockStates>(
        &mut self,
        old: BlockStateId,
        f: &mut dyn FnMut(BlockStateId) -> Option<BlockStateId>,
        properties: &B,
    ) -> Option<(BlockStateId, i16)> {
        if let Some(e) = self.entries[..self.len].iter().find(|e| e.0 == old) {
            return e.1.map(|state| (state, e.2));
        }
        let new = f(old);
        let delta = new.map_or(0, |state| {
            let old_air = properties.heightmap_state(old).is_air();
            let new_air = properties.heightmap_state(state).is_air();
            i16::from(old_air && !new_air) - i16::from(!old_air && new_air)
        });
        self.entries[self.next] = (old, new, delta);
        self.next = (self.next + 1) % self.entries.len();
        self.len = (self.len + 1).min(self.entries.len());
        new.map(|state| (state, delta))
    }
}

/// `TerrainChunk::worldgen_column_bounds`, indexed by `HeightmapType`.
pub(crate) type ColumnBounds = [i32; HeightmapType::ALL.len()];

impl<'r, B: TerrainBlockStates> TerrainChunk<'r, B> {
    /// Its heap bytes by part (`horizon-ctl memory`; scheduled ticks not
    /// counted): packed block storage, the sections' palettes and lists,
    /// heightmaps, and the rest (post-processing marks, non-air counts).
    pub fn heap_bytes(&self) -> [usize; 4] {
        let blocks = &self.blocks;
        let [mut storage, mut lists] = [
            0,
            blocks.sections.capacity() * std::mem::size_of::<BlockSection<'_>>(),
        ];
        for section in &blocks.sections {
            let [s, l] = section.heap_bytes();
            storage += s;
            lists += l;
        }
        [
            storage,
            lists,
            self.heightmaps
                .iter()
                .flatten()
                .map(Heightmap::heap_bytes)
                .sum(),
            blocks.non_air.capacity() * std::mem::size_of::<u16>()
                + self.postprocessing.capacity() * std::mem::size_of::<Vec<u16>>()
                + self
                    .postprocessing
                    .iter()
                    .map(|p| p.capacity() * std::mem::size_of::<u16>())
                    .sum::<usize>(),
        ]
    }
    /// How many of its sections pack their blocks at each bit width (index
    /// = bits, 0 = a single state).
    pub fn storage_bits(&self) -> [usize; 16] {
        let mut bits = [0; 16];
        for section in &self.blocks.sections {
            bits[usize::from(section.storage_bits()).min(15)] += 1;
        }
        bits
    }
    pub fn new(
        position: ChunkPos,
        min_y: i32,
        height: i32,
        registry: &'r BlockRegistry,
        properties: B,
    ) -> Result<Self, TerrainError> {
        // Share the heightmap's supported range and fail before allocating sections.
        Heightmap::new(HeightmapType::WorldSurfaceWg, min_y, height)?;
        let air = registry
            .default_state("minecraft:air")
            .ok_or(TerrainError::MissingAir)?;
        let void_air = registry
            .default_state("minecraft:void_air")
            .ok_or(TerrainError::MissingAir)?;
        let section_count = (height / 16) as usize;
        let sections = (0..section_count)
            .map(|_| BlockSection::uniform(air, registry))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self {
            stage: PreLightChunkStage::Empty,
            below_zero_retrogen: None,
            position,
            blocks: TerrainBlocks {
                min_y,
                height,
                sections,
                non_air: vec![0; section_count],
                registry,
                properties,
                air,
                void_air,
            },
            heightmaps: std::array::from_fn(|_| None),
            postprocessing: vec![Vec::new(); section_count],
            block_ticks: ProtoChunkTicks::default(),
            fluid_ticks: ProtoChunkTicks::default(),
        })
    }
    pub(crate) fn registry(&self) -> &BlockRegistry {
        self.blocks.registry
    }
    /// The indexes of the sections holding a block other than air
    /// (`LevelChunkSection.hasOnlyAir`, negated).
    pub fn filled_sections(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.blocks.non_air.len()).filter(|&i| self.blocks.non_air[i] != 0)
    }
    pub fn highest_filled_section_index(&self) -> Option<usize> {
        self.blocks.non_air.iter().rposition(|&count| count != 0)
    }
    pub fn block_properties(&self, state: BlockStateId) -> B::State {
        self.blocks.properties.heightmap_state(state)
    }
    pub fn same_block(&self, left: BlockStateId, right: BlockStateId) -> bool {
        self.blocks
            .registry
            .state(left)
            .zip(self.blocks.registry.state(right))
            .is_some_and(|(left, right)| left.name == right.name)
    }
    pub fn generation_stage(&self) -> PreLightChunkStage {
        self.stage
    }
    /// Retained old terrain may already have completed stages that the new
    /// below-zero generation has not reached. Biome reads use this status.
    pub fn highest_generated_status(&self) -> crate::chunk_status::ChunkStatus {
        self.below_zero_retrogen
            .as_ref()
            .map_or(self.stage.status(), |data| {
                self.stage.status().max(data.target_status())
            })
    }
    /// Changing status alone does not prime or clear any heightmaps. Native
    /// persisted-status assignment also permits earlier statuses. Reaching an
    /// installed upgrade target clears its metadata; save-dirty tracking is separate.
    pub fn set_generation_stage(&mut self, stage: PreLightChunkStage) {
        self.stage = stage;
        if self
            .below_zero_retrogen
            .as_ref()
            .is_some_and(|data| stage.status() >= data.target_status())
        {
            self.below_zero_retrogen = None;
        }
    }
    pub fn below_zero_retrogen(&self) -> Option<&crate::retrogen::BelowZeroRetrogen> {
        self.below_zero_retrogen.as_ref()
    }
    /// Installation alone does not compare the target with the current status.
    /// A subsequent persisted-stage assignment performs completion cleanup.
    pub fn set_below_zero_retrogen(&mut self, data: Option<crate::retrogen::BelowZeroRetrogen>) {
        self.below_zero_retrogen = data;
    }
    pub fn is_upgrading(&self) -> bool {
        self.below_zero_retrogen.is_some()
    }
    /// Generation minimum Y and height. Storage bounds remain unchanged.
    pub fn height_for_generation(&self) -> (i32, i32) {
        if self.is_upgrading() {
            (-64, 64)
        } else {
            (self.min_y(), self.height())
        }
    }
    /// ProtoChunk writes before lighting. The current stage selects heightmaps;
    /// missing maps are primed after existing maps are updated.
    pub fn set_block_worldgen(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        state: BlockStateId,
    ) -> Result<BlockStateId, TerrainError> {
        let Some(section) = self.blocks.section_index(y) else {
            return Ok(self.blocks.void_air);
        };
        if self.blocks.non_air[section] == 0 && state == self.blocks.air {
            return Ok(state);
        }
        let previous = self.set_block_raw(x, y, z, state)?;
        let mut missing = Vec::new();
        for &kind in self.stage.heightmaps() {
            match self.heightmap(kind) {
                // `Heightmap::update` would return at once: no state is read.
                Some(map) if map.below_update(&self.blocks, x & 15, y, z & 15) => {
                    if self.blocks.registry.state(state).is_none() {
                        return Err(TerrainError::InvalidState);
                    }
                }
                Some(_) => {
                    self.update_heightmap(kind, x & 15, y, z & 15, state)?;
                }
                None => missing.push(kind),
            }
        }
        if !missing.is_empty() {
            self.prime_heightmaps(&missing)?;
        }
        Ok(previous)
    }
    /// Per stage heightmap (indexed by `HeightmapType`), the highest y at
    /// which a write cannot change it in column `x`, `z`: its first available
    /// y - 2 (`Heightmap::update`). `None` when a stage map is missing.
    pub(crate) fn worldgen_column_bounds(&self, x: i32, z: i32) -> Option<ColumnBounds> {
        let mut bounds = [i32::MAX; HeightmapType::ALL.len()];
        for &kind in self.stage.heightmaps() {
            bounds[kind as usize] = self
                .heightmap(kind)?
                .first_available(x & 15, z & 15)
                .ok()?
                .wrapping_sub(2);
        }
        Some(bounds)
    }
    /// `set_block_worldgen` for writes in one column, with that column's
    /// `worldgen_column_bounds`: updates at or below a map's bound, which
    /// change nothing, are not made. Refreshes `bounds` when a map changes.
    pub(crate) fn set_block_worldgen_column(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        state: BlockStateId,
        bounds: &mut Option<ColumnBounds>,
    ) -> Result<BlockStateId, TerrainError> {
        let Some(limits) = bounds else {
            // A missing map is primed by the general path.
            let previous = self.set_block_worldgen(x, y, z, state)?;
            *bounds = self.worldgen_column_bounds(x, z);
            return Ok(previous);
        };
        let Some(section) = self.blocks.section_index(y) else {
            return Ok(self.blocks.void_air);
        };
        if self.blocks.non_air[section] == 0 && state == self.blocks.air {
            return Ok(state);
        }
        // `BlockSection::set` rejects a state outside the registry first.
        let previous = self.set_block_raw(x, y, z, state)?;
        let (x, z) = (x & 15, z & 15);
        let properties = self.blocks.properties.heightmap_state(state);
        for &kind in self.stage.heightmaps() {
            if y <= limits[kind as usize] {
                continue;
            }
            let map = self.heightmaps[kind as usize]
                .as_mut()
                .expect("bounds are taken from present maps");
            if map.update(&self.blocks, x, y, z, properties)? {
                limits[kind as usize] = map.first_available(x, z)?.wrapping_sub(2);
            }
        }
        Ok(previous)
    }
    /// Whether writing `state` at every y of a run changes the stage's
    /// heightmaps as writing it at the run's top y alone does: an opaque block
    /// raises a map only at the y written (`Heightmap::update` returns below
    /// `first_available - 2`), and no fluid waits for post-processing.
    pub(crate) fn stage_heightmaps_opaque(&self, state: BlockStateId) -> bool {
        let properties = self.blocks.properties.heightmap_state(state);
        !properties.has_fluid()
            && self
                .stage
                .heightmaps()
                .iter()
                .all(|&kind| kind.is_opaque(properties))
    }
    pub fn position(&self) -> ChunkPos {
        self.position
    }
    pub fn min_y(&self) -> i32 {
        self.blocks.min_y
    }
    pub fn height(&self) -> i32 {
        self.blocks.height
    }
    pub fn highest_section_y(&self) -> i32 {
        self.blocks.highest_section_y()
    }
    /// ProtoChunk reads collapse an all-air section to its default AIR state.
    #[inline(always)]
    pub fn block(&self, x: i32, y: i32, z: i32) -> BlockStateId {
        self.blocks.block(x, y, z)
    }
    /// The states of the 16-block section starting at block `y` (a multiple
    /// of 16 inside this chunk), in `(y << 8) | (z << 4) | x` order, read in
    /// one pass; false (and `out` untouched) when `y` is not such a start.
    /// The section at `y` (a section's lowest Y) as `PalettedContainer::from_values` builds it from
    /// what `section_states` fills, without unpacking it (`BlockSection::to_container`).
    pub fn section_container(
        &self,
        y: i32,
        strategy: horizon_core::paletted::Strategy,
    ) -> Option<horizon_core::paletted::PalettedContainer> {
        match self.blocks.section_index(y) {
            Some(index) if y & 15 == 0 => Some(if self.blocks.non_air[index] == 0 {
                horizon_core::paletted::PalettedContainer::new(strategy, self.blocks.air.0)
            } else {
                self.blocks.sections[index].to_container(strategy)
            }),
            _ => None,
        }
    }
    pub fn section_states(&self, y: i32, out: &mut [u32; 4096]) -> bool {
        match self.blocks.section_index(y) {
            Some(index) if y & 15 == 0 => {
                self.blocks.section_states(index, out);
                true
            }
            _ => false,
        }
    }
    /// `FeatureWorld::scan_column` in this chunk: an all-air section is
    /// tested once, a mixed one read without looking the section up per block.
    pub fn scan_column(
        &self,
        [x, mut y, z]: [i32; 3],
        step: i32,
        mut limit: i32,
        mut keep: impl FnMut(BlockStateId) -> bool,
    ) -> i32 {
        debug_assert!(step == 1 || step == -1);
        let blocks = &self.blocks;
        while limit > 0 {
            let Some(section) = blocks.section_index(y) else {
                if !keep(blocks.void_air) {
                    return y;
                }
                y = y.wrapping_add(step);
                limit -= 1;
                continue;
            };
            // Blocks from y to this section's end in the walk's direction.
            let within = (y - blocks.min_y) & 15;
            let run = if step > 0 { 16 - within } else { within + 1 }.min(limit);
            limit -= run;
            if blocks.non_air[section] == 0 {
                if !keep(blocks.air) {
                    return y;
                }
                y = y.wrapping_add(step * run);
                continue;
            }
            let storage = &blocks.sections[section];
            for _ in 0..run {
                if !keep(storage.state(local_index(x, y, z))) {
                    return y;
                }
                y = y.wrapping_add(step);
            }
        }
        y
    }
    /// The blocks of column `x`, `z` from `low` up to `high`, as
    /// `visit(state, count)` runs: an all-air or one-state section's part at
    /// once, a mixed section's blocks one by one from its storage. The same
    /// states `block` reads.
    pub(crate) fn column_runs(
        &self,
        x: i32,
        z: i32,
        low: i32,
        high: i32,
        mut visit: impl FnMut(BlockStateId, usize),
    ) {
        let blocks = &self.blocks;
        let mut y = low;
        while y <= high {
            let Some(section) = blocks.section_index(y) else {
                visit(blocks.void_air, 1);
                if y == high {
                    break;
                }
                y += 1;
                continue;
            };
            // To this section's top, or `high`.
            let end = y.wrapping_add(15 - ((y - blocks.min_y) & 15)).min(high);
            let run = (end - y + 1) as usize;
            if blocks.non_air[section] == 0 {
                visit(blocks.air, run);
            } else if let Some(state) = blocks.sections[section].uniform_state() {
                visit(state, run);
            } else {
                let storage = &blocks.sections[section];
                for y in y..=end {
                    visit(storage.state(local_index(x, y, z)), 1);
                }
            }
            if end == i32::MAX {
                break;
            }
            y = end + 1;
        }
    }
    pub fn section(&self, index: usize) -> Option<&BlockSection<'r>> {
        self.blocks.sections.get(index)
    }
    pub fn heightmap(&self, kind: HeightmapType) -> Option<&Heightmap> {
        self.heightmaps[kind as usize].as_ref()
    }
    pub fn block_ticks(&self) -> &ProtoChunkTicks {
        &self.block_ticks
    }
    pub fn block_ticks_mut(&mut self) -> &mut ProtoChunkTicks {
        &mut self.block_ticks
    }
    pub fn fluid_ticks(&self) -> &ProtoChunkTicks {
        &self.fluid_ticks
    }
    pub fn fluid_ticks_mut(&mut self) -> &mut ProtoChunkTicks {
        &mut self.fluid_ticks
    }
    pub fn postprocessing(&self) -> &[Vec<u16>] {
        &self.postprocessing
    }

    pub fn ensure_heightmap(&mut self, kind: HeightmapType) -> &mut Heightmap {
        self.heightmaps[kind as usize].get_or_insert_with(|| {
            Heightmap::new(kind, self.blocks.min_y, self.blocks.height)
                .expect("validated chunk range")
        })
    }
    /// Section write only. The caller chooses which generation-stage maps to update.
    pub fn set_block_raw(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        state: BlockStateId,
    ) -> Result<BlockStateId, TerrainError> {
        let section = self
            .blocks
            .section_index(y)
            .ok_or(TerrainError::OutsideBuildHeight)?;
        let old = self.blocks.sections[section].set(local_index(x, y, z), state)?;
        let old_air = self.blocks.properties.heightmap_state(old).is_air();
        let new_air = self.blocks.properties.heightmap_state(state).is_air();
        if old_air && !new_air {
            self.blocks.non_air[section] += 1;
        } else if !old_air && new_air {
            self.blocks.non_air[section] -= 1;
        }
        Ok(old)
    }
    /// `set_block_raw` along one row of cells in this chunk: for each set bit `i` of `bits`, the
    /// cell `(x, y, z + i)` is read as `block` reads it, and written with the state `f` returns
    /// for it (given its Z), if any. Whether a cell was written.
    pub fn replace_row_raw(
        &mut self,
        [x, y, z]: [i32; 3],
        bits: u32,
        mut f: impl FnMut(i32, BlockStateId) -> Option<BlockStateId>,
    ) -> Result<bool, TerrainError> {
        let section = self.blocks.section_index(y);
        let mut wrote = false;
        let mut bits = bits;
        while bits != 0 {
            let zz = z.wrapping_add(bits.trailing_zeros() as i32);
            bits &= bits - 1;
            // A section with non-air blocks: read, test and write at one slot. Otherwise `f` sees
            // what `block` reads (void air outside the height, air in a section of air).
            let Some(index) = section.filter(|&s| self.blocks.non_air[s] != 0) else {
                let read = if section.is_some() {
                    self.blocks.air
                } else {
                    self.blocks.void_air
                };
                let Some(state) = f(zz, read) else {
                    continue;
                };
                let section = section.ok_or(TerrainError::OutsideBuildHeight)?;
                let old = self.blocks.sections[section].set(local_index(x, y, zz), state)?;
                self.blocks.count_write(section, old, state);
                wrote = true;
                continue;
            };
            let Some((old, state)) = self.blocks.sections[index]
                .replace_with(local_index(x, y, zz), |old| f(zz, old))?
            else {
                continue;
            };
            self.blocks.count_write(index, old, state);
            wrote = true;
        }
        Ok(wrote)
    }
    /// [`replace_row_raw`](Self::replace_row_raw) for an `f` whose answer depends only on the
    /// state it is given: the same cells read, the same writes in the same order. `f` is asked once
    /// per state met in the row, and the non-air count of a section changes by what its answer
    /// gives, worked out once per state too.
    pub fn replace_row_states(
        &mut self,
        [x, y, z]: [i32; 3],
        bits: u32,
        f: &mut dyn FnMut(BlockStateId) -> Option<BlockStateId>,
    ) -> Result<bool, TerrainError> {
        let section = self.blocks.section_index(y);
        let mut memo = StateMemo::default();
        let mut wrote = false;
        let mut bits = bits;
        while bits != 0 {
            let zz = z.wrapping_add(bits.trailing_zeros() as i32);
            bits &= bits - 1;
            let li = local_index(x, y, zz);
            let Some(index) = section.filter(|&s| self.blocks.non_air[s] != 0) else {
                let read = if section.is_some() {
                    self.blocks.air
                } else {
                    self.blocks.void_air
                };
                let Some((state, _)) = memo.get(read, f, &self.blocks.properties) else {
                    continue;
                };
                let section = section.ok_or(TerrainError::OutsideBuildHeight)?;
                let old = self.blocks.sections[section].set(li, state)?;
                self.blocks.count_write(section, old, state);
                wrote = true;
                continue;
            };
            let Some((slot, old_id, old)) = self.blocks.sections[index].slot_state(li) else {
                let old = self.blocks.sections[index].state(li);
                let Some((state, _)) = memo.get(old, f, &self.blocks.properties) else {
                    continue;
                };
                let old = self.blocks.sections[index].set(li, state)?;
                self.blocks.count_write(index, old, state);
                wrote = true;
                continue;
            };
            let Some((state, delta)) = memo.get(old, f, &self.blocks.properties) else {
                continue;
            };
            wrote = true;
            if state == old {
                continue;
            }
            self.blocks.sections[index].set_at(li, slot, old_id, state)?;
            self.blocks.non_air[index] = self.blocks.non_air[index].wrapping_add_signed(delta);
        }
        Ok(wrote)
    }
    /// Write a vertical run of one non-air state from `y_top` downward, all
    /// inside the chunk. Returns how many blocks changed from air; section
    /// non-air counts are updated here and the caller owns the heightmaps.
    pub fn fill_solid_run(
        &mut self,
        x: i32,
        y_top: i32,
        z: i32,
        len: i32,
        state: BlockStateId,
    ) -> Result<u16, TerrainError> {
        let air = self.blocks.air;
        let mut added = 0u16;
        let mut top = y_top;
        let mut left = len;
        while left > 0 {
            let section = self
                .blocks
                .section_index(top)
                .ok_or(TerrainError::OutsideBuildHeight)?;
            let local_top = top & 15;
            let take = (local_top + 1).min(left);
            let local_start = local_top - take + 1;
            let now = self.blocks.sections[section].fill_run(
                x & 15,
                local_start,
                z & 15,
                take,
                state,
                air,
            )?;
            self.blocks.non_air[section] = self.blocks.non_air[section].saturating_add(now);
            added = added.saturating_add(now);
            top -= take;
            left -= take;
        }
        Ok(added)
    }
    pub fn update_heightmap(
        &mut self,
        kind: HeightmapType,
        x: i32,
        y: i32,
        z: i32,
        state: BlockStateId,
    ) -> Result<bool, TerrainError> {
        if self.blocks.registry.state(state).is_none() {
            return Err(TerrainError::InvalidState);
        }
        self.ensure_heightmap(kind);
        let state = self.blocks.properties.heightmap_state(state);
        Ok(self.heightmaps[kind as usize]
            .as_mut()
            .unwrap()
            .update(&self.blocks, x, y, z, state)?)
    }
    pub fn prime_heightmaps(&mut self, kinds: &[HeightmapType]) -> Result<(), TerrainError> {
        let mut order = Vec::new();
        for &kind in kinds {
            self.ensure_heightmap(kind);
            if !order.contains(&(kind as usize)) {
                order.push(kind as usize);
            }
        }
        // Split references preserve the supplied set's iteration order.
        let mut maps: Vec<_> = self
            .heightmaps
            .iter_mut()
            .enumerate()
            .filter_map(|(i, map)| {
                order
                    .iter()
                    .position(|&key| key == i)
                    .map(|position| (position, map.as_mut().unwrap()))
            })
            .collect();
        maps.sort_by_key(|(position, _)| *position);
        Heightmap::prime(
            &self.blocks,
            &mut maps.into_iter().map(|(_, map)| map).collect::<Vec<_>>(),
        )?;
        Ok(())
    }
    /// Before replaying a saved chunk's `PostProcessing` lists.
    pub fn clear_postprocessing(&mut self) {
        self.postprocessing.iter_mut().for_each(Vec::clear);
    }
    /// Preserves duplicates and insertion order, as ProtoChunk does.
    pub fn mark_postprocessing(&mut self, x: i32, y: i32, z: i32) {
        if let Some(section) = self.blocks.section_index(y) {
            self.postprocessing[section]
                .push(((x & 15) | ((y & 15) << 4) | ((z & 15) << 8)) as u16);
        }
    }
}

/// Evaluate final density in bulk before writes, then visit Z/X/descending Y.
/// The aquifer and final density must share this already prepared sampler context.
/// Failures retain earlier writes and aquifer/cache state; there is no rollback.
/// Regular gameplay only: developer preliminary-surface markers are not applied.
pub fn fill_noise<B: TerrainBlockStates>(
    chunk: &mut TerrainChunk<'_, B>,
    volume: DensityVolume,
    final_density: &dyn DensitySampler,
    context: &mut SamplerContext,
    aquifer: &mut (impl TerrainAquifer + ?Sized),
    default_block: BlockStateId,
) -> Result<(), TerrainError> {
    fill_noise_with(
        chunk,
        volume,
        final_density,
        context,
        aquifer,
        default_block,
        |_| false,
    )
}

/// [`fill_noise`] whose final density `presample` may write itself (the GPU
/// path): when it returns true, `final_density` is not sampled on the CPU.
pub fn fill_noise_with<B: TerrainBlockStates>(
    chunk: &mut TerrainChunk<'_, B>,
    volume: DensityVolume,
    final_density: &dyn DensitySampler,
    context: &mut SamplerContext,
    aquifer: &mut (impl TerrainAquifer + ?Sized),
    default_block: BlockStateId,
    presample: impl FnOnce(&mut [f32]) -> bool,
) -> Result<(), TerrainError> {
    let size = volume.size();
    SCRATCH_DENSITY.with_borrow_mut(|scratch| {
        if scratch.len() < size {
            scratch.resize(size, 0.0);
        }
        scratch[..size].fill(0.0);
        let sampled = presample(&mut scratch[..size]);
        fill_noise_inner(
            chunk,
            volume,
            (!sampled).then_some(final_density),
            context,
            aquifer,
            default_block,
            &mut scratch[..size],
        )
    })
}

/// [`fill_noise`] from block codes the GPU decided from the final density and
/// the aquifer: 0 positive density (written in runs, as the CPU fill does), 8
/// default block by aquifer pressure, 1 air, 2 default fluid, 3 lava, plus 4
/// for a scheduled fluid update. Writes the same blocks, in the same order and
/// with the same heightmap updates, as the CPU fill.
pub fn fill_noise_codes<B: TerrainBlockStates>(
    chunk: &mut TerrainChunk<'_, B>,
    volume: DensityVolume,
    codes: &[i8],
    default_block: BlockStateId,
    fluid: BlockStateId,
    lava: BlockStateId,
) -> Result<(), TerrainError> {
    let [sx, sy, sz] = volume.sizes();
    let [bx, by, bz] = volume.minimum();
    if sx != 16
        || sz != 16
        || volume.steps() != [1; 3]
        || codes.len() != volume.size()
        || bx != chunk.position.x.wrapping_shl(4)
        || bz != chunk.position.z.wrapping_shl(4)
        || by < chunk.min_y()
        || i64::from(by) + sy as i64 > i64::from(chunk.min_y()) + i64::from(chunk.height())
        || chunk.blocks.registry.state(default_block).is_none()
    {
        return Err(TerrainError::InvalidVolume);
    }
    chunk.ensure_heightmap(HeightmapType::OceanFloorWg);
    chunk.ensure_heightmap(HeightmapType::WorldSurfaceWg);
    let solid_runs = default_block != chunk.blocks.air
        && !chunk
            .blocks
            .properties
            .heightmap_state(default_block)
            .has_fluid();
    const MAPS: [HeightmapType; 2] = [HeightmapType::OceanFloorWg, HeightmapType::WorldSurfaceWg];
    for z in 0..sz {
        let world_z = bz.wrapping_add(z as i32);
        for x in 0..sx {
            let world_x = bx.wrapping_add(x as i32);
            let col_base = (x + z * sx) * sy;
            let col = &codes[col_base..col_base + sy];
            let mut bounds = MAPS.map(|kind| {
                chunk.heightmaps[kind as usize]
                    .as_ref()
                    .expect("ensured heightmap")
                    .first_available(x as i32, z as i32)
                    .map(|first| first.wrapping_sub(2))
            });
            let mut y = sy;
            while y > 0 {
                y -= 1;
                let world_y = by.wrapping_add(y as i32);
                let code = col[y];
                if solid_runs && code == 0 {
                    let mut run = 1usize;
                    while run <= y && col[y - run] == 0 {
                        run += 1;
                    }
                    chunk.fill_solid_run(x as i32, world_y, z as i32, run as i32, default_block)?;
                    for (kind, bound) in MAPS.into_iter().zip(&mut bounds) {
                        if matches!(bound, Ok(current) if world_y <= *current) {
                            continue;
                        }
                        chunk.update_heightmap(kind, x as i32, world_y, z as i32, default_block)?;
                        *bound = chunk.heightmaps[kind as usize]
                            .as_ref()
                            .expect("ensured heightmap")
                            .first_available(x as i32, z as i32)
                            .map(|first| first.wrapping_sub(2));
                    }
                    y -= run - 1;
                    continue;
                }
                let state = match code & 3 {
                    0 => default_block,
                    1 => continue,
                    2 => fluid,
                    _ => lava,
                };
                if state == chunk.blocks.air {
                    continue;
                }
                chunk.set_block_raw(x as i32, world_y, z as i32, state)?;
                for (kind, bound) in MAPS.into_iter().zip(&mut bounds) {
                    if matches!(bound, Ok(current) if world_y <= *current) {
                        continue;
                    }
                    chunk.update_heightmap(kind, x as i32, world_y, z as i32, state)?;
                    *bound = chunk.heightmaps[kind as usize]
                        .as_ref()
                        .expect("ensured heightmap")
                        .first_available(x as i32, z as i32)
                        .map(|first| first.wrapping_sub(2));
                }
                if code & 4 != 0 && chunk.blocks.properties.heightmap_state(state).has_fluid() {
                    chunk.mark_postprocessing(world_x, world_y, world_z);
                }
            }
        }
    }
    Ok(())
}

/// Like `fill_noise_codes`, from the chunk's final states after the fill and the surface, computed on
/// the GPU (`horizon_gpu::SurfaceGpu::chunk_states`): the state in the low 30 bits, bit 31 where the
/// fill scheduled a fluid update, bit 30 where the surface wrote a state with fluid. The surface only
/// replaces solid blocks and fills air or water, so one pass in the fill's order gives the fill's
/// post-processing marks and the heightmaps of the final blocks; the surface's own marks follow, in
/// its order (x then z, descending y), as the fill followed by the surface leaves them.
pub fn fill_noise_states<B: TerrainBlockStates>(
    chunk: &mut TerrainChunk<'_, B>,
    volume: DensityVolume,
    states: &[u32],
) -> Result<(), TerrainError> {
    let [sx, sy, sz] = volume.sizes();
    let [bx, by, bz] = volume.minimum();
    if sx != 16
        || sz != 16
        || volume.steps() != [1; 3]
        || states.len() != volume.size()
        || bx != chunk.position.x.wrapping_shl(4)
        || bz != chunk.position.z.wrapping_shl(4)
        || by < chunk.min_y()
        || i64::from(by) + sy as i64 > i64::from(chunk.min_y()) + i64::from(chunk.height())
    {
        return Err(TerrainError::InvalidVolume);
    }
    chunk.ensure_heightmap(HeightmapType::OceanFloorWg);
    chunk.ensure_heightmap(HeightmapType::WorldSurfaceWg);
    const MAPS: [HeightmapType; 2] = [HeightmapType::OceanFloorWg, HeightmapType::WorldSurfaceWg];
    let air = chunk.blocks.air;
    for z in 0..sz {
        let world_z = bz.wrapping_add(z as i32);
        for x in 0..sx {
            let world_x = bx.wrapping_add(x as i32);
            let col_base = (x + z * sx) * sy;
            let col = &states[col_base..col_base + sy];
            let mut bounds = MAPS.map(|kind| {
                chunk.heightmaps[kind as usize]
                    .as_ref()
                    .expect("ensured heightmap")
                    .first_available(x as i32, z as i32)
                    .map(|first| first.wrapping_sub(2))
            });
            let mut y = sy;
            while y > 0 {
                y -= 1;
                let world_y = by.wrapping_add(y as i32);
                let value = col[y];
                let state = BlockStateId(value & 0x3fff_ffff);
                if state == air {
                    continue;
                }
                let fluid = chunk.blocks.properties.heightmap_state(state).has_fluid();
                // A run of one state without fluid nor mark: written through the section fill, the
                // heightmaps updated at its top (the column is filled downwards).
                let mut run = 1usize;
                if !fluid && value & 0xc000_0000 == 0 {
                    while run <= y && col[y - run] == value {
                        run += 1;
                    }
                    chunk.fill_solid_run(x as i32, world_y, z as i32, run as i32, state)?;
                } else {
                    chunk.set_block_raw(x as i32, world_y, z as i32, state)?;
                }
                for (kind, bound) in MAPS.into_iter().zip(&mut bounds) {
                    if matches!(bound, Ok(current) if world_y <= *current) {
                        continue;
                    }
                    chunk.update_heightmap(kind, x as i32, world_y, z as i32, state)?;
                    *bound = chunk.heightmaps[kind as usize]
                        .as_ref()
                        .expect("ensured heightmap")
                        .first_available(x as i32, z as i32)
                        .map(|first| first.wrapping_sub(2));
                }
                // The fill's mark stays where the surface later froze the water.
                if value & 0x8000_0000 != 0 {
                    chunk.mark_postprocessing(world_x, world_y, world_z);
                }
                y -= run - 1;
            }
        }
    }
    if states.iter().any(|v| v & 0x4000_0000 != 0) {
        for x in 0..sx {
            for z in 0..sz {
                let col_base = (x + z * sx) * sy;
                for y in (0..sy).rev() {
                    if states[col_base + y] & 0x4000_0000 != 0 {
                        chunk.mark_postprocessing(
                            bx.wrapping_add(x as i32),
                            by.wrapping_add(y as i32),
                            bz.wrapping_add(z as i32),
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

impl<B: TerrainBlockStates> TerrainChunk<'_, B> {
    /// Per state of the registry, what the GPU's compact sections read
    /// (`horizon_gpu::SurfaceGpu::enable_sections`): 1 air, 2 fluid, 4 blocks motion in a heightmap.
    pub fn section_flags(&self) -> Vec<u8> {
        (0..self.blocks.registry.len())
            .map(|i| {
                let s = self
                    .blocks
                    .properties
                    .heightmap_state(BlockStateId(i as u32));
                u8::from(s.is_air())
                    | u8::from(s.has_fluid()) << 1
                    | u8::from(s.blocks_motion_in_heightmap()) << 2
            })
            .collect()
    }
    /// The chunk's `minecraft:air`.
    pub fn air(&self) -> BlockStateId {
        self.blocks.air
    }
    /// Nothing written yet: every section air, no generation heightmap, no post-processing mark.
    pub fn is_fresh(&self) -> bool {
        self.blocks.non_air.iter().all(|&n| n == 0)
            && self
                .blocks
                .sections
                .iter()
                .all(|s| s.uniform_state() == Some(self.blocks.air))
            && self.heightmaps[HeightmapType::OceanFloorWg as usize].is_none()
            && self.heightmaps[HeightmapType::WorldSurfaceWg as usize].is_none()
            && self.postprocessing.iter().all(Vec::is_empty)
    }
}

/// One section of the GPU's compact output (`horizon_gpu::SECTION_WORDS` words, sections.cu).
fn gpu_section<'r>(
    data: &[u32],
    s: usize,
    air: BlockStateId,
    registry: &'r BlockRegistry,
) -> Result<(BlockSection<'r>, u16), TerrainError> {
    let o = &data[s * horizon_gpu::SECTION_WORDS..(s + 1) * horizon_gpu::SECTION_WORDS];
    let (kind, bits, len) = (
        o[0] & 0xFF,
        ((o[0] >> 8) & 0xFF) as u8,
        (o[0] >> 16) as usize,
    );
    let non_air = u16::try_from(o[1]).map_err(|_| TerrainError::InvalidState)?;
    let section = match kind {
        0 => BlockSection::uniform(air, registry)?,
        1 => BlockSection::uniform(BlockStateId(o[2]), registry)?,
        2 => {
            const K: usize = 128;
            if len == 0 || len > K + 1 {
                return Err(TerrainError::InvalidState);
            }
            let palette = o[5..5 + len].iter().map(|&v| BlockStateId(v)).collect();
            let counts = o[5 + K..5 + K + len]
                .iter()
                .map(|&c| u16::try_from(c).map_err(|_| TerrainError::InvalidState))
                .collect::<Result<Vec<_>, _>>()?;
            let free = if o[4] != 0 { vec![0] } else { Vec::new() };
            let word_count = o[3] as usize;
            let d = &o[5 + 2 * K..];
            if word_count * 2 > d.len() {
                return Err(TerrainError::InvalidState);
            }
            let words = (0..word_count)
                .map(|w| u64::from(d[2 * w]) | (u64::from(d[2 * w + 1]) << 32))
                .collect();
            let storage = horizon_core::storage::PackedStorage::from_words(bits, 4096, words)
                .map_err(|_| TerrainError::InvalidState)?;
            BlockSection::from_parts(registry, palette, counts, free, Some(storage))?
        }
        _ => return Err(TerrainError::InvalidState),
    };
    Ok((section, non_air))
}

/// Where the parts of the GPU's compact output start: heights, then the two mark lists.
fn gpu_layout(data: &[u32], sections: usize) -> Result<(usize, usize), TerrainError> {
    let hm = sections * horizon_gpu::SECTION_WORDS;
    let mq = hm + 512;
    if data.len() != mq + 2 + 2 * horizon_gpu::SECTION_MARKS + 1 || data[data.len() - 1] != 0 {
        return Err(TerrainError::InvalidVolume);
    }
    Ok((hm, mq))
}

/// `fill_noise_states` from the sections the GPU built (`horizon_gpu::SurfaceGpu::enable_sections`):
/// each section as `BlockSection::set` would have left it, its non-air count, the generation
/// heightmaps and the post-processing marks in `fill_noise_states`' order. The chunk must be fresh
/// (`TerrainChunk::is_fresh`) and the volume its whole height.
pub fn fill_noise_sections<B: TerrainBlockStates>(
    chunk: &mut TerrainChunk<'_, B>,
    volume: DensityVolume,
    data: &[u32],
) -> Result<(), TerrainError> {
    let [sx, sy, sz] = volume.sizes();
    let [bx, by, bz] = volume.minimum();
    let sections = chunk.blocks.sections.len();
    if sx != 16
        || sz != 16
        || volume.steps() != [1; 3]
        || bx != chunk.position.x.wrapping_shl(4)
        || bz != chunk.position.z.wrapping_shl(4)
        || by != chunk.min_y()
        || sy as i32 != chunk.height()
        || sy != 16 * sections
        || !chunk.is_fresh()
    {
        return Err(TerrainError::InvalidVolume);
    }
    let (hm, mq) = gpu_layout(data, sections)?;
    let (air, registry) = (chunk.blocks.air, chunk.blocks.registry);
    let mut built = Vec::with_capacity(sections);
    for s in 0..sections {
        built.push(gpu_section(data, s, air, registry)?);
    }
    for (s, (section, non_air)) in built.into_iter().enumerate() {
        chunk.blocks.sections[s] = section;
        chunk.blocks.non_air[s] = non_air;
    }
    let min_y = chunk.min_y();
    for (i, kind) in [HeightmapType::OceanFloorWg, HeightmapType::WorldSurfaceWg]
        .into_iter()
        .enumerate()
    {
        let map = chunk.ensure_heightmap(kind);
        for t in 0..256 {
            let value = data[hm + 256 * i + t] as i32;
            map.set_first_available((t & 15) as i32, (t >> 4) as i32, min_y.wrapping_add(value))?;
        }
    }
    let (na, nb) = (data[mq] as usize, data[mq + 1] as usize);
    if na > horizon_gpu::SECTION_MARKS || nb > horizon_gpu::SECTION_MARKS {
        return Err(TerrainError::InvalidVolume);
    }
    let a = &data[mq + 2..mq + 2 + na];
    let b = &data[mq + 2 + horizon_gpu::SECTION_MARKS..mq + 2 + horizon_gpu::SECTION_MARKS + nb];
    for &m in a.iter().chain(b) {
        let section = (m >> 16) as usize;
        chunk
            .postprocessing
            .get_mut(section)
            .ok_or(TerrainError::InvalidVolume)?
            .push(m as u16);
    }
    Ok(())
}

/// HORIZON_GPU_SECTIONS=test: whether a chunk `fill_noise_states` just filled holds what the GPU's
/// compact output describes (sections and their parts, non-air counts, generation heightmaps,
/// post-processing marks); the first difference otherwise.
pub fn check_noise_sections<B: TerrainBlockStates>(
    chunk: &TerrainChunk<'_, B>,
    data: &[u32],
) -> Result<(), String> {
    let sections = chunk.blocks.sections.len();
    let (hm, mq) = gpu_layout(data, sections).map_err(|e| format!("{e:?}"))?;
    for s in 0..sections {
        let (gpu, non_air) = gpu_section(data, s, chunk.blocks.air, chunk.blocks.registry)
            .map_err(|e| format!("section {s}: {e:?}"))?;
        let cpu = &chunk.blocks.sections[s];
        let (pg, cg, fg, sg) = gpu.parts();
        let (pc, cc, fc, sc) = cpu.parts();
        if pg != pc || cg != cc || fg != fc {
            return Err(format!(
                "section {s}: palette {pg:?} {cg:?} {fg:?} against {pc:?} {cc:?} {fc:?}"
            ));
        }
        if sg.map(|s| (s.bits(), s.words())) != sc.map(|s| (s.bits(), s.words())) {
            return Err(format!("section {s}: different words"));
        }
        if non_air != chunk.blocks.non_air[s] {
            return Err(format!(
                "section {s}: non-air {non_air} against {}",
                chunk.blocks.non_air[s]
            ));
        }
    }
    let min_y = chunk.min_y();
    for (i, kind) in [HeightmapType::OceanFloorWg, HeightmapType::WorldSurfaceWg]
        .into_iter()
        .enumerate()
    {
        let map = chunk.heightmaps[kind as usize]
            .as_ref()
            .ok_or("missing heightmap")?;
        for t in 0..256 {
            let gpu = min_y.wrapping_add(data[hm + 256 * i + t] as i32);
            let cpu = map
                .first_available((t & 15) as i32, (t >> 4) as i32)
                .map_err(|e| format!("{e:?}"))?;
            if gpu != cpu {
                return Err(format!(
                    "heightmap {kind:?} column {t}: {gpu} against {cpu}"
                ));
            }
        }
    }
    let (na, nb) = (data[mq] as usize, data[mq + 1] as usize);
    let mut marks = vec![Vec::new(); sections];
    let m = horizon_gpu::SECTION_MARKS;
    for &v in data[mq + 2..mq + 2 + na.min(m)]
        .iter()
        .chain(&data[mq + 2 + m..mq + 2 + m + nb.min(m)])
    {
        if let Some(l) = marks.get_mut((v >> 16) as usize) {
            l.push(v as u16);
        }
    }
    if marks != chunk.postprocessing {
        return Err(format!(
            "marks: {na} + {nb} against {:?}",
            chunk
                .postprocessing
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>()
        ));
    }
    Ok(())
}

fn fill_noise_inner<B: TerrainBlockStates>(
    chunk: &mut TerrainChunk<'_, B>,
    volume: DensityVolume,
    final_density: Option<&dyn DensitySampler>,
    context: &mut SamplerContext,
    aquifer: &mut (impl TerrainAquifer + ?Sized),
    default_block: BlockStateId,
    density: &mut [f32],
) -> Result<(), TerrainError> {
    let [sx, sy, sz] = volume.sizes();
    let [bx, by, bz] = volume.minimum();
    if sx != 16
        || sz != 16
        || volume.steps() != [1; 3]
        || bx != chunk.position.x.wrapping_shl(4)
        || bz != chunk.position.z.wrapping_shl(4)
        || by < chunk.min_y()
        || i64::from(by) + sy as i64 > i64::from(chunk.min_y()) + i64::from(chunk.height())
        || chunk.blocks.registry.state(default_block).is_none()
    {
        return Err(TerrainError::InvalidVolume);
    }
    chunk.ensure_heightmap(HeightmapType::OceanFloorWg);
    chunk.ensure_heightmap(HeightmapType::WorldSurfaceWg);
    if let Some(final_density) = final_density {
        final_density.sample_volume(context, density, volume)?;
    }
    let air_above = aquifer.air_above_y();
    // A non-air, non-fluid default block may be written as a run without asking
    // the aquifer or touching the heightmap below the run's top.
    let solid_runs = default_block != chunk.blocks.air
        && !chunk
            .blocks
            .properties
            .heightmap_state(default_block)
            .has_fluid();
    const MAPS: [HeightmapType; 2] = [HeightmapType::OceanFloorWg, HeightmapType::WorldSurfaceWg];
    for z in 0..sz {
        let world_z = bz.wrapping_add(z as i32);
        for x in 0..sx {
            let world_x = bx.wrapping_add(x as i32);
            let col_base = (x + z * sx) * sy;
            let col_density = &density[col_base..col_base + sy];
            // Heightmap updates at or below first-available - 2 are no-ops. In a
            // top-down vertical sweep, once an opaque block updates first-available
            // to world_y + 1, all lower blocks have world_y <= bound and can be
            // skipped entirely.
            let mut bounds = MAPS.map(|kind| {
                chunk.heightmaps[kind as usize]
                    .as_ref()
                    .expect("ensured heightmap")
                    .first_available(x as i32, z as i32)
                    .map(|first| first.wrapping_sub(2))
            });
            let mut y = sy;
            while y > 0 {
                y -= 1;
                let world_y = by.wrapping_add(y as i32);
                let value = col_density[y];
                // A positive-density run is entirely the default block. Write it
                // through the section once instead of per block; the heightmap
                // only changes at the run's top.
                if solid_runs && value > 0.0 {
                    let mut run = 1usize;
                    while run <= y && col_density[y - run] > 0.0 {
                        run += 1;
                    }
                    aquifer.note_skipped();
                    chunk.fill_solid_run(x as i32, world_y, z as i32, run as i32, default_block)?;
                    for (kind, bound) in MAPS.into_iter().zip(&mut bounds) {
                        if matches!(bound, Ok(current) if world_y <= *current) {
                            continue;
                        }
                        chunk.update_heightmap(kind, x as i32, world_y, z as i32, default_block)?;
                        *bound = chunk.heightmaps[kind as usize]
                            .as_ref()
                            .expect("ensured heightmap")
                            .first_available(x as i32, z as i32)
                            .map(|first| first.wrapping_sub(2));
                    }
                    y -= run - 1;
                    continue;
                }
                // The aquifer answers `None` for positive density, which the fill
                // writes as the default block, and air above its skip/fluid level.
                // Both are decided here so the per-block call is not made at all,
                // with the same pending-update side effect the call had.
                let solid = value > 0.0;
                if solid {
                    aquifer.note_skipped();
                }
                let state = if solid {
                    default_block
                } else if let Some(limit) = air_above.filter(|&limit| world_y > limit) {
                    // Air above the aquifer's level, and so is the run of
                    // non-positive densities below that stays above it: nothing is
                    // written for them and `note_skipped` only clears a flag, so
                    // one call covers the run. A NaN (not `<= 0.0`) ends the run and
                    // starts its own.
                    let mut run = 1usize;
                    while run <= y
                        && col_density[y - run] <= 0.0
                        && world_y.wrapping_sub(run as i32) > limit
                    {
                        run += 1;
                    }
                    aquifer.note_skipped();
                    y -= run - 1;
                    continue;
                } else {
                    aquifer
                        .compute_substance(context, [world_x, world_y, world_z], f64::from(value))?
                        .unwrap_or(default_block)
                };
                if state == chunk.blocks.air {
                    continue;
                }
                chunk.set_block_raw(x as i32, world_y, z as i32, state)?;
                for (kind, bound) in MAPS.into_iter().zip(&mut bounds) {
                    if matches!(bound, Ok(current) if world_y <= *current) {
                        continue;
                    }
                    chunk.update_heightmap(kind, x as i32, world_y, z as i32, state)?;
                    *bound = chunk.heightmaps[kind as usize]
                        .as_ref()
                        .expect("ensured heightmap")
                        .first_available(x as i32, z as i32)
                        .map(|first| first.wrapping_sub(2));
                }
                if aquifer.should_schedule_fluid_update()
                    && chunk.blocks.properties.heightmap_state(state).has_fluid()
                {
                    chunk.mark_postprocessing(world_x, world_y, world_z);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbt::Tag;
    use crate::worldgen::aquifer::FluidPicker;
    use horizon_core::density::ConstantSampler;

    #[derive(Clone, Copy)]
    struct State(u32);
    impl HeightmapState for State {
        fn is_air(self) -> bool {
            self.0 < 2
        }
        fn is_air_block(self) -> bool {
            self.0 == 0
        }
        fn has_fluid(self) -> bool {
            self.0 == 3
        }
        fn blocks_motion_in_heightmap(self) -> bool {
            self.0 == 2
        }
        fn blocks_motion_in_heightmap_no_leaves(self) -> bool {
            self.0 == 2
        }
    }
    struct Properties;
    impl TerrainBlockStates for Properties {
        type State = State;
        fn heightmap_state(&self, id: BlockStateId) -> Self::State {
            State(id.0)
        }
    }
    #[derive(Debug)]
    struct TestPicker;
    impl FluidPicker for TestPicker {
        fn compute(&self, _: [i32; 3]) -> crate::worldgen::aquifer::FluidStatus {
            crate::worldgen::aquifer::FluidStatus {
                level: -64,
                state: BlockStateId(3),
            }
        }
    }

    fn test_registry() -> BlockRegistry {
        BlockRegistry::from_nbt(&Tag::compound([
            ("DataVersion", Tag::Int(crate::DATA_VERSION)),
            (
                "states",
                Tag::List(vec![
                    Tag::compound([("id", Tag::String("minecraft:air".into()))]),
                    Tag::compound([("id", Tag::String("minecraft:void_air".into()))]),
                    Tag::compound([("id", Tag::String("minecraft:stone".into()))]),
                    Tag::compound([("id", Tag::String("minecraft:water".into()))]),
                ]),
            ),
            (
                "defaults",
                Tag::compound([
                    ("minecraft:air", Tag::Int(0)),
                    ("minecraft:void_air", Tag::Int(1)),
                    ("minecraft:stone", Tag::Int(2)),
                    ("minecraft:water", Tag::Int(3)),
                ]),
            ),
        ]))
        .unwrap()
    }

    /// A registry of air, void air and `n - 2` other states (ids below 2 are air for `Properties`).
    fn wide_registry(n: u32) -> BlockRegistry {
        let names: Vec<String> = (0..n)
            .map(|i| match i {
                0 => "minecraft:air".to_owned(),
                1 => "minecraft:void_air".to_owned(),
                _ => format!("minecraft:s{i}"),
            })
            .collect();
        BlockRegistry::from_nbt(&Tag::compound([
            ("DataVersion", Tag::Int(crate::DATA_VERSION)),
            (
                "states",
                Tag::List(
                    names
                        .iter()
                        .map(|n| Tag::compound([("id", Tag::String(n.as_str().into()))]))
                        .collect(),
                ),
            ),
            (
                "defaults",
                Tag::compound(
                    names
                        .iter()
                        .enumerate()
                        .map(|(i, n)| (n.as_str(), Tag::Int(i as i32))),
                ),
            ),
        ]))
        .unwrap()
    }

    /// `replace_row_states` reads and writes as `replace_row_raw` with the same decision: same
    /// answers, same sections (palette order, counts, free entries and words, as `to_nbt` writes
    /// them) and same non-air counts, through uniform, air, growing and shrinking sections.
    #[test]
    fn replace_row_states_matches_replace_row_raw() {
        let registry = wide_registry(40);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for round in 0..40 {
            let mut a =
                TerrainChunk::new(ChunkPos::new(0, 0), -64, 384, &registry, Properties).unwrap();
            let mut b =
                TerrainChunk::new(ChunkPos::new(0, 0), -64, 384, &registry, Properties).unwrap();
            // Few states per round for the first rounds (long runs, uniform sections), up to 38.
            let kinds = 2 + (round as u64 % 8) * 5;
            for op in 0..3000 {
                let y = -70 + next(400) as i32;
                let x = next(16) as i32;
                let z = next(16) as i32;
                let bits = (next(1 << 16) as u32) & (u32::MAX >> (z as u32 + 16));
                // A decision on the state only: none, or one of a few states (air at times).
                let table: Vec<Option<BlockStateId>> = (0..40)
                    .map(|_| match next(10) {
                        0..=3 => None,
                        4 => Some(BlockStateId(next(2) as u32)),
                        _ => Some(BlockStateId(2 + next(kinds.min(38)) as u32)),
                    })
                    .collect();
                let ra = a.replace_row_raw([x, y, z], bits, |_, s| table[s.0 as usize]);
                let rb = b.replace_row_states([x, y, z], bits, &mut |s| table[s.0 as usize]);
                assert_eq!(ra.is_ok(), rb.is_ok(), "round {round} op {op}");
                if let (Ok(ra), Ok(rb)) = (ra, rb) {
                    assert_eq!(ra, rb, "round {round} op {op}");
                }
            }
            assert_eq!(a.blocks.non_air, b.blocks.non_air, "round {round}");
            for (i, (sa, sb)) in a.blocks.sections.iter().zip(&b.blocks.sections).enumerate() {
                assert_eq!(
                    sa.to_nbt().unwrap(),
                    sb.to_nbt().unwrap(),
                    "round {round} section {i}"
                );
            }
        }
    }

    #[test]
    fn fill_noise_poisoned_scratch_does_not_leak() {
        let registry = test_registry();
        let volume = DensityVolume::new([16, 384, 16], [0, -64, 0], [1, 1, 1]).unwrap();
        let sampler = ConstantSampler(1.0);
        let mut context = SamplerContext::sparse();

        let mut chunk_clean =
            TerrainChunk::new(ChunkPos::new(0, 0), -64, 384, &registry, Properties).unwrap();
        let mut aquifer_clean = DisabledAquifer::new(TestPicker, BlockStateId(0));
        fill_noise(
            &mut chunk_clean,
            volume,
            &sampler,
            &mut context,
            &mut aquifer_clean,
            BlockStateId(2),
        )
        .unwrap();

        // Deliberately poison the thread-local scratch buffer with NaN
        SCRATCH_DENSITY.with_borrow_mut(|scratch| {
            scratch.fill(f32::NAN);
        });

        let mut chunk_poisoned =
            TerrainChunk::new(ChunkPos::new(0, 0), -64, 384, &registry, Properties).unwrap();
        let mut aquifer_poisoned = DisabledAquifer::new(TestPicker, BlockStateId(0));
        fill_noise(
            &mut chunk_poisoned,
            volume,
            &sampler,
            &mut context,
            &mut aquifer_poisoned,
            BlockStateId(2),
        )
        .unwrap();

        for (sec1, sec2) in chunk_clean
            .blocks
            .sections
            .iter()
            .zip(&chunk_poisoned.blocks.sections)
        {
            for idx in 0..4096 {
                assert_eq!(sec1.get(idx).unwrap(), sec2.get(idx).unwrap());
            }
        }
        for (h1, h2) in chunk_clean
            .heightmaps
            .iter()
            .zip(&chunk_poisoned.heightmaps)
        {
            assert_eq!(
                h1.as_ref().map(|m| m.raw_data()),
                h2.as_ref().map(|m| m.raw_data())
            );
        }
    }

    #[test]
    fn fill_noise_heightmap_matches_top_down_surface() {
        let registry = test_registry();
        let volume = DensityVolume::new([16, 384, 16], [0, -64, 0], [1, 1, 1]).unwrap();
        let sampler = ConstantSampler(1.0);
        let mut context = SamplerContext::sparse();

        let mut chunk =
            TerrainChunk::new(ChunkPos::new(0, 0), -64, 384, &registry, Properties).unwrap();
        let mut aquifer = DisabledAquifer::new(TestPicker, BlockStateId(0));
        fill_noise(
            &mut chunk,
            volume,
            &sampler,
            &mut context,
            &mut aquifer,
            BlockStateId(2),
        )
        .unwrap();

        let ocean_floor = chunk.heightmaps[HeightmapType::OceanFloorWg as usize]
            .as_ref()
            .unwrap();
        let world_surface = chunk.heightmaps[HeightmapType::WorldSurfaceWg as usize]
            .as_ref()
            .unwrap();
        for z in 0..16 {
            for x in 0..16 {
                assert_eq!(ocean_floor.first_available(x, z).unwrap(), 320);
                assert_eq!(world_surface.first_available(x, z).unwrap(), 320);
            }
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn benchmark_fill_noise() {
        use std::time::Instant;
        let registry = test_registry();

        let volume = DensityVolume::new([16, 384, 16], [0, -64, 0], [1, 1, 1]).unwrap();
        let sampler = ConstantSampler(1.0);
        let mut context = SamplerContext::sparse();

        let iters = 50;
        let start = Instant::now();
        for _ in 0..iters {
            let mut chunk =
                TerrainChunk::new(ChunkPos::new(0, 0), -64, 384, &registry, Properties).unwrap();
            let mut aquifer = DisabledAquifer::new(TestPicker, BlockStateId(0));
            fill_noise(
                &mut chunk,
                volume,
                &sampler,
                &mut context,
                &mut aquifer,
                BlockStateId(2),
            )
            .unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "benchmark_fill_noise ({} chunks, 384 height): {:?} total, {:.2} ms/chunk",
            iters,
            elapsed,
            elapsed.as_secs_f64() * 1000.0 / iters as f64
        );
    }
}
