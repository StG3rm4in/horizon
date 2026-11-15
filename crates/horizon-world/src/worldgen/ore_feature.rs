//! Ore placement with ordered replacement rules and air-exposure checks.
use super::{
    DecodeError, Object,
    block_feature::{BlockReplacements, FeatureBlockBindings, decode_targets},
    density::identifier,
    feature::{FeatureError, FeatureWorld},
    feature_graph::FeatureExecutionError,
    int_range, number_text,
    placement::PlacementContext,
    required,
};
use crate::heightmap::HeightmapType;
use crate::registry::BlockStateId;
use horizon_core::{geometry::Direction, random::RandomSource};

#[derive(Clone, Debug)]
pub struct ScatteredOreFeature {
    targets: BlockReplacements,
    size: i32,
    discard: f32,
    air: [BlockStateId; 3],
}
impl ScatteredOreFeature {
    pub fn from_json(json: &str, bindings: &FeatureBlockBindings<'_>) -> Result<Self, DecodeError> {
        Self::decode(json, bindings, 0, &mut 65536)
    }
    pub(crate) fn decode(
        json: &str,
        bindings: &FeatureBlockBindings<'_>,
        depth: usize,
        remaining: &mut usize,
    ) -> Result<Self, DecodeError> {
        let object: Object = serde_json::from_str(json)?;
        let kind = identifier(&serde_json::from_str::<String>(
            required(&object, "type")?.get(),
        )?)?;
        if kind != "minecraft:scattered_ore" {
            return Err(DecodeError::Field("scattered ore type"));
        }
        let targets = decode_targets(&object, bindings, depth, remaining)?;
        let size = int_range(required(&object, "size")?, "ore size", 0, 64)?;
        let discard: f32 = number_text(required(&object, "discard_chance_on_air_exposure")?)?
            .parse()
            .map_err(|_| DecodeError::Field("ore air exposure"))?;
        if !(0.0..=1.0).contains(&discard) || discard.is_sign_negative() {
            return Err(DecodeError::Field("ore air exposure"));
        }
        Ok(Self {
            targets,
            size,
            discard,
            air: bindings.air,
        })
    }
    pub fn place<W: FeatureWorld>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        origin: [i32; 3],
    ) -> Result<bool, FeatureError<W::Error>> {
        let count = random
            .next_int_bound(self.size + 1)
            .map_err(FeatureError::Random)?;
        for attempt in 0..count {
            let spread = attempt.min(7) as f32;
            let position = origin.map(|component| {
                let offset = (random.next_float() - random.next_float()) * spread;
                // Widen before adding 0.5 so a float just below a tie cannot
                // round up during addition. Values are bounded by +/-7.
                let offset = (f64::from(offset) + 0.5).floor() as i32;
                component.wrapping_add(offset)
            });
            let current = world.block(position).map_err(FeatureError::World)?;
            for (rule, state) in &self.targets {
                if !rule.test(current, position, random) {
                    continue;
                }
                let skip_air = self.discard <= 0.0
                    || (self.discard < 1.0 && random.next_float() >= self.discard);
                if !skip_air {
                    let mut exposed = false;
                    for direction in Direction::ALL {
                        let delta = direction.normal();
                        let neighbor = std::array::from_fn(|i| position[i].wrapping_add(delta[i]));
                        if self
                            .air
                            .contains(&world.block(neighbor).map_err(FeatureError::World)?)
                        {
                            exposed = true;
                            break;
                        }
                    }
                    if exposed {
                        continue;
                    }
                }
                world
                    .set_block(position, *state, 2)
                    .map_err(FeatureError::World)?;
                break;
            }
        }
        Ok(true)
    }
}

/// Vanilla `minecraft:ore`: an interpolated chain of spheres written straight
/// into chunk sections (BulkSectionAccess), bypassing block update effects.
#[derive(Clone, Debug)]
pub struct OreFeature {
    targets: BlockReplacements,
    size: i32,
    discard: f32,
    air: [BlockStateId; 3],
    /// When no target rule draws from the random source (and there are at
    /// most 32 targets): every Y range the rules test. Which targets match
    /// is then a function of the state and the ranges holding Y.
    heights: Option<Vec<(i32, i32)>>,
}
impl OreFeature {
    pub fn from_json(json: &str, bindings: &FeatureBlockBindings<'_>) -> Result<Self, DecodeError> {
        Self::decode(json, bindings, 0, &mut 65536)
    }
    pub(crate) fn decode(
        json: &str,
        bindings: &FeatureBlockBindings<'_>,
        depth: usize,
        remaining: &mut usize,
    ) -> Result<Self, DecodeError> {
        let object: Object = serde_json::from_str(json)?;
        let kind = identifier(&serde_json::from_str::<String>(
            required(&object, "type")?.get(),
        )?)?;
        if kind != "minecraft:ore" {
            return Err(DecodeError::Field("ore type"));
        }
        let targets = decode_targets(&object, bindings, depth, remaining)?;
        let size = int_range(required(&object, "size")?, "ore size", 0, 64)?;
        let discard: f32 = number_text(required(&object, "discard_chance_on_air_exposure")?)?
            .parse()
            .map_err(|_| DecodeError::Field("ore air exposure"))?;
        if !(0.0..=1.0).contains(&discard) || discard.is_sign_negative() {
            return Err(DecodeError::Field("ore air exposure"));
        }
        let mut heights = Vec::new();
        let pure = targets
            .iter()
            .try_for_each(|(rule, _)| rule.pure_heights(&mut heights));
        let heights =
            (pure.is_some() && targets.len() <= 32 && heights.len() <= 64).then_some(heights);
        Ok(Self {
            targets,
            size,
            discard,
            air: bindings.air,
            heights,
        })
    }
    /// Whether a GPU plan can stand in for this ore's blobs ([`OreFeature::predict`], then
    /// [`OreFeature::place_cells`]): no random draw once a blob's spheres are drawn (no air
    /// exposure chance, pure target rules), so the cells a blob tests follow from the draws
    /// alone; and with no air check each cell is read once and no neighbour is, so the order
    /// the cells are tested in changes nothing.
    pub fn plannable(&self) -> bool {
        self.discard <= 0.0 && self.heights.is_some() && (1..=64).contains(&self.size)
    }
    /// The number of spheres in a blob.
    pub fn size(&self) -> i32 {
        self.size
    }
    pub fn place<W: FeatureWorld + PlacementContext>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        origin: [i32; 3],
    ) -> Result<bool, FeatureExecutionError<W::Error>> {
        let blob = self.blob(random, origin)?;
        if !column_under(world, blob.start, blob.size_xz)? {
            return Ok(false);
        }
        self.do_place(world, random, &blob)
    }
    /// `place`'s draws for one attempt, its column search answered by `world` as it is now. When
    /// a column is found, the blob's spheres (`[x, y, z, r]`, before `do_place` prunes them) are
    /// pushed to `spheres`.
    pub fn predict<W: FeatureWorld + PlacementContext>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        origin: [i32; 3],
        spheres: &mut Vec<[f64; 4]>,
    ) -> Result<OreAttempt, FeatureExecutionError<W::Error>> {
        let blob = self.blob(random, origin)?;
        let found = column_under(world, blob.start, blob.size_xz)?;
        if found {
            self.spheres(random, &blob, spheres);
        }
        Ok(OreAttempt {
            start: blob.start,
            size_xz: blob.size_xz,
            size_y: blob.size_y,
            found,
        })
    }
    /// Whether the column search of a predicted attempt finds a column in `world` as it is now.
    pub fn column_under<W: FeatureWorld + PlacementContext>(
        world: &mut W,
        attempt: &OreAttempt,
    ) -> Result<bool, FeatureExecutionError<W::Error>> {
        column_under(world, attempt.start, attempt.size_xz)
    }
    /// `do_place`'s cell tests and writes for a blob whose cells are given by rows: the box cell
    /// (x, y, z) from the blob's start is in it when bit z of `rows[x * size_y + y]` is set. For a
    /// `plannable` ore only (the cells are not visited in `do_place`'s order), as `place_cell`
    /// with no air check: the first target whose rule matches is written.
    pub fn place_cells<W: FeatureWorld + PlacementContext>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        attempt: &OreAttempt,
        rows: &[u32],
    ) -> Result<bool, FeatureExecutionError<W::Error>> {
        let Some(heights) = self.heights.as_ref().filter(|_| self.plannable()) else {
            return Err(FeatureExecutionError::InvalidState);
        };
        let [x_start, y_start, z_start] = attempt.start;
        let size_y = attempt.size_y.max(1) as usize;
        let mut placed = false;
        let mut matches = TargetMemo::default();
        for (row, &bits) in rows.iter().enumerate() {
            if bits == 0 {
                continue;
            }
            let y = y_start.wrapping_add((row % size_y) as i32);
            if world
                .outside_build_height(y)
                .map_err(FeatureExecutionError::Query)?
            {
                continue;
            }
            let x = x_start.wrapping_add((row / size_y) as i32);
            let ranges = heights.iter().enumerate().fold(0u64, |m, (i, &(lo, hi))| {
                m | u64::from(lo <= y && y <= hi) << i
            });
            placed |= world
                .replace_row([x, y, z_start], bits, &mut |p, state| {
                    let matched = matches.get(state, ranges, || {
                        self.targets
                            .iter()
                            .enumerate()
                            .fold(0u32, |m, (i, (rule, _))| {
                                m | u32::from(rule.test(state, p, random)) << i
                            })
                    });
                    self.targets
                        .iter()
                        .enumerate()
                        .find(|&(i, _)| matched >> i & 1 != 0)
                        .map(|(_, (_, target))| *target)
                })
                .map_err(FeatureExecutionError::Query)?;
        }
        Ok(placed)
    }
    /// The rows `place_cells` takes for a blob, worked out on the CPU from its spheres as `predict`
    /// pushed them (to check other ways of working them out); None when a cell leaves the box.
    pub fn blob_rows(spheres: &[[f64; 4]], attempt: &OreAttempt) -> Option<Vec<u32>> {
        let mut data = spheres.to_vec();
        prune_spheres(&mut data);
        let (sx, sy) = (attempt.size_xz, attempt.size_y);
        let mut rows = vec![0u32; (sx.max(0) * sy.max(0)) as usize];
        let mut squares = SphereSquares::default();
        for sphere in data {
            if sphere[3] < 0.0 {
                continue;
            }
            squares.each_cell(sphere, attempt.start, |[x, y, z]| {
                let b = [0, 1, 2].map(|i| [x, y, z][i].wrapping_sub(attempt.start[i]));
                if b.iter().any(|&v| v < 0) || b[0] >= sx || b[1] >= sy || b[2] >= sx.min(32) {
                    return Err(());
                }
                rows[(b[0] * sy + b[1]) as usize] |= 1 << b[2];
                Ok(())
            })
            .ok()?;
        }
        Some(rows)
    }
    /// One attempt's blob: `place`'s draws (its direction, then its two end heights) and its box.
    fn blob<E>(
        &self,
        random: &mut impl RandomSource,
        origin: [i32; 3],
    ) -> Result<Blob, FeatureExecutionError<E>> {
        let dir = random.next_float() * std::f32::consts::PI;
        let spread_xy = self.size as f32 / 8.0;
        let max_radius = ((self.size as f32 / 16.0 * 2.0 + 1.0) / 2.0).ceil() as i32;
        let spread = f64::from(spread_xy);
        let y0 = f64::from(
            origin[1]
                .wrapping_add(random.next_int_bound(3)?)
                .wrapping_sub(2),
        );
        let y1 = f64::from(
            origin[1]
                .wrapping_add(random.next_int_bound(3)?)
                .wrapping_sub(2),
        );
        let spread_ceil = spread.ceil() as i32;
        let x_start = origin[0].wrapping_sub(spread_ceil).wrapping_sub(max_radius);
        let y_start = origin[1].wrapping_sub(2).wrapping_sub(max_radius);
        let z_start = origin[2].wrapping_sub(spread_ceil).wrapping_sub(max_radius);
        Ok(Blob {
            origin,
            dir,
            spread,
            y: [y0, y1],
            start: [x_start, y_start, z_start],
            size_xz: 2 * (spread_ceil + max_radius),
            size_y: 2 * (2 + max_radius),
        })
    }
    /// `do_place`'s spheres `[x, y, z, r]` before pruning, one `next_double` each.
    fn spheres(&self, random: &mut impl RandomSource, blob: &Blob, data: &mut Vec<[f64; 4]>) {
        // The blob's ends, needed only once a column under it is found
        // (most attempts high in To The Max's tall range find none).
        let [x0, x1, z0, z1] = blob.ends();
        let [y0, y1] = blob.y;
        for i in 0..self.size as usize {
            let step = i as f32 / self.size as f32;
            let step64 = f64::from(step);
            let xx = x0 + step64 * (x1 - x0);
            let yy = y0 + step64 * (y1 - y0);
            let zz = z0 + step64 * (z1 - z0);
            let ss = random.next_double() * f64::from(self.size) / 16.0;
            let sine = horizon_core::geometry::trig::sin(f64::from(std::f32::consts::PI * step));
            let r = (f64::from(sine + 1.0) * ss + 1.0) / 2.0;
            data.push([xx, yy, zz, r]);
        }
    }
    fn do_place<W: FeatureWorld + PlacementContext>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        blob: &Blob,
    ) -> Result<bool, FeatureExecutionError<W::Error>> {
        let mut data = Vec::with_capacity(self.size as usize);
        self.spheres(random, blob, &mut data);
        prune_spheres(&mut data);
        let [x_start, y_start, z_start] = blob.start;
        let (size_xz, size_y) = (blob.size_xz, blob.size_y);
        let tested_len = (size_xz as usize) * (size_y as usize) * (size_xz as usize);
        let mut tested = vec![0u64; tested_len.div_ceil(64)];
        let mut placed = 0;
        let mut matches = TargetMemo::default();
        let mut squares = SphereSquares::default();
        for sphere in data {
            if sphere[3] < 0.0 {
                continue;
            }
            squares.each_cell(sphere, blob.start, |[x, y, z]| {
                if world
                    .outside_build_height(y)
                    .map_err(FeatureExecutionError::Query)?
                {
                    return Ok(());
                }
                let index = i64::from(x.wrapping_sub(x_start))
                    + i64::from(y.wrapping_sub(y_start)) * i64::from(size_xz)
                    + i64::from(z.wrapping_sub(z_start)) * i64::from(size_xz) * i64::from(size_y);
                // The sphere bounds keep indices inside the native BitSet.
                let Some(index) = usize::try_from(index).ok().filter(|&i| i < tested_len) else {
                    return Err(FeatureExecutionError::WorkLimit);
                };
                let (word, bit) = (&mut tested[index / 64], 1u64 << (index % 64));
                if *word & bit != 0 {
                    return Ok(());
                }
                *word |= bit;
                if self.place_cell(world, random, [x, y, z], &mut matches)? {
                    placed += 1;
                }
                Ok(())
            })?;
        }
        Ok(placed > 0)
    }
    /// One cell of a blob, tested once: whether it may be written, then the first target rule
    /// that matches it (and passes the air check) writes it. Whether it was written.
    fn place_cell<W: FeatureWorld + PlacementContext>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        p: [i32; 3],
        matches: &mut TargetMemo,
    ) -> Result<bool, FeatureExecutionError<W::Error>> {
        if !world
            .ensure_can_write(p)
            .map_err(FeatureExecutionError::Query)?
        {
            return Ok(false);
        }
        let Some(state) = world
            .section_block(p)
            .map_err(FeatureExecutionError::Query)?
        else {
            return Ok(false);
        };
        let matched = match &self.heights {
            Some(heights) => {
                let ranges = heights.iter().enumerate().fold(0u64, |m, (i, &(lo, hi))| {
                    m | u64::from(lo <= p[1] && p[1] <= hi) << i
                });
                matches.get(state, ranges, || {
                    self.targets
                        .iter()
                        .enumerate()
                        .fold(0u32, |m, (i, (rule, _))| {
                            m | u32::from(rule.test(state, p, random)) << i
                        })
                })
            }
            None => u32::MAX,
        };
        for (i, (rule, target)) in self.targets.iter().enumerate() {
            if self.heights.is_some() {
                if matched >> i & 1 == 0 {
                    continue;
                }
            } else if !rule.test(state, p, random) {
                continue;
            }
            if !skip_air_check(random, self.discard) {
                let mut exposed = false;
                for direction in Direction::ALL {
                    let n = direction.normal();
                    let q = std::array::from_fn(|i| p[i].wrapping_add(n[i]));
                    let neighbor = world
                        .section_block(q)
                        .map_err(FeatureExecutionError::Query)?
                        .unwrap_or(self.air[0]);
                    if self.air.contains(&neighbor) {
                        exposed = true;
                        break;
                    }
                }
                if exposed {
                    continue;
                }
            }
            world
                .set_section_block(p, *target)
                .map_err(FeatureExecutionError::Query)?;
            return Ok(true);
        }
        Ok(false)
    }
}
/// One attempt of a planned ore ([`OreFeature::predict`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OreAttempt {
    /// The blob's box: its lowest corner, then its X and Z size and its Y size.
    pub start: [i32; 3],
    pub size_xz: i32,
    pub size_y: i32,
    /// The column search found a column: the blob was placed, with `size` spheres.
    pub found: bool,
}
/// One attempt's blob before its spheres are drawn.
struct Blob {
    origin: [i32; 3],
    dir: f32,
    spread: f64,
    y: [f64; 2],
    start: [i32; 3],
    size_xz: i32,
    size_y: i32,
}
impl Blob {
    /// The blob's ends along X and Z: x0, x1, z0, z1.
    fn ends(&self) -> [f64; 4] {
        let (ox, oz) = (f64::from(self.origin[0]), f64::from(self.origin[2]));
        let (sin, cos) = (
            horizon_core::jmath::sin(f64::from(self.dir)),
            horizon_core::jmath::cos(f64::from(self.dir)),
        );
        [
            ox + sin * self.spread,
            ox - sin * self.spread,
            oz + cos * self.spread,
            oz - cos * self.spread,
        ]
    }
}
/// OreFeature.place's search for a column under a blob: an OCEAN_FLOOR_WG height at or above the
/// box's lowest Y, over its columns.
fn column_under<W: FeatureWorld + PlacementContext>(
    world: &mut W,
    [x_start, y_start, z_start]: [i32; 3],
    size_xz: i32,
) -> Result<bool, FeatureExecutionError<W::Error>> {
    if let Some(found) = world
        .any_height_at_least(
            HeightmapType::OceanFloorWg,
            [x_start, z_start],
            size_xz,
            y_start,
        )
        .map_err(FeatureExecutionError::Query)?
    {
        return Ok(found);
    }
    for x in 0..=size_xz {
        for z in 0..=size_xz {
            let (px, pz) = (x_start.wrapping_add(x), z_start.wrapping_add(z));
            let floor = world
                .height(HeightmapType::OceanFloorWg, px, pz)
                .map_err(FeatureExecutionError::Query)?;
            if y_start <= floor {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
/// do_place's pruning: of two spheres, the bigger one removes the other (radius -1) when the
/// difference of their radii exceeds the distance of their centres.
fn prune_spheres(data: &mut [[f64; 4]]) {
    let size = data.len();
    for i1 in 0..size.saturating_sub(1) {
        if data[i1][3] <= 0.0 {
            continue;
        }
        for i2 in i1 + 1..size {
            if data[i2][3] <= 0.0 {
                continue;
            }
            let dx = data[i1][0] - data[i2][0];
            let dy = data[i1][1] - data[i2][1];
            let dz = data[i1][2] - data[i2][2];
            let dr = data[i1][3] - data[i2][3];
            if dr * dr > dx * dx + dy * dy + dz * dz {
                if dr > 0.0 {
                    data[i2][3] = -1.0;
                } else {
                    data[i1][3] = -1.0;
                }
            }
        }
    }
}
/// The cells OreFeature.doPlace tests for one sphere `[x, y, z, r]` of a
/// blob: those of the sphere's box (clamped below to the blob's `start`)
/// whose `xd * xd + yd * yd + zd * zd` is under 1, in vanilla's x, y, z order.
///
/// Along one axis, `d * d` depends on that coordinate alone, so each axis's
/// squares are worked out once per sphere with vanilla's expression, and
/// every sum is bit for bit vanilla's. Along an axis the squares fall and then
/// rise (each step rounds a monotonic function, as does adding a fixed sum),
/// so the cells of a row that pass form one run around the smallest square:
/// it is found from there outwards, and the cells outside the sphere are
/// never visited.
#[derive(Default)]
struct SphereSquares {
    axes: [Vec<f64>; 3],
}
impl SphereSquares {
    fn each_cell<E>(
        &mut self,
        [xx, yy, zz, r]: [f64; 4],
        start: [i32; 3],
        mut cell: impl FnMut([i32; 3]) -> Result<(), E>,
    ) -> Result<(), E> {
        let centre = [xx, yy, zz];
        let mut from = [0; 3];
        let mut lowest = [0; 3];
        for axis in 0..3 {
            let low = ((centre[axis] - r).floor() as i32).max(start[axis]);
            let high = ((centre[axis] + r).floor() as i32).max(low);
            let squares = &mut self.axes[axis];
            squares.clear();
            squares.extend((low..=high).map(|v| {
                let d = (f64::from(v) + 0.5 - centre[axis]) / r;
                d * d
            }));
            from[axis] = low;
            lowest[axis] = (0..squares.len())
                .min_by(|&a, &b| squares[a].total_cmp(&squares[b]))
                .unwrap_or(0);
        }
        let [xs, ys, zs] = &self.axes;
        // Vanilla's `xd * xd >= 1.0` is `0.0 + xd * xd >= 1.0`: a square is
        // never negative zero, and adding zero to it rounds nothing.
        let Some((x_low, x_high)) = pass_run(xs, lowest[0], 0.0) else {
            return Ok(());
        };
        for i in x_low..=x_high {
            let xd2 = xs[i];
            let Some((y_low, y_high)) = pass_run(ys, lowest[1], xd2) else {
                continue;
            };
            for j in y_low..=y_high {
                let xy = xd2 + ys[j];
                let Some((z_low, z_high)) = pass_run(zs, lowest[2], xy) else {
                    continue;
                };
                for k in z_low..=z_high {
                    cell([from[0] + i as i32, from[1] + j as i32, from[2] + k as i32])?;
                }
            }
        }
        Ok(())
    }
}
/// The run of `squares` around index `lowest` (a smallest one) where vanilla's
/// `base + square >= 1.0` fails, or None when even the smallest reaches 1.
/// The sums are never NaN (a sphere's radius is at least 0.5), so failing
/// that test is being under 1.
fn pass_run(squares: &[f64], lowest: usize, base: f64) -> Option<(usize, usize)> {
    let passes = |i: usize| base + squares[i] < 1.0;
    if !passes(lowest) {
        return None;
    }
    let (mut low, mut high) = (lowest, lowest);
    while low > 0 && passes(low - 1) {
        low -= 1;
    }
    while high + 1 < squares.len() && passes(high + 1) {
        high += 1;
    }
    Some((low, high))
}
/// Recent (state, Y ranges) -> matching targets of a pure-rule ore.
#[derive(Default)]
struct TargetMemo {
    entries: [(u32, u64, u32); 8],
    len: usize,
    next: usize,
}
impl TargetMemo {
    fn get(&mut self, state: BlockStateId, ranges: u64, compute: impl FnOnce() -> u32) -> u32 {
        let key = (state.0, ranges);
        if let Some(e) = self.entries[..self.len].iter().find(|e| (e.0, e.1) == key) {
            return e.2;
        }
        let value = compute();
        self.entries[self.next] = (key.0, key.1, value);
        self.next = (self.next + 1) % self.entries.len();
        self.len = (self.len + 1).min(self.entries.len());
        value
    }
}
fn skip_air_check(random: &mut impl RandomSource, discard: f32) -> bool {
    if discard <= 0.0 {
        true
    } else if discard >= 1.0 {
        false
    } else {
        random.next_float() >= discard
    }
}

#[cfg(test)]
mod tests {
    use super::SphereSquares;

    /// OreFeature.doPlace's loops over one sphere's whole box, as vanilla
    /// writes them: the cells it goes on to test.
    fn box_scan([xx, yy, zz, r]: [f64; 4], [x_start, y_start, z_start]: [i32; 3]) -> Vec<[i32; 3]> {
        let x_min = ((xx - r).floor() as i32).max(x_start);
        let y_min = ((yy - r).floor() as i32).max(y_start);
        let z_min = ((zz - r).floor() as i32).max(z_start);
        let x_max = ((xx + r).floor() as i32).max(x_min);
        let y_max = ((yy + r).floor() as i32).max(y_min);
        let z_max = ((zz + r).floor() as i32).max(z_min);
        let mut cells = Vec::new();
        for x in x_min..=x_max {
            let xd = (f64::from(x) + 0.5 - xx) / r;
            if xd * xd >= 1.0 {
                continue;
            }
            for y in y_min..=y_max {
                let yd = (f64::from(y) + 0.5 - yy) / r;
                if xd * xd + yd * yd >= 1.0 {
                    continue;
                }
                for z in z_min..=z_max {
                    let zd = (f64::from(z) + 0.5 - zz) / r;
                    if xd * xd + yd * yd + zd * zd >= 1.0 {
                        continue;
                    }
                    cells.push([x, y, z]);
                }
            }
        }
        cells
    }

    #[test]
    fn a_sphere_visits_exactly_the_cells_vanillas_box_scan_tests() {
        // SplitMix64: any fixed spread of spheres will do.
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut unit = move || {
            seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut squares = SphereSquares::default();
        let mut visited = 0;
        for _ in 0..20_000 {
            // Centres and radii on whole and half blocks put cells exactly on
            // the sphere's edge, where a sum is exactly 1.
            let mut sphere = [0.0; 4];
            let r = 0.5 + unit() * 4.0;
            sphere[3] = if unit() < 0.3 {
                (r * 2.0).round() / 2.0
            } else {
                r
            };
            let mut start = [0; 3];
            for axis in 0..3 {
                let c = (unit() - 0.5) * 80_000.0;
                sphere[axis] = match (unit() * 3.0) as u32 {
                    0 => c.floor(),
                    1 => c.floor() + 0.5,
                    _ => c,
                };
                // Sometimes the blob's start cuts the sphere's box.
                start[axis] = (sphere[axis] - sphere[3]).floor() as i32 + (unit() * 5.0) as i32 - 1;
            }
            let mut cells = Vec::new();
            squares
                .each_cell::<()>(sphere, start, |p| {
                    cells.push(p);
                    Ok(())
                })
                .unwrap();
            assert_eq!(
                cells,
                box_scan(sphere, start),
                "sphere {sphere:?}, start {start:?}"
            );
            visited += cells.len();
        }
        assert!(visited > 100_000, "only {visited} cells");
    }
}
