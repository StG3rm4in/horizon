//! Explosion exposure rays on the GPU (`HORIZON_GPU_TNT=1`, or `test` to check each answer against
//! the CPU's rays): `ServerExplosion.getSeenPercent` for every entity box in a blast's reach, all
//! at once before its entity walk (`explosion_hurt_entities`), each box traced once however many
//! entities share it.
//!
//! The walk asks the entities' exposures one by one, and an entity's hurt or push may run hooks
//! between two asks; the rays themselves read only blocks and their collision shapes. So each of
//! the GPU's answers is kept with what its rays read: the stamp of the box from the entity's box
//! to the centre (`World::region_stamp`: the write counts of its sections and the columns
//! missing), taken with the cells. An ask takes the answer only while that stamp is the same,
//! as the CPU's exposure memo does, and the CPU traces the box otherwise.
//! The cube goes to the GPU as the collision shape of each cell, and a blast goes there only when
//! every cell's shape is the one the CPU would take for any position and any box of the blast:
//! a fixed shape (`cached_shape`), the shape a contextual state has for the blast's context
//! (`context_collision_shape`), or a liquid's, which collides as nothing for that context (no
//! strider: `SceneCollisions` without a subject; liquid_collision.rs), as the CPU's own answer
//! for the liquid's first cell must confirm. A dynamic shape (moving piston, shulker box),
//! scaffolding or powder snow in the cube leaves the blast to the CPU. Random draws are not
//! involved: the rays draw nothing.
//!
//! When it pays: a blast goes to the GPU only when its cost there (a round trip, the cube's cells
//! the CPU reads for it, its rays) is lower than its rays' on the CPU, from costs measured on both
//! sides ([`Cost`]; `HORIZON_GPU_TNT_COST` sets them). A TNT stack is one box, 27 rays: it stays on
//! the CPU, whose exposure memo serves it; a blast among many dropped items or mobs comes here.
//! The test mode sends every blast and logs both sides' time for each, which is how they are
//! measured. The kernel, a set of buffers and its first launch are readied on their own thread
//! from the server's start ([`start`]), so that no blast waits for them.
use super::{
    Game,
    entity::{Entity, with_entity},
    explosion::exposure_starts,
};
use horizon_client::collisions::SceneCollisions;
use horizon_client::world::RegionStamp;
use horizon_core::{geometry::Vec3, position::BlockPos};
use horizon_world::registry::BlockStateId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Once, OnceLock};
use std::time::Instant;

/// Cells of a blast's cube at most (48 a side): past that, the CPU.
const CUBE_CELLS: usize = 48 * 48 * 48;
/// Rays of the buffers readied before the first blast: some 300 distinct boxes.
const WARM_RAYS: usize = 8192;

/// The GPU's answers for the blast going off now.
struct Answers {
    center: [u64; 3],
    /// A box's bits (minimum, maximum), the stamp of its rays' cells when they were read, and its
    /// exposure; a box not here is the CPU's.
    seen: HashMap<[u64; 6], (RegionStamp, f32)>,
}
thread_local! {
    static ANSWERS: RefCell<Option<Answers>> = const { RefCell::new(None) };
}

/// Keeps a blast's answers for as long as it lives, then puts back those of the blast around it,
/// if any (a hook may set off a blast inside a walk).
pub(super) struct GpuExposureScope(Option<Answers>);
impl Drop for GpuExposureScope {
    fn drop(&mut self) {
        let outer = self.0.take();
        ANSWERS.with(|a| *a.borrow_mut() = outer);
    }
}

#[derive(Default)]
struct Stats {
    blasts: AtomicU64,
    boxes: AtomicU64,
    rays: AtomicU64,
    ns: AtomicU64,
    left: AtomicU64,
    answered: AtomicU64,
    stale: AtomicU64,
    errors: AtomicU64,
    compared: AtomicU64,
    mismatches: AtomicU64,
}
static STATS: OnceLock<Stats> = OnceLock::new();
fn stats() -> &'static Stats {
    STATS.get_or_init(Stats::default)
}

/// HORIZON_GPU_TNT: 0 off, 1 on, 2 test.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("HORIZON_GPU_TNT").as_deref() {
        Ok("1") => 1,
        Ok("test") => 2,
        _ => 0,
    })
}
/// What a blast costs, in nanoseconds: on the GPU, a round trip, each cell of its cube (which the
/// CPU reads and sends) and each ray; on the CPU, each ray. Fitted on 23/11 to the test mode's
/// lines of 872 blasts of the TNT fixtures on the server (i5-10600K at 4.9 GHz, RTX 3080), where
/// this rule picks the side that was faster for 839 of them: a TNT stack (one box) stays on the
/// CPU, a blast among some 5 to 8 distinct boxes and more goes to the GPU.
/// `HORIZON_GPU_TNT_COST="TRIP,CELL,GPU_RAY,CPU_RAY"` sets them for another machine.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cost {
    round_trip: f64,
    cell: f64,
    gpu_ray: f64,
    cpu_ray: f64,
}
const MEASURED: Cost = Cost {
    round_trip: 48_500.0,
    cell: 27.0,
    gpu_ray: 20.0,
    cpu_ray: 500.0,
};
impl Cost {
    /// Four numbers, none negative, or None.
    fn parse(v: &str) -> Option<Self> {
        let parts: Vec<f64> = v.split(',').map(|p| p.trim().parse().ok()).collect::<Option<_>>()?;
        match parts[..] {
            [round_trip, cell, gpu_ray, cpu_ray]
                if parts.iter().all(|p| p.is_finite() && *p >= 0.0) =>
            {
                Some(Self {
                    round_trip,
                    cell,
                    gpu_ray,
                    cpu_ray,
                })
            }
            _ => None,
        }
    }
    /// Whether `rays` rays over a cube of `cells` cells are traced sooner on the GPU.
    fn gpu_sooner(self, rays: usize, cells: usize) -> bool {
        let (rays, cells) = (rays as f64, cells as f64);
        self.round_trip + self.cell * cells + self.gpu_ray * rays < self.cpu_ray * rays
    }
}
fn cost() -> Cost {
    static COST: OnceLock<Cost> = OnceLock::new();
    *COST.get_or_init(|| match std::env::var("HORIZON_GPU_TNT_COST") {
        Ok(v) => Cost::parse(&v).unwrap_or_else(|| {
            log::warn!("GPU TNT: HORIZON_GPU_TNT_COST={v:?} is not four numbers; measured costs");
            MEASURED
        }),
        Err(_) => MEASURED,
    })
}
/// Whether a blast of `rays` rays over a cube of `cells` cells goes to the GPU: when it is sooner
/// there, and always in test mode, which compares every blast.
fn worth_the_gpu(rays: usize, cells: usize) -> bool {
    rays > 0 && (mode() == 2 || cost().gpu_sooner(rays, cells))
}
/// Readies the GPU's kernel, buffers and first launch on their own thread when HORIZON_GPU_TNT
/// asks for them: the dedicated server calls it as it prepares its level, before any blast.
pub(super) fn start() {
    let _ = gpu();
}
/// The GPU's kernel, compiled on a thread of its own from the server's start ([`start`]) or the
/// first blast that asks, then readied ([`horizon_gpu::ExposureGpu::warm_up`]), so that no tick
/// waits for NVRTC or a first launch: the blasts until it is ready stay on the CPU, as do all of
/// them without a GPU, without CUDA or after a CUDA error.
fn gpu() -> Option<&'static horizon_gpu::ExposureGpu> {
    static GPU: OnceLock<Option<horizon_gpu::ExposureGpu>> = OnceLock::new();
    static START: Once = Once::new();
    if mode() == 0 {
        return None;
    }
    START.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("gpu-tnt".into())
            .spawn(|| {
                let gpu = horizon_gpu::ExposureGpu::new()
                    .and_then(|gpu| gpu.warm_up(WARM_RAYS, CUBE_CELLS).map(|()| gpu))
                    .map_err(|e| log::warn!("GPU TNT: no exposure rays on the GPU: {e}"))
                    .ok();
                let _ = GPU.set(gpu);
            });
        if spawned.is_err() {
            let _ = GPU.set(None);
        }
    });
    GPU.get()?.as_ref()
}

/// A box's bits: its minimum, then its maximum.
fn box_key(min: [f64; 3], max: [f64; 3]) -> [u64; 6] {
    std::array::from_fn(|i| (if i < 3 { min[i] } else { max[i - 3] }).to_bits())
}
/// Every cell a ray from the box to `center` may visit: its sample points lie up to half a step
/// past the box's corner, and the traversal's ends 1e-7 past the ray's, so a block to spare.
fn ray_cells(center: [f64; 3], min: [f64; 3], max: [f64; 3]) -> ([i32; 3], [i32; 3]) {
    (
        std::array::from_fn(|i| (min[i].min(center[i]).floor() as i32).saturating_sub(1)),
        std::array::from_fn(|i| (max[i].max(center[i]).floor() as i32).saturating_add(1)),
    )
}

/// The GPU's answer for a box, if the blast going off at `center` traced it and none of the cells
/// its rays may cross was written since. In test mode, the CPU's answer, compared with the GPU's.
pub(super) fn gpu_seen_percent(
    game: &Game<'_>,
    center: [f64; 3],
    min: [f64; 3],
    max: [f64; 3],
) -> Option<f32> {
    let at = center.map(f64::to_bits);
    let (stamp, seen) = ANSWERS.with(|a| {
        let a = a.borrow();
        let a = a.as_ref().filter(|a| a.center == at)?;
        a.seen.get(&box_key(min, max)).copied()
    })?;
    let (lo, hi) = ray_cells(center, min, max);
    if game.world.region_stamp(lo, hi) != Some(stamp) {
        stats().stale.fetch_add(1, Relaxed);
        return None;
    }
    stats().answered.fetch_add(1, Relaxed);
    if mode() == 2 {
        let cpu = game.seen_percent_cpu(center, min, max);
        let s = stats();
        let n = s.compared.fetch_add(1, Relaxed) + 1;
        if cpu.to_bits() != seen.to_bits() {
            let m = s.mismatches.fetch_add(1, Relaxed) + 1;
            if m <= 16 {
                log::warn!(
                    "GPU TNT test: centre {center:?}, box {min:?}..{max:?}: GPU {seen}, CPU {cpu}"
                );
            }
        }
        if n.is_power_of_two() {
            log::info!(
                "GPU TNT test: {n} exposures compared, {} mismatches",
                s.mismatches.load(Relaxed)
            );
        }
        return Some(cpu);
    }
    Some(seen)
}

impl Game<'_> {
    /// Traces on the GPU, at once, the exposure of every entity box within the reach of the blast
    /// at `center` (the player's and those of the entities in the storages; any other box asks
    /// the CPU), and keeps the answers until the returned scope drops. None, and every exposure
    /// on the CPU: no GPU, a blast the CPU traces sooner, a cube too large, a cell whose shape the
    /// GPU cannot take.
    pub(super) fn gpu_exposures(
        &self,
        center: [f64; 3],
        diameter: f64,
    ) -> Option<GpuExposureScope> {
        let gpu = gpu()?;
        let started = Instant::now();
        let reach = |feet: [f64; 3]| {
            let d = [0, 1, 2].map(|i| feet[i] - center[i]);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= diameter * diameter
        };
        // The boxes, each once, in a fixed order.
        let mut boxes: Vec<([f64; 3], [f64; 3])> = Vec::new();
        let mut keys = std::collections::HashSet::new();
        let mut add = |min: [f64; 3], max: [f64; 3]| {
            if keys.insert(box_key(min, max)) {
                boxes.push((min, max));
            }
        };
        if let Some(p) = self.player.as_ref() {
            let feet = [p.position.x, p.position.y, p.position.z];
            if reach(feet) {
                // As `explosion_hurt_player` builds it.
                let w = f64::from(p.dimensions.width) / 2.0;
                let h = f64::from(p.dimensions.height);
                add(
                    [feet[0] - w, feet[1], feet[2] - w],
                    [feet[0] + w, feet[1] + h, feet[2] + w],
                );
            }
        }
        let mut cursor = self.entity_id_cursor();
        while let Some((_, handle)) = self.next_entity_id(&mut cursor) {
            let (feet, b) = with_entity!(self, handle, e => (e.position(), e.explosion_box()));
            if !reach([feet.x, feet.y, feet.z]) {
                continue;
            }
            let (lo, hi) = (b.min_position(), b.max_position());
            add([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]);
        }
        // The rays, box after box, from the same sample points as the CPU's.
        let to = Vec3::new(center[0], center[1], center[2]);
        let mut rays: Vec<[f64; 6]> = Vec::new();
        let mut spans = Vec::with_capacity(boxes.len());
        for &(min, max) in &boxes {
            let first = rays.len();
            let sampled = exposure_starts(min, max, |from| {
                rays.push([from.x, from.y, from.z, to.x, to.y, to.z]);
            });
            spans.push((first, rays.len(), sampled));
        }
        // The cube: every cell a ray visits, with a block to spare for the traversal's ends.
        let mut lo = center.map(|v| v.floor() as i32);
        let mut hi = lo;
        for &(min, max) in &boxes {
            for i in 0..3 {
                lo[i] = lo[i].min(min[i].floor() as i32);
                hi[i] = hi[i].max(max[i].floor() as i32);
            }
        }
        let lo = lo.map(|v| v.saturating_sub(1));
        let hi = hi.map(|v| v.saturating_add(1));
        let dims: [i32; 3] =
            std::array::from_fn(|i| hi[i].saturating_sub(lo[i]).saturating_add(1));
        let volume = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))?;
        if volume > CUBE_CELLS {
            stats().left.fetch_add(1, Relaxed);
            return None;
        }
        if !worth_the_gpu(rays.len(), volume) {
            return None;
        }
        let mut shapes = horizon_gpu::RayShapes::default();
        let mut by_state: HashMap<BlockStateId, u16> = HashMap::new();
        let mut cells = Vec::with_capacity(volume);
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let state = self.world.block(x, y, z);
                    let id = match by_state.get(&state) {
                        Some(&id) => id,
                        None => {
                            let Some(id) = self.gpu_ray_shape(state, [x, y, z], &mut shapes) else {
                                stats().left.fetch_add(1, Relaxed);
                                return None;
                            };
                            by_state.insert(state, id);
                            id
                        }
                    };
                    cells.push(id);
                }
            }
        }
        let cube = horizon_gpu::RayCube {
            lo,
            dims,
            cells: &cells,
        };
        let answers = match gpu.trace(&rays, &cube, &shapes) {
            Ok(answers) => answers,
            Err(e) => {
                stats().errors.fetch_add(1, Relaxed);
                log::error!("GPU TNT: exposure rays: {e}; blasts on the CPU from now on");
                return None;
            }
        };
        // The stamps are taken with the cells, before any hook may write.
        let mut seen = HashMap::with_capacity(boxes.len());
        for (&(min, max), &(first, end, sampled)) in boxes.iter().zip(&spans) {
            let (from, to) = ray_cells(center, min, max);
            let Some(stamp) = self.world.region_stamp(from, to) else {
                continue;
            };
            let flags = &answers[first..end];
            if !sampled {
                // A negative extent: no ray, and the blast does not see the box.
                seen.insert(box_key(min, max), (stamp, 0.0));
            } else if !flags.contains(&horizon_gpu::RAY_OUTSIDE) {
                let hits = flags.iter().filter(|&&f| f == horizon_gpu::RAY_MISS).count();
                seen.insert(box_key(min, max), (stamp, hits as f32 / flags.len() as f32));
            }
        }
        let gpu_ns = started.elapsed().as_nanos() as u64;
        if mode() == 2 {
            // The CPU's rays for the same boxes, timed: the two sides' costs (`Cost`) come from
            // these lines.
            let cpu_started = Instant::now();
            for &(min, max) in &boxes {
                std::hint::black_box(self.seen_percent_cpu(center, min, max));
            }
            log::info!(
                "GPU TNT test blast: {} boxes, {} rays, {volume} cells: GPU {:.1} us, CPU {:.1} us",
                boxes.len(),
                rays.len(),
                gpu_ns as f64 / 1000.0,
                cpu_started.elapsed().as_nanos() as f64 / 1000.0,
            );
        }
        let s = stats();
        let n = s.blasts.fetch_add(1, Relaxed) + 1;
        s.boxes.fetch_add(boxes.len() as u64, Relaxed);
        s.rays.fetch_add(rays.len() as u64, Relaxed);
        s.ns.fetch_add(gpu_ns, Relaxed);
        if n.is_power_of_two() {
            log::info!(
                "GPU TNT: {n} blasts on the GPU, {} boxes, {} rays, {} us a blast; {} left to the \
                 CPU; {} answers taken, {} stale; {} errors",
                s.boxes.load(Relaxed),
                s.rays.load(Relaxed),
                s.ns.load(Relaxed) / n / 1000,
                s.left.load(Relaxed),
                s.answered.load(Relaxed),
                s.stale.load(Relaxed),
                s.errors.load(Relaxed),
            );
        }
        let answers = Answers {
            center: center.map(f64::to_bits),
            seen,
        };
        Some(GpuExposureScope(
            ANSWERS.with(|a| a.replace(Some(answers))),
        ))
    }
    /// The index in `shapes` of the collision shape the CPU's rays take for `state` (at `pos`, a
    /// cell where it stands), the same for every cell and box of the blast; None when it may
    /// depend on either, or on the world.
    fn gpu_ray_shape(
        &self,
        state: BlockStateId,
        pos: [i32; 3],
        shapes: &mut horizon_gpu::RayShapes,
    ) -> Option<u16> {
        let binding = self.collision;
        let cached = binding.cached_shapes();
        // `collision_shape_with_context`: no contextual rule, the fixed shape (none: dynamic).
        if !binding.has_context_rule(state) {
            return shapes.push(cached.cached_shape(state).ok()??);
        }
        // A contextual state: the shape its context kind declares, as for the CPU's rays...
        let context = super::explosion::exposure_context(0.0);
        if let Some(shape) = cached.context_collision_shape(state, context.entity_kind()) {
            return shapes.push(shape);
        }
        // ...else a liquid, nothing for this context, which the CPU's own answer must say.
        let description = binding.registry().state(state)?;
        horizon_world::liquid_collision::LiquidCollision::from_state(description).ok()??;
        if context.always_collide_with_fluid() {
            return None;
        }
        let mut scene = SceneCollisions {
            world: &self.world,
            binding,
            subject: None,
        };
        let at = BlockPos::new(pos[0], pos[1], pos[2]);
        let shape = binding
            .collision_shape_with_context(&mut scene, state, at, &context)
            .ok()?;
        if !shape.is_empty() {
            return None;
        }
        shapes.push(&shape)
    }
}

#[cfg(test)]
mod cost_tests {
    use super::{Cost, MEASURED};

    /// The costs' setting takes four numbers, none negative, and nothing else.
    #[test]
    fn the_costs_parse_four_numbers() {
        assert_eq!(
            Cost::parse("1, 2.5,3,4"),
            Some(Cost {
                round_trip: 1.0,
                cell: 2.5,
                gpu_ray: 3.0,
                cpu_ray: 4.0
            })
        );
        for bad in ["", "1,2,3", "1,2,3,4,5", "1,2,x,4", "1,-2,3,4", "1,inf,3,4"] {
            assert_eq!(Cost::parse(bad), None, "{bad:?}");
        }
    }

    /// With the measured costs a TNT stack (one box, 27 rays, a small cube) stays on the CPU and
    /// a blast among many dropped items (some 50 boxes) goes to the GPU; the GPU's side grows with
    /// the cube, so the same rays over a much larger cube may not.
    #[test]
    fn a_stack_stays_on_the_cpu_and_many_boxes_go_to_the_gpu() {
        assert!(!MEASURED.gpu_sooner(27, 8 * 8 * 8));
        assert!(MEASURED.gpu_sooner(50 * 27, 18 * 18 * 18));
        let free = Cost {
            round_trip: 0.0,
            cell: 0.0,
            gpu_ray: 0.0,
            cpu_ray: 1.0,
        };
        assert!(free.gpu_sooner(1, 1));
        assert!(!free.gpu_sooner(0, 1));
        let costly_cells = Cost {
            cell: 1_000.0,
            ..MEASURED
        };
        assert!(!costly_cells.gpu_sooner(50 * 27, 48 * 48 * 48));
    }
}
