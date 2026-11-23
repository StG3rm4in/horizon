//! Explosion exposure rays on the GPU (`HORIZON_GPU_TNT`): the rays of
//! `ServerExplosion.getSeenPercent` for many entity boxes of one blast at once, on the tick
//! thread, which waits for them. The game builds the blast's cube of cells and their collision
//! shapes, the rays and the guards (horizon-client, `game/entities/explosion_gpu.rs`); this module
//! only traces (`kex_rays`, exposure.cu) and says, per ray, whether it met no shape.
use super::{Context, context, load_module, priority_stream};
use crate::cuda::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice,
    PushKernelArg,
};
use horizon_core::geometry::{Axis, VoxelShape};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

const SOURCE: &str = include_str!("exposure.cu");

/// What `kex_rays` answers for a ray.
pub const RAY_HIT: u8 = 0;
pub const RAY_MISS: u8 = 1;
/// The ray left the cube: the box it belongs to is traced on the CPU.
pub const RAY_OUTSIDE: u8 = 2;

/// Cells of a shape's grid and boxes of a shape at most (vanilla's finest, the 16 x 16 x 16
/// grids of some models, have 4,096 cells): beyond, the blast stays on the CPU.
const SHAPE_CELLS: usize = 4096;
const SHAPE_BOXES: usize = 1024;

/// The collision shapes of a blast's cells as `kex_rays` reads them (`VoxelShape::clip`): per
/// shape eight words (whether its grid is a cube, which `find_index` scales instead of
/// searching; its three sizes; where its boundaries, its occupancy bits and its boxes start; its
/// box count), the boundaries and the boxes in `values`, the occupancy in `bits`.
#[derive(Debug, Default)]
pub struct RayShapes {
    header: Vec<u32>,
    values: Vec<f64>,
    bits: Vec<u64>,
}
impl RayShapes {
    /// Adds `shape` and gives its index; None past 65,536 shapes, or for a shape finer than the
    /// GPU's tables take.
    pub fn push(&mut self, shape: &VoxelShape) -> Option<u16> {
        let index = u16::try_from(self.header.len() / 8).ok()?;
        let sizes = shape.sizes();
        let cells = sizes.iter().try_fold(1usize, |n, &s| n.checked_mul(s))?;
        let boxes = shape.boxes();
        if cells > SHAPE_CELLS || boxes.len() > SHAPE_BOXES {
            return None;
        }
        let values = u32::try_from(self.values.len()).ok()?;
        let bits = u32::try_from(self.bits.len()).ok()?;
        for axis in Axis::ALL {
            let coordinates = shape.coordinates(axis);
            if coordinates.len() != sizes[axis as usize] + 1 {
                return None;
            }
            self.values.extend_from_slice(coordinates);
        }
        let first_box = u32::try_from(self.values.len()).ok()?;
        for b in boxes {
            self.values.extend(Axis::ALL.map(|a| b.min(a)));
            self.values.extend(Axis::ALL.map(|a| b.max(a)));
        }
        let mut words = vec![0u64; cells.div_ceil(64).max(1)];
        for x in 0..sizes[0] {
            for y in 0..sizes[1] {
                for z in 0..sizes[2] {
                    if shape.is_full(x as i32, y as i32, z as i32) {
                        let i = (x * sizes[1] + y) * sizes[2] + z;
                        words[i / 64] |= 1 << (i % 64);
                    }
                }
            }
        }
        self.bits.extend(words);
        self.header.extend([
            u32::from(shape.is_cube()),
            sizes[0] as u32,
            sizes[1] as u32,
            sizes[2] as u32,
            values,
            bits,
            first_box,
            boxes.len() as u32,
        ]);
        Some(index)
    }
    /// Shapes added.
    pub fn len(&self) -> usize {
        self.header.len() / 8
    }
    pub fn is_empty(&self) -> bool {
        self.header.is_empty()
    }
}

/// The cells a blast's rays may cross: from `lo`, `dims` cells a side, the shape index of cell
/// (x, y, z) at `(x * dims[1] + y) * dims[2] + z`.
#[derive(Debug)]
pub struct RayCube<'a> {
    pub lo: [i32; 3],
    pub dims: [i32; 3],
    pub cells: &'a [u16],
}

/// One blast's buffers, grown to the largest blast so far.
struct Resources {
    st: Arc<CudaStream>,
    rays: CudaSlice<f64>,
    cells: CudaSlice<u16>,
    header: CudaSlice<u32>,
    values: CudaSlice<f64>,
    bits: CudaSlice<u64>,
    out: CudaSlice<u8>,
    h_out: PinnedHostSlice<u8>,
}

pub struct ExposureGpu {
    ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    k_rays: CudaFunction,
    pool: Mutex<Vec<Resources>>,
    /// After a CUDA error every blast stays on the CPU for the session.
    broken: AtomicBool,
}
impl std::fmt::Debug for ExposureGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ExposureGpu")
    }
}

/// `slot` with room for `n` elements (at least one), replaced by a larger one if it has not.
fn room<T: crate::cuda::driver::DeviceRepr + crate::cuda::driver::ValidAsZeroBits>(
    st: &Arc<CudaStream>,
    slot: &mut CudaSlice<T>,
    n: usize,
) -> Result<(), String> {
    if slot.len() < n.max(1) {
        *slot = st
            .alloc_zeros::<T>(n.max(1).next_power_of_two())
            .ctx("exposure buffer")?;
    }
    Ok(())
}

impl ExposureGpu {
    /// The exposure kernel, compiled (or read from the PTX cache) for the process's device.
    pub fn new() -> Result<Self, String> {
        let ctx = context()?;
        let (major, minor) = ctx.compute_capability().ctx("GPU compute capability")?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        let module = load_module(&ctx, SOURCE, arch).ctx("exposure module")?;
        let k_rays = module.load_function("kex_rays").ctx("kex_rays")?;
        Ok(Self {
            ctx,
            _module: module,
            k_rays,
            pool: Mutex::new(Vec::new()),
            broken: AtomicBool::new(false),
        })
    }
    fn resource(&self) -> Result<Resources, String> {
        if let Some(r) = self.pool.lock().unwrap().pop() {
            return Ok(r);
        }
        // The tick thread waits for the answer: the critical path's priority.
        let st = priority_stream(&self.ctx)?;
        Ok(Resources {
            rays: st.alloc_zeros::<f64>(1).ctx("rays")?,
            cells: st.alloc_zeros::<u16>(1).ctx("cells")?,
            header: st.alloc_zeros::<u32>(1).ctx("shapes")?,
            values: st.alloc_zeros::<f64>(1).ctx("shapes")?,
            bits: st.alloc_zeros::<u64>(1).ctx("shapes")?,
            out: st.alloc_zeros::<u8>(1).ctx("answers")?,
            h_out: unsafe { self.ctx.alloc_pinned_with_flags::<u8>(1, 0) }.ctx("host buffer")?,
            st,
        })
    }
    /// Readies what the first blast would otherwise make on the thread that waits for it: a set of
    /// buffers for `rays` rays and `cells` cells (the stream, the device buffers, the pinned answers)
    /// and the kernel itself, which its first launch loads (one ray through one empty cell). Call it
    /// on the thread that made the kernel, before any blast.
    pub fn warm_up(&self, rays: usize, cells: usize) -> Result<(), String> {
        let mut res = self.resource()?;
        let st = res.st.clone();
        room(&st, &mut res.rays, rays.saturating_mul(6))?;
        room(&st, &mut res.cells, cells)?;
        room(&st, &mut res.out, rays)?;
        if res.h_out.len() < rays {
            res.h_out = unsafe {
                self.ctx
                    .alloc_pinned_with_flags::<u8>(rays.max(1).next_power_of_two(), 0)
            }
            .ctx("host buffer")?;
        }
        self.pool.lock().unwrap().push(res);
        let mut shapes = RayShapes::default();
        let empty = shapes
            .push(&VoxelShape::empty())
            .ok_or_else(|| "the empty shape".to_string())?;
        let cube = RayCube {
            lo: [0, 0, 0],
            dims: [1, 1, 1],
            cells: &[empty],
        };
        let answer = self.trace(&[[0.25, 0.25, 0.25, 0.75, 0.75, 0.75]], &cube, &shapes)?;
        if answer != [RAY_MISS] {
            return Err(format!("a ray through an empty cell answered {answer:?}"));
        }
        Ok(())
    }
    /// For each ray (from x, y, z, to x, y, z), [`RAY_MISS`], [`RAY_HIT`] or [`RAY_OUTSIDE`], in
    /// order. Blocking.
    pub fn trace(
        &self,
        rays: &[[f64; 6]],
        cube: &RayCube<'_>,
        shapes: &RayShapes,
    ) -> Result<Vec<u8>, String> {
        let n = rays.len();
        let volume = cube
            .dims
            .iter()
            .map(|&d| d.max(0) as usize)
            .product::<usize>();
        if self.broken.load(Relaxed) {
            return Err("off after an earlier CUDA error".into());
        }
        if n == 0 || n > i32::MAX as usize || cube.cells.len() != volume || shapes.is_empty() {
            return Err("empty or inconsistent blast".into());
        }
        let mut res = self.resource()?;
        let result = (|| -> Result<Vec<u8>, String> {
            let st = res.st.clone();
            let flat: Vec<f64> = rays.iter().flatten().copied().collect();
            room(&st, &mut res.rays, flat.len())?;
            room(&st, &mut res.cells, volume)?;
            room(&st, &mut res.header, shapes.header.len())?;
            room(&st, &mut res.values, shapes.values.len())?;
            room(&st, &mut res.bits, shapes.bits.len())?;
            room(&st, &mut res.out, n)?;
            if res.h_out.len() < n {
                res.h_out = unsafe {
                    self.ctx
                        .alloc_pinned_with_flags::<u8>(n.next_power_of_two(), 0)
                }
                .ctx("host buffer")?;
            }
            st.memcpy_htod(&flat[..], &mut res.rays.slice_mut(0..flat.len()))
                .ctx("ray upload")?;
            st.memcpy_htod(cube.cells, &mut res.cells.slice_mut(0..volume))
                .ctx("cell upload")?;
            let header = &shapes.header[..];
            st.memcpy_htod(header, &mut res.header.slice_mut(0..header.len()))
                .ctx("shape upload")?;
            if !shapes.values.is_empty() {
                let values = &shapes.values[..];
                st.memcpy_htod(values, &mut res.values.slice_mut(0..values.len()))
                    .ctx("shape upload")?;
            }
            let bits = &shapes.bits[..];
            st.memcpy_htod(bits, &mut res.bits.slice_mut(0..bits.len()))
                .ctx("shape upload")?;
            let ni = n as i32;
            let ([lx, ly, lz], [nx, ny, nz]) = (cube.lo, cube.dims);
            let mut b = st.launch_builder(&self.k_rays);
            b.arg(&res.rays).arg(&ni).arg(&res.cells);
            b.arg(&lx).arg(&ly).arg(&lz).arg(&nx).arg(&ny).arg(&nz);
            b.arg(&res.header)
                .arg(&res.values)
                .arg(&res.bits)
                .arg(&mut res.out);
            let config = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(64), 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(config) }.ctx("exposure rays")?;
            let h = res.h_out.as_mut_slice().ctx("host buffer")?;
            st.memcpy_dtoh(&res.out.slice(0..n), &mut h[..n])
                .ctx("ray readback")?;
            st.synchronize().ctx("exposure wait")?;
            Ok(res.h_out.as_slice().ctx("host buffer")?[..n].to_vec())
        })();
        match result {
            Ok(v) => {
                self.pool.lock().unwrap().push(res);
                Ok(v)
            }
            Err(e) => {
                let _ = res.st.synchronize();
                self.broken.store(true, Relaxed);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shape's words as `kex_rays` reads them: the unit cube, the empty shape, a slab and a
    /// grid of two boxes that is not a cube, each after the others.
    #[test]
    fn ray_shapes_lay_out_grids_bits_and_boxes() {
        let mut shapes = RayShapes::default();
        let slab = VoxelShape::from_cube([1, 2, 1], &[true, false]).unwrap();
        let grid = VoxelShape::from_grid(
            [vec![0.0, 0.25, 1.0], vec![0.0, 0.5], vec![0.0, 1.0]],
            &[true, false],
        )
        .unwrap();
        for (i, shape) in [VoxelShape::block(), VoxelShape::empty(), slab, grid.clone()]
            .iter()
            .enumerate()
        {
            assert_eq!(shapes.push(shape), Some(i as u16));
        }
        assert_eq!(shapes.len(), 4);
        // The unit cube: a cube grid of one full cell and one box
        assert_eq!(shapes.header[..8], [1, 1, 1, 1, 0, 0, 6, 1]);
        assert_eq!(shapes.values[6..12], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(shapes.bits[0], 1);
        // The empty shape has no box
        assert_eq!(shapes.header[8 + 7], 0);
        // The slab: its lower half full
        assert_eq!(shapes.header[16..20], [1, 1, 2, 1]);
        assert_eq!(shapes.bits[shapes.header[16 + 5] as usize], 0b01);
        // The grid: its boundaries, then its box
        let h = &shapes.header[24..32];
        assert_eq!(h[..4], [0, 2, 1, 1]);
        let at = h[4] as usize;
        assert_eq!(
            shapes.values[at..at + 7],
            [0.0, 0.25, 1.0, 0.0, 0.5, 0.0, 1.0]
        );
        assert_eq!(h[7] as usize, grid.boxes().len());
        assert_eq!(shapes.bits[h[5] as usize], 0b01);
    }

    /// exposure.cu compiles with the crate's NVRTC options (skipped without the NVRTC library, as
    /// in a build without the `cuda` feature).
    #[test]
    fn the_exposure_kernel_compiles() {
        if crate::nvrtc_version().is_none() {
            eprintln!("skipping: no NVRTC library");
            return;
        }
        let options = crate::cuda::nvrtc::CompileOptions {
            arch: Some("compute_75"),
            options: crate::NVRTC_OPTIONS.iter().map(|&o| o.into()).collect(),
            ..Default::default()
        };
        crate::cuda::nvrtc::compile_ptx_with_opts(SOURCE, options).expect("exposure.cu");
    }
}
