use super::*;
use crate::worldgen::int_range;

#[derive(Clone, Debug)]
pub struct CanyonShape {
    pub distance_factor: FloatProvider,
    pub thickness: FloatProvider,
    pub width_smoothness: i32,
    pub horizontal_radius_factor: FloatProvider,
    pub vertical_radius_default_factor: f32,
    pub vertical_radius_center_factor: f32,
    pub y_scale: FloatProvider,
}
#[derive(Clone, Debug)]
pub struct CanyonCarver {
    pub probability: f32,
    pub y: HeightProvider,
    pub vertical_rotation: FloatProvider,
    pub shape: CanyonShape,
}
impl CanyonCarver {
    pub fn from_json(json: &str) -> Result<Self, DecodeError> {
        let object: Object = serde_json::from_str(json)?;
        let kind = identifier(&serde_json::from_str::<String>(
            required(&object, "type")?.get(),
        )?)?;
        if kind != "minecraft:canyon" {
            return Err(DecodeError::Field("canyon carver type"));
        }
        let probability = scalar(&object, "probability")?;
        if !(0.0..=1.0).contains(&probability) || probability.is_sign_negative() {
            return Err(DecodeError::Field("carver probability"));
        }
        let shape: Object = serde_json::from_str(required(&object, "shape")?.get())?;
        let provider = |name| FloatProvider::from_json(required(&shape, name)?.get());
        Ok(Self {
            probability,
            y: HeightProvider::from_json(required(&object, "y")?.get())?,
            vertical_rotation: FloatProvider::from_json(
                required(&object, "vertical_rotation")?.get(),
            )?,
            shape: CanyonShape {
                distance_factor: provider("distance_factor")?,
                thickness: provider("thickness")?,
                width_smoothness: int_range(
                    required(&shape, "width_smoothness")?,
                    "width_smoothness",
                    1,
                    i32::MAX,
                )?,
                horizontal_radius_factor: provider("horizontal_radius_factor")?,
                vertical_radius_default_factor: scalar(&shape, "vertical_radius_default_factor")?,
                vertical_radius_center_factor: scalar(&shape, "vertical_radius_center_factor")?,
                y_scale: provider("y_scale")?,
            },
        })
    }
    pub fn is_start_chunk(&self, random: &mut impl RandomSource) -> bool {
        random.next_float() <= self.probability
    }
    /// Generate a canyon from one origin chunk. Width factors cover the entire
    /// generation height and use a separate legacy random stream. Output bounds
    /// must lie within that height; paths are bounded to 4096 steps, 4064 width
    /// entries and 2^24 candidate cells. Earlier output survives errors.
    pub fn carve<O: CarverOutput>(
        &self,
        context: WorldGenerationContext,
        random: &mut impl RandomSource,
        target: ChunkPos,
        origin: ChunkPos,
        output: &mut O,
    ) -> Result<bool, CarverError<O::Error>> {
        let x = f64::from(
            origin
                .x
                .wrapping_shl(4)
                .wrapping_add(random.next_int_bound(16)?),
        );
        let y = f64::from(self.y.sample(random, context)?);
        let z = f64::from(
            origin
                .z
                .wrapping_shl(4)
                .wrapping_add(random.next_int_bound(16)?),
        );
        let yaw = random.next_float() * std::f32::consts::TAU;
        let pitch = self.vertical_rotation.sample(random);
        let y_scale = f64::from(self.shape.y_scale.sample(random));
        let thickness = self.shape.thickness.sample(random);
        let end = (112f32 * self.shape.distance_factor.sample(random)) as i32;
        let mut local = LegacyRandom::new(random.next_long());
        if !(1..=4064).contains(&context.height)
            || output.min_y() < context.min_y
            || i64::from(output.max_y()) > i64::from(context.min_y) + i64::from(context.height)
        {
            return Err(CarverError::InvalidBounds);
        }
        if end > 4096 {
            return Err(CarverError::IterationLimit);
        }
        let height = context.height as usize;
        // Vanilla heights (<= 1024) stay on the stack; taller custom heights spill.
        let mut stack_width = [0f32; 1024];
        let mut heap_width = Vec::new();
        let width: &mut [f32] = if height <= stack_width.len() {
            &mut stack_width[..height]
        } else {
            heap_width.resize(height, 0f32);
            &mut heap_width
        };
        self.fill_width(&mut local, width)?;
        let mut budget = 1 << 24;
        self.trace(
            context,
            CanyonStart {
                position: [x, y, z],
                yaw,
                pitch,
                y_scale,
                thickness,
                end,
            },
            &mut local,
            width,
            target,
            output,
            &mut budget,
        )?;
        Ok(true)
    }
    /// The draws of `carve` that do not depend on the target (`PlannedCanyon`), in the same order.
    /// An error is one `carve` would also return, except the output bounds `execute` checks.
    pub fn plan(
        &self,
        context: WorldGenerationContext,
        random: &mut impl RandomSource,
        origin: ChunkPos,
    ) -> Result<PlannedCanyon, CarverError<std::convert::Infallible>> {
        let x = f64::from(
            origin
                .x
                .wrapping_shl(4)
                .wrapping_add(random.next_int_bound(16)?),
        );
        let y = f64::from(self.y.sample(random, context)?);
        let z = f64::from(
            origin
                .z
                .wrapping_shl(4)
                .wrapping_add(random.next_int_bound(16)?),
        );
        let yaw = random.next_float() * std::f32::consts::TAU;
        let pitch = self.vertical_rotation.sample(random);
        let y_scale = f64::from(self.shape.y_scale.sample(random));
        let thickness = self.shape.thickness.sample(random);
        let end = (112f32 * self.shape.distance_factor.sample(random)) as i32;
        let mut local = LegacyRandom::new(random.next_long());
        if !(1..=4064).contains(&context.height) {
            return Err(CarverError::InvalidBounds);
        }
        if end > 4096 {
            return Err(CarverError::IterationLimit);
        }
        let mut width = vec![0f32; context.height as usize];
        self.fill_width(&mut local, &mut width)?;
        Ok(PlannedCanyon {
            start: CanyonStart {
                position: [x, y, z],
                yaw,
                pitch,
                y_scale,
                thickness,
                end,
            },
            local: local.state(),
            width,
        })
    }
    /// Carve a planned canyon for a target, as `carve` would.
    pub fn execute<O: CarverOutput>(
        &self,
        context: WorldGenerationContext,
        planned: &PlannedCanyon,
        target: ChunkPos,
        output: &mut O,
    ) -> Result<(), CarverError<O::Error>> {
        if output.min_y() < context.min_y
            || i64::from(output.max_y()) > i64::from(context.min_y) + i64::from(context.height)
        {
            return Err(CarverError::InvalidBounds);
        }
        let mut budget = 1 << 24;
        self.trace(
            context,
            planned.start,
            &mut LegacyRandom::from_state(planned.local),
            &planned.width,
            target,
            output,
            &mut budget,
        )
    }
    /// Width factors over the generation height, from the canyon's own stream.
    fn fill_width<E>(
        &self,
        local: &mut LegacyRandom,
        width: &mut [f32],
    ) -> Result<(), CarverError<E>> {
        let mut factor = 1f32;
        for (index, slot) in width.iter_mut().enumerate() {
            if index == 0 || local.next_int_bound(self.shape.width_smoothness)? == 0 {
                factor = 1. + local.next_float() * local.next_float();
            }
            *slot = factor * factor;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn trace<O: CarverOutput>(
        &self,
        context: WorldGenerationContext,
        start: CanyonStart,
        local: &mut LegacyRandom,
        width: &[f32],
        target: ChunkPos,
        output: &mut O,
        budget: &mut usize,
    ) -> Result<(), CarverError<O::Error>> {
        let CanyonStart {
            position: [mut x, mut y, mut z],
            mut yaw,
            mut pitch,
            y_scale,
            thickness,
            end,
        } = start;
        let (mut yaw_change, mut pitch_change) = (0f32, 0f32);
        for step in 0..end {
            let mut horizontal = 1.5
                + f64::from(
                    sin(f64::from(step as f32 * std::f32::consts::PI / end as f32)) * thickness,
                );
            let mut vertical = horizontal * y_scale;
            horizontal *= f64::from(self.shape.horizontal_radius_factor.sample(local));
            let center = 1. - (0.5 - step as f32 / end as f32).abs() * 2.;
            let multiplier = self.shape.vertical_radius_default_factor
                + self.shape.vertical_radius_center_factor * center;
            vertical = (f64::from(multiplier) * vertical)
                * f64::from(local.next_float() * (1. - 0.75) + 0.75);
            let cos_pitch = cos(f64::from(pitch));
            let sin_pitch = sin(f64::from(pitch));
            x += f64::from(cos(f64::from(yaw)) * cos_pitch);
            y += f64::from(sin_pitch);
            z += f64::from(sin(f64::from(yaw)) * cos_pitch);
            pitch *= 0.7;
            pitch += pitch_change * 0.05;
            yaw += yaw_change * 0.05;
            pitch_change *= 0.8;
            yaw_change *= 0.5;
            pitch_change += (local.next_float() - local.next_float()) * local.next_float() * 2.;
            yaw_change += (local.next_float() - local.next_float()) * local.next_float() * 4.;
            if local.next_int_bound(4)? == 0 {
                continue;
            }
            if !can_reach(target, x, z, step, end, thickness) {
                return Ok(());
            }
            carve_ellipsoid_with_budget(
                target,
                [x, y, z],
                horizontal,
                vertical,
                output,
                |[nx, ny, nz], by| {
                    let factor = width[by.wrapping_sub(context.min_y).wrapping_sub(1) as usize];
                    (nx * nx + nz * nz) * f64::from(factor) + ny * ny / 6. >= 1.
                },
                budget,
            )?;
        }
        Ok(())
    }
}
/// A canyon's path parameters, drawn by `carve` before its width table.
#[derive(Clone, Copy, Debug)]
pub struct CanyonStart {
    pub position: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub y_scale: f64,
    pub thickness: f32,
    pub end: i32,
}
/// A canyon planned without its target (`CanyonCarver::plan`): its start, the state of its own
/// random stream after the width table, and the table (factor per Y above the generation minimum).
#[derive(Clone, Debug)]
pub struct PlannedCanyon {
    pub start: CanyonStart,
    pub local: u64,
    pub width: Vec<f32>,
}
fn scalar(object: &Object, name: &'static str) -> Result<f32, DecodeError> {
    number_text(required(object, name)?)?
        .parse()
        .map_err(|_| DecodeError::Field(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::anchor::VerticalAnchor;
    use horizon_core::carving::CarvingMask;

    #[test]
    fn stack_and_heap_width_buffers_carve_across_height_boundary() {
        let make_carver = |y: i32| CanyonCarver {
            probability: 1.0,
            y: HeightProvider::Constant(VerticalAnchor::Absolute(y)),
            vertical_rotation: FloatProvider::Constant(0.0),
            shape: CanyonShape {
                distance_factor: FloatProvider::Constant(1.0),
                thickness: FloatProvider::Constant(3.0),
                width_smoothness: 3,
                horizontal_radius_factor: FloatProvider::Constant(1.0),
                vertical_radius_default_factor: 1.0,
                vertical_radius_center_factor: 0.0,
                y_scale: FloatProvider::Constant(3.0),
            },
        };
        let target = ChunkPos::new(0, 0);
        // Exercise inline stack buffer (384, 1024) and heap-spill buffer (1025),
        // including cells at y = 1025 that index the spilled slot width[1024].
        for (height, start_y, min_y, max_y) in [
            (384, 32, 0, 64),
            (1024, 1018, 992, 1024),
            (1025, 1020, 992, 1025),
        ] {
            let carver = make_carver(start_y);
            let context = WorldGenerationContext::new(0, height, 63);
            let mut mask_a = CarvingMask::new(min_y, max_y).unwrap();
            let mut mask_b = CarvingMask::new(min_y, max_y).unwrap();
            let mut rng_a = LegacyRandom::new(99);
            let mut rng_b = LegacyRandom::new(99);
            assert!(
                carver
                    .carve(context, &mut rng_a, target, target, &mut mask_a)
                    .unwrap()
            );
            let planned = carver.plan(context, &mut rng_b, target).unwrap();
            carver
                .execute(context, &planned, target, &mut mask_b)
                .unwrap();
            let cols_a: Vec<_> = mask_a.columns().collect();
            let cols_b: Vec<_> = mask_b.columns().collect();
            assert!(
                !cols_a.is_empty(),
                "expected canyon cells for height={height}"
            );
            assert!(
                cols_a.iter().any(|c| c.max_y >= start_y),
                "expected canyon cells near start_y={start_y} for height={height}"
            );
            assert_eq!(cols_a, cols_b);
            assert_eq!(rng_a.state(), rng_b.state());
        }
    }
}
