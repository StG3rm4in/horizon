//! A carver planned without its target (`WorldCarver::plan`) and executed for a target carves the
//! cells `carve` does.
use horizon_core::{carving::CarvingMask, position::ChunkPos, random::LegacyRandom};
use horizon_world::worldgen::{anchor::WorldGenerationContext, carver::WorldCarver};

const CAVE: &str = r#"{"type":"minecraft:cave","count":{"type":"minecraft:very_biased_to_bottom","max_inclusive":14,"min_inclusive":0},"floor_level":{"type":"minecraft:uniform","max_exclusive":-0.4,"min_inclusive":-1.0},"horizontal_radius_multiplier":{"type":"minecraft:uniform","max_exclusive":1.4,"min_inclusive":0.7},"probability":0.15,"room_vertical_radius_multiplier":{"type":"minecraft:uniform","max_exclusive":0.9,"min_inclusive":0.1},"thickness":{"type":"minecraft:trapezoid","max":3.0,"min":0.0,"plateau":1.0},"vertical_radius_multiplier":{"type":"minecraft:uniform","max_exclusive":1.3,"min_inclusive":0.8},"weird_thickness_bias":true,"y":{"type":"minecraft:uniform","max_inclusive":{"absolute":180},"min_inclusive":{"above_bottom":8}}}"#;
const NETHER_CAVE: &str = r#"{"type":"minecraft:cave","count":{"type":"minecraft:very_biased_to_bottom","max_inclusive":9,"min_inclusive":0},"floor_level":-0.7,"horizontal_radius_multiplier":1.0,"probability":0.2,"room_vertical_radius_multiplier":0.5,"start_vertical_radius_multiplier":5.0,"thickness":{"type":"minecraft:trapezoid","max":6.0,"min":0.0,"plateau":2.0},"vertical_radius_multiplier":1.0,"y":{"type":"minecraft:uniform","max_inclusive":{"below_top":1},"min_inclusive":{"absolute":0}}}"#;
const CANYON: &str = r#"{"type":"minecraft:canyon","probability":0.01,"shape":{"distance_factor":{"type":"minecraft:uniform","max_exclusive":1.0,"min_inclusive":0.75},"horizontal_radius_factor":{"type":"minecraft:uniform","max_exclusive":1.0,"min_inclusive":0.75},"thickness":{"type":"minecraft:trapezoid","max":6.0,"min":0.0,"plateau":2.0},"vertical_radius_center_factor":0.0,"vertical_radius_default_factor":1.0,"width_smoothness":3,"y_scale":3.0},"vertical_rotation":{"type":"minecraft:uniform","max_exclusive":0.125,"min_inclusive":-0.125},"y":{"type":"minecraft:uniform","max_inclusive":{"absolute":67},"min_inclusive":{"absolute":10}}}"#;

#[test]
fn planned_carvers_carve_the_same_cells() {
    let context = WorldGenerationContext::new(-64, 384, 63);
    let mut cells = 0u64;
    for json in [CAVE, NETHER_CAVE, CANYON] {
        let carver = WorldCarver::from_json(json).unwrap();
        for seed in 0..120i64 {
            let origin = ChunkPos::new((seed % 7) as i32 - 3, (seed % 5) as i32 - 2);
            let mut planned = Vec::new();
            carver
                .plan(context, &mut LegacyRandom::new(seed), origin, &mut planned)
                .unwrap();
            for dx in -8..=8 {
                for dz in [-8, -3, 0, 2, 8] {
                    let target = ChunkPos::new(origin.x + dx, origin.z + dz);
                    let mut a = CarvingMask::new(-63, 312).unwrap();
                    carver
                        .carve(
                            context,
                            &mut LegacyRandom::new(seed),
                            target,
                            origin,
                            &mut a,
                        )
                        .unwrap();
                    let mut b = CarvingMask::new(-63, 312).unwrap();
                    carver.execute(context, &planned, target, &mut b).unwrap();
                    assert_eq!(a.words(), b.words(), "seed {seed}, target {target:?}");
                    cells += a
                        .words()
                        .iter()
                        .map(|w| u64::from(w.count_ones()))
                        .sum::<u64>();
                }
            }
        }
    }
    assert!(cells > 100_000, "{cells} cells");
}
