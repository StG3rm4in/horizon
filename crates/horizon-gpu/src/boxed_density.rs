//! Horizontal launch footprints for GPU world generation with boxed native density.
use horizon_core::density::graph::BoxedDensityDomain;

pub(crate) fn chunk_square(domain: &BoxedDensityDomain, first: [i64; 2], side: i64) -> bool {
    // Include the closing chunk edge. CPU climate's strided bounding volume
    // can reach block 15, even though the last quart is at block 12.
    domain.permits(first.map(|c| c * 16), first.map(|c| (c + side) * 16))
}

pub fn chunk_neighbours(domain: &BoxedDensityDomain, cx: i32, cz: i32) -> bool {
    chunk_square(domain, [i64::from(cx) - 1, i64::from(cz) - 1], 3)
}

pub fn carvers(domain: &BoxedDensityDomain, cx: i32, cz: i32) -> bool {
    // submit_caves reads origins [-8,+8]; a carver_block plans all eight
    // origins along each axis even if this target uses just one of them.
    let c = [i64::from(cx), i64::from(cz)];
    let first = c.map(|v| (v - 8).div_euclid(8) * 8 * 16);
    let end = c.map(|v| ((v + 8).div_euclid(8) + 1) * 8 * 16);
    domain.permits(first, end)
}

pub(crate) fn region(
    domain: &BoxedDensityDomain,
    cx: i32,
    cz: i32,
    side: i32,
    aquifers: bool,
) -> bool {
    if domain.is_empty() {
        return true;
    }
    let side = i64::from(side);
    let first = [cx, cz].map(|c| i64::from(c).div_euclid(side) * side);
    chunk_square(domain, first, side) && (!aquifers || aquifer_region(domain, first, side))
}

fn aquifer_region(domain: &BoxedDensityDomain, first: [i64; 2], side: i64) -> bool {
    let low = first.map(|c| c * 16);
    let high = first.map(|c| (c + side) * 16 - 1);

    // Engine::aquifers: the location grid and its 0..=9 cell jitter.
    let cells_low = low.map(|x| (x - 5).div_euclid(16) * 16);
    let cells_high = high.map(|x| ((x - 5).div_euclid(16) + 1) * 16 + 9);
    let [x0, z0] = cells_low;
    let [x1, z1] = cells_high;
    let quart_block = |x: i64| x.div_euclid(4) * 4;
    // kaq_surface materializes the full rectangle, including offsets that
    // an individual fluid decision might never consult.
    let surface_low = [quart_block(x0 - 48), quart_block(z0 - 16)];
    let surface_high = [quart_block(x1 + 16), quart_block(z1 + 16)];
    domain.permits(surface_low, surface_high)
        && domain.permits(cells_low, cells_high)
        // aq_level/AQ_SPREAD and aq_fluid/AQ_LAVA receive transformed block inputs.
        && [16, 64].into_iter().all(|divisor| {
            domain.permits(
                cells_low.map(|x| x.div_euclid(divisor)),
                cells_high.map(|x| x.div_euclid(divisor)),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::density::{
        ConstantSampler, DensitySampler, Interval,
        graph::{DensityGraph, GraphNode, SeededDensity},
    };
    use std::sync::Arc;

    #[derive(Debug)]
    struct Leaf(Option<([i32; 2], [i32; 2], f32)>);

    impl SeededDensity for Leaf {
        fn sampler(&self, _: i64) -> Arc<dyn DensitySampler> {
            Arc::new(ConstantSampler(0.0))
        }

        fn constant_outside_xz(&self, _: i64) -> Option<([i32; 2], [i32; 2], f32)> {
            self.0
        }
    }

    fn domain(point: [i32; 2], steps: &[i32]) -> BoxedDensityDomain {
        let mut graph = DensityGraph::new();
        let mut id = graph
            .bind_seeded(
                Arc::new(Leaf(Some((point, point, 0.0)))),
                Interval::new(0.0, 1.0).unwrap(),
                5,
            )
            .unwrap();
        for &xz in steps {
            id = graph
                .insert(GraphNode::Interpolated {
                    input: id,
                    xz,
                    y: 4,
                })
                .unwrap();
        }
        graph.boxed_density_domain(0).unwrap()
    }

    #[test]
    fn f11_boxed_region_corner_and_negative_coordinates() {
        let d = domain([0, 0], &[]);
        assert!(!region(&d, -1, -1, 8, false));
        assert!(region(&d, -9, -1, 8, false));
        assert!(region(&d, 8, 0, 8, false));
        let d = domain([24, 8], &[4]);
        assert!(!region(&d, 0, 0, 1, false));
        assert!(region(&domain([25, 8], &[4]), 0, 0, 1, false));
        assert!(!region(&domain([28, 8], &[4, 2]), 0, 0, 1, false));
        assert!(!d.permits([i64::from(i32::MIN), 0], [0, 16]));
    }

    #[test]
    fn f11_boxed_aquifer_and_origin_footprints() {
        let d = domain([-64, -32], &[]);
        assert!(region(&d, 0, 0, 1, false));
        assert!(!region(&d, 0, 0, 1, true));
        assert!(region(&domain([-65, -32], &[]), 0, 0, 1, true));
        let d = domain([100, 100], &[]);
        assert!(region(&d, 100, 100, 1, false));
        assert!(!region(&d, 100, 100, 1, true));
        assert!(!region(&domain([25, 25], &[]), 100, 100, 1, true));
        assert!(!carvers(&domain([240, 240], &[]), 0, 0));
        assert!(carvers(&domain([257, 240], &[]), 0, 0));
        assert!(!chunk_neighbours(&domain([-16, 32], &[]), 0, 0));
    }

    #[test]
    fn f11_boxed_refuses_invalid_or_missing_declarations_and_horizontal_slices() {
        for declaration in [
            None,
            Some(([2, 0], [1, 0], 0.0)),
            Some(([0, 0], [1, 1], 2.0)),
        ] {
            let mut graph = DensityGraph::new();
            graph
                .bind_seeded(
                    Arc::new(Leaf(declaration)),
                    Interval::new(0.0, 1.0).unwrap(),
                    5,
                )
                .unwrap();
            assert!(graph.boxed_density_domain(0).is_err());
        }

        let mut graph = DensityGraph::new();
        let id = graph
            .bind_seeded(
                Arc::new(Leaf(Some(([0, 0], [1, 1], 0.0)))),
                Interval::new(0.0, 1.0).unwrap(),
                5,
            )
            .unwrap();
        graph
            .insert(GraphNode::Slice {
                axis: horizon_core::geometry::Axis::X,
                coordinate: 0,
                input: id,
            })
            .unwrap();
        assert!(graph.boxed_density_domain(0).is_err());
    }
}
