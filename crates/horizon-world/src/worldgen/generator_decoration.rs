//! FEATURES status over a 3×3 neighborhood of generated chunks.
//!
//! Placed features are resolved by name from the bundled feature registries.
//! Roots whose algorithm or bindings are not implemented are left out of
//! execution (but keep their place in vanilla's per-step feature ordering, so
//! decoration seeds of the remaining features are unchanged).
use super::structure::{
    BoundingBox,
    start::{STEPS, place_in_chunk},
};
use super::{
    anchor::WorldGenerationContext,
    biome::BiomeId,
    biome_chunk::BiomeRegion,
    biome_predicate::BiomePredicateBindings,
    biome_source::ResolvedBiomeSource,
    biome_temperature::BiomeTemperature,
    biome_weather::BiomeWeatherBindings,
    block_feature::FeatureBlockBindings,
    decoration::{DecorationWorld, FeatureDecorationPlan},
    feature::{FeatureBlocks, FeatureWorld},
    feature_catalog::FeatureCatalog,
    feature_graph::{FeatureDefinition, FeatureExecutionError, PlacedFeatureDefinition},
    feature_region::{RegionError, TerrainRegion},
    feature_sort::FeatureIdentity,
    generator::{
        DecorateStats, GenBlockStates, GeneratorData, GeneratorError, ProtoChunk, WorkerState,
    },
    ore_feature::{OreAttempt, OreFeature},
    placement::{
        FeaturePlacer, LayerBlock, PlacementContext, PlacementModifier, PlacementQueryError,
    },
    tags::TagDefinitions,
    terrain::PreLightChunkStage,
    write_zone::GenerationWriteZone,
};
use super::{
    generator_blocks::BlockBehavior,
    state_provider::{StateProviderBindings, StateProviderCatalog},
};
use crate::{
    block_collision::{DynamicCollisionWorld, ShulkerCollisionState},
    block_placement::BlockPlacementBindings,
    block_support::SupportType,
    block_survival::SurvivalWorld,
    dynamic_support::DynamicSupportWorld,
    fluid::{FluidKind, FluidState},
    heightmap::{HeightmapState, HeightmapType},
    registry::BlockStateId,
};
use horizon_core::random::{WorldgenRandom, XoroshiroRandom};
use horizon_core::{
    biome::obfuscate_seed,
    geometry::{Direction, VoxelShape},
    position::{BlockPos, ChunkPos},
};
use serde_json::Value;
use std::collections::HashMap;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub(super) struct Decoration {
    plan: FeatureDecorationPlan,
    names: Vec<String>,
    /// Executable roots by identity, or why the root cannot be executed.
    features: Vec<Result<PlacedFeatureDefinition, String>>,
    /// Placed features of each biome's generation settings, by BiomeId: a
    /// bit per FeatureIdentity.
    biome_features: Vec<Vec<u64>>,
    bindings: FeatureBlockBindings<'static>,
    weather: Option<BiomeWeatherBindings>,
    blocks: FeatureBlocks,
    catalog: FeatureCatalog,
    /// Placed features referenced by name (feature pool elements), decoded lazily.
    named: std::sync::Mutex<HashMap<String, Option<Arc<PlacedFeatureDefinition>>>>,
    /// BlockBehaviour.Properties.postProcess: states whose worldgen writes mark
    /// their own position (mushrooms) or the one above (soul sand, magma).
    postprocess_self: Vec<BlockStateId>,
    postprocess_above: Vec<BlockStateId>,
    /// A superflat world's `FillLayerFeature`s (`TOP_LAYER_MODIFICATION`): the level Y and state
    /// of each layer that is not opaque to `MOTION_BLOCKING`, filled after the structures.
    pub(super) top_layers: Vec<(i32, BlockStateId)>,
    times: Option<FeatureTimes>,
}
/// HORIZON_FEATURE_TIMES=1: the wall time of each placed feature, of each structure's pieces, of
/// the ore prediction and of the rest of `Decoration::decorate`, summed over every decorated
/// chunk; the list in latest.log every 10 000 chunks. Measurement only: nothing it places changes.
struct FeatureTimes {
    /// Per feature identity: nanoseconds, calls, calls that placed something
    features: Vec<[std::sync::atomic::AtomicU64; 3]>,
    /// Per structure index: nanoseconds, calls
    structures: std::sync::Mutex<Vec<[u64; 2]>>,
    /// Whole decorate, setup before the plan, ore prediction, top layers and after
    stages: [std::sync::atomic::AtomicU64; 4],
    chunks: std::sync::atomic::AtomicU64,
}
impl FeatureTimes {
    fn from_env(features: usize) -> Option<Self> {
        std::env::var_os("HORIZON_FEATURE_TIMES")?;
        Some(Self {
            features: (0..features).map(|_| Default::default()).collect(),
            structures: std::sync::Mutex::new(Vec::new()),
            stages: Default::default(),
            chunks: Default::default(),
        })
    }
    fn feature(&self, id: usize, ns: u64, placed: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(f) = self.features.get(id) {
            f[0].fetch_add(ns, Relaxed);
            f[1].fetch_add(1, Relaxed);
            f[2].fetch_add(u64::from(placed), Relaxed);
        }
    }
    fn structure(&self, index: usize, ns: u64) {
        let mut s = self.structures.lock().unwrap();
        if s.len() <= index {
            s.resize(index + 1, [0; 2]);
        }
        s[index][0] += ns;
        s[index][1] += 1;
    }
    fn stage(&self, stage: usize, ns: u64) {
        self.stages[stage].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }
    /// One more decorated chunk; the list every 10 000.
    fn chunk(&self, names: &[String], structures: Option<&super::generator_structures::StructureState>) {
        use std::sync::atomic::Ordering::Relaxed;
        let n = self.chunks.fetch_add(1, Relaxed) + 1;
        if n % 10_000 != 0 {
            return;
        }
        let total = self.stages[0].load(Relaxed).max(1) as f64;
        let mut rows: Vec<(u64, u64, u64, String)> = self
            .features
            .iter()
            .zip(names)
            .map(|(f, name)| (f[0].load(Relaxed), f[1].load(Relaxed), f[2].load(Relaxed), name.clone()))
            .collect();
        if let Some(st) = structures {
            for (i, [ns, calls]) in self.structures.lock().unwrap().iter().enumerate() {
                let name = st.registry.structures.get(i).map_or("?", |d| d.name.as_str());
                rows.push((*ns, *calls, *calls, format!("structure {name}")));
            }
        }
        for (k, name) in ["(setup before the plan)", "(ore prediction)", "(top layers and after)"]
            .iter()
            .enumerate()
        {
            rows.push((self.stages[k + 1].load(Relaxed), n, n, name.to_string()));
        }
        rows.sort_by(|a, b| b.0.cmp(&a.0));
        let mut out = format!(
            "feature times: {n} chunks, decorate {:.0} us per chunk",
            total / n as f64 / 1000.0
        );
        for (ns, calls, placed, name) in rows.iter().take(60).filter(|r| r.0 > 0) {
            out.push_str(&format!(
                "\n  {:5.2} %  {:7.1} us/chunk  {:6.2} us/call  {:8} calls  {:5.1} % placed  {name}",
                100.0 * *ns as f64 / total,
                *ns as f64 / n as f64 / 1000.0,
                *ns as f64 / (*calls).max(1) as f64 / 1000.0,
                calls,
                100.0 * *placed as f64 / (*calls).max(1) as f64,
            ));
        }
        log::info!("{out}");
    }
}
fn resource(message: impl Into<String>) -> GeneratorError {
    GeneratorError::Resource(message.into())
}
/// Biomes in vanilla's possible-biome encounter order for this source.
pub(super) fn possible_biomes(source: &ResolvedBiomeSource) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let ordered: Vec<String> = match source {
        ResolvedBiomeSource::MultiNoise(source) => source
            .parameters()
            .entries()
            .iter()
            .map(|(_, name)| name.clone())
            .collect(),
        ResolvedBiomeSource::End(source) => [
            &source.center,
            &source.highlands,
            &source.midlands,
            &source.islands,
            &source.barrens,
        ]
        .into_iter()
        .cloned()
        .collect(),
        other => other
            .possible_biomes()
            .into_iter()
            .map(str::to_owned)
            .collect(),
    };
    ordered
        .into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect()
}
/// `FlatLevelGeneratorSettings.adjustGenerationSettings`: what a superflat world keeps of its
/// biome's generation settings.
#[derive(Clone, Copy, Debug)]
pub(super) struct FlatDecoration {
    /// `(!voidGen || biome is the_void) && decoration`: the biome's own placed features, but
    /// those of the two structure steps (and of the lakes step when `lakes` is on).
    pub(super) features: bool,
    /// `addLakes`: the lakes step is `createLakesList`.
    pub(super) lakes: bool,
}
/// `GenerationStep.Decoration` ordinals `adjustGenerationSettings` treats on their own.
const LAKES_STEP: usize = 1;
const UNDERGROUND_STRUCTURES_STEP: usize = 3;
const SURFACE_STRUCTURES_STEP: usize = 4;
/// `FlatLevelGeneratorSettings.createLakesList`.
const FLAT_LAKES: [&str; 2] = [
    "minecraft:lake_lava_underground",
    "minecraft:lake_lava_surface",
];
impl FlatDecoration {
    /// The settings' `features` and `lakes`, for layers that are all air (`void_gen`) or not and a
    /// biome that is the_void or not.
    pub(super) fn new(features: bool, lakes: bool, void_gen: bool, the_void: bool) -> Self {
        Self {
            features: (!void_gen || the_void) && features,
            lakes,
        }
    }
    /// Which of the biome's own entries step `step` keeps.
    fn keeps_biome_step(self, step: usize) -> bool {
        self.features
            && step != UNDERGROUND_STRUCTURES_STEP
            && step != SURFACE_STRUCTURES_STEP
            && !(self.lakes && step == LAKES_STEP)
    }
}
impl Decoration {
    pub(super) fn new(
        data: &Arc<GeneratorData>,
        source: &ResolvedBiomeSource,
    ) -> Result<Option<Self>, GeneratorError> {
        Self::new_with(data, source, None)
    }
    /// `new` for a superflat world (`flat`): its fixed biome's settings as
    /// `adjustGenerationSettings` leaves them. The steps stay (structures place in them);
    /// the `FillLayerFeature`s are `top_layers`.
    pub(super) fn new_with(
        data: &Arc<GeneratorData>,
        source: &ResolvedBiomeSource,
        flat: Option<FlatDecoration>,
    ) -> Result<Option<Self>, GeneratorError> {
        let Some(resources) = &data.feature_resources else {
            return Ok(None);
        };
        let registry = data.registry();
        let mut feature_tags = TagDefinitions::default();
        for (name, json) in &resources.feature_tags {
            feature_tags.apply_json(name, "vanilla", json)?;
        }
        let catalog =
            FeatureCatalog::from_resources(&resources.features, &resources.placed, &feature_tags)?;
        let biomes = Arc::new(BiomePredicateBindings::new(
            &data
                .biome_names()
                .iter()
                .enumerate()
                .map(|(i, name)| (name.to_string(), BiomeId(i as u32)))
                .collect(),
            &data.biome_tags,
        )?);
        let mut bindings = data
            .feature_block_bindings()?
            .with_fluid_tags(&resources.fluid_tags)?
            .with_biomes(biomes.clone());
        if let Some(structures) = &data.structures {
            bindings = bindings.with_structures(structures.clone());
        }
        if let Some(blocks) = &resources.blocks {
            bindings = bindings
                .with_block_properties(&blocks.properties)?
                .with_state_providers(
                    StateProviderBindings::new(
                        registry,
                        data.integers().clone(),
                        &data.block_tags,
                    )?
                    .with_block_properties(&blocks.properties)?
                    .with_fluid_tags(&resources.fluid_tags)?
                    .with_biomes(biomes),
                    StateProviderCatalog::from_resources(&resources.state_providers)?,
                    BlockPlacementBindings::new(
                        registry,
                        &blocks.properties,
                        &blocks.double_plants,
                        &blocks.mossy_carpets,
                    )?,
                )?;
        }
        let tags = feature_tags
            .build(|name, _| resources.placed.contains_key(name).then(|| name.to_owned()))
            .map_err(|e| resource(format!("feature tags: {e:?}")))?
            .tags;
        let mut ids = BTreeMap::<String, FeatureIdentity>::new();
        let mut names = Vec::new();
        let mut features = Vec::new();
        let mut identity = |name: &str| -> FeatureIdentity {
            *ids.entry(name.to_owned()).or_insert_with(|| {
                let id = FeatureIdentity(names.len() as u32);
                names.push(name.to_owned());
                features.push(
                    catalog
                        .placed_feature_with_blocks(&Value::from(name).to_string(), &bindings)
                        .map_err(|e| e.to_string()),
                );
                id
            })
        };
        let mut biome_features = vec![Vec::new(); data.biome_names().len()];
        let mut plan = Vec::new();
        for name in possible_biomes(source) {
            let id = data
                .biome_id(&name)
                .ok_or_else(|| resource(format!("unknown biome {name}")))?;
            let json: Value = serde_json::from_str(data.biome_json(id))?;
            let mut steps = Vec::new();
            for (index, step) in json["features"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let mut list = Vec::new();
                // The builder takes the lakes first, then the biome's steps it keeps.
                if flat.is_some_and(|f| f.lakes) && index == LAKES_STEP {
                    for name in FLAT_LAKES {
                        let feature = identity(name);
                        list.push(feature);
                        set_feature_bit(&mut biome_features[id.0 as usize], feature);
                    }
                }
                if flat.is_some_and(|f| !f.keeps_biome_step(index)) {
                    steps.push(list);
                    continue;
                }
                let entries = match step {
                    Value::String(tag) if tag.starts_with('#') => tags
                        .get(&super::tags::identifier(&tag[1..])?)
                        .cloned()
                        .ok_or_else(|| resource(format!("unknown feature tag {tag}")))?,
                    Value::String(name) => vec![name.clone()],
                    Value::Array(values) => values
                        .iter()
                        .map(|v| {
                            v.as_str()
                                .map(str::to_owned)
                                .ok_or_else(|| resource("inline placed feature in biome"))
                        })
                        .collect::<Result<_, _>>()?,
                    _ => return Err(resource("biome feature step")),
                };
                for entry in entries {
                    let feature = identity(&super::tags::identifier(&entry)?);
                    list.push(feature);
                    set_feature_bit(&mut biome_features[id.0 as usize], feature);
                }
                steps.push(list);
            }
            // A biome with fewer steps still gets the lakes (`PlainBuilder` grows its list).
            if flat.is_some_and(|f| f.lakes) && steps.len() <= LAKES_STEP {
                steps.resize_with(LAKES_STEP, Vec::new);
                let lakes = FLAT_LAKES.map(&mut identity).to_vec();
                for &feature in &lakes {
                    set_feature_bit(&mut biome_features[id.0 as usize], feature);
                }
                steps.push(lakes);
            }
            plan.push((id, steps));
        }
        let plan = FeatureDecorationPlan::new(plan).map_err(|e| resource(e.to_string()))?;
        let weather = BiomeWeatherBindings::new(&bindings).ok();
        let states = |names: &[&str]| -> Vec<BlockStateId> {
            names
                .iter()
                .filter_map(|n| registry.default_state(n))
                .collect()
        };
        let postprocess_self = states(&["minecraft:brown_mushroom", "minecraft:red_mushroom"]);
        let postprocess_above = states(&["minecraft:soul_sand", "minecraft:magma_block"]);
        Ok(Some(Self {
            postprocess_self,
            postprocess_above,
            plan,
            times: FeatureTimes::from_env(names.len()),
            names,
            features,
            biome_features,
            weather,
            blocks: FeatureBlocks::from_registry(registry)?,
            bindings,
            catalog,
            named: std::sync::Mutex::new(HashMap::new()),
            top_layers: Vec::new(),
        }))
    }
    fn named_feature(&self, name: &str) -> Option<Arc<PlacedFeatureDefinition>> {
        let mut named = self.named.lock().unwrap();
        named
            .entry(name.to_owned())
            .or_insert_with(|| {
                self.catalog
                    .placed_feature_with_blocks(&Value::from(name).to_string(), &self.bindings)
                    .ok()
                    .map(Arc::new)
            })
            .clone()
    }
    /// Vanilla's feature identity and the step-local index/step mixed into
    /// `set_feature_seed`.
    pub(super) fn seed(&self, name: &str) -> Option<(u32, i32, i32)> {
        let identity = self.names.iter().position(|candidate| candidate == name)? as u32;
        let (index, step) = self.plan.seed(FeatureIdentity(identity))?;
        Some((identity, index as i32, step as i32))
    }
    /// Replays a feature's placement chain and the feature itself on a no-op
    /// world. This advances vanilla's random stream but writes no blocks, so a
    /// caller can reproduce the exact origins without generating terrain.
    pub(super) fn origins(
        &self,
        world_seed: i64,
        chunk: ChunkPos,
        mut biome_allows: impl FnMut([i32; 3]) -> bool,
    ) -> Vec<[i32; 3]> {
        let Some((identity, index, step)) = self.seed("minecraft:end_island_decorated") else {
            return Vec::new();
        };
        let Ok(feature) = &self.features[identity as usize] else {
            return Vec::new();
        };
        let blocks = self.blocks;
        let mut random = WorldgenRandom::new(XoroshiroRandom::new(0));
        let seed = random.set_decoration_seed(world_seed, chunk.x * 16, chunk.z * 16);
        random.set_feature_seed(seed, index, step);
        let mut world = OriginWorld {
            air: blocks.air,
            generation: WorldGenerationContext::new(0, 256, 0),
            biome_allows: &mut biome_allows,
        };
        let mut origins = Vec::new();
        let mut placer = FeaturePlacer::new();
        let _ = placer.place_with_biome_check(
            &feature.placement,
            &mut world,
            &mut random,
            [chunk.x * 16, 0, chunk.z * 16],
            FeatureIdentity(identity),
            |world, random, position| {
                origins.push(position);
                feature
                    .feature
                    .place(world, random, position, blocks)
                    .map(|_| true)
                    .map_err(|_| ())
            },
        );
        origins
    }
    /// Placed-feature roots that cannot run, with their decode errors.
    pub(super) fn unsupported(&self) -> Vec<(String, String)> {
        self.names
            .iter()
            .zip(&self.features)
            .filter_map(|(name, feature)| feature.as_ref().err().map(|e| (name.clone(), e.clone())))
            .collect()
    }
    /// The chunk's ores planned on the GPU: the attempts of its plannable ores (pure targets, no
    /// air-exposure chance, placement modifiers that read no block) predicted from the region as
    /// it stands before any of the chunk's features, and their blobs sent to the GPU. Every draw
    /// of such an attempt before its cells depends on the world only through its column search,
    /// which is asked again at the ore step (earlier features may raise the floor).
    fn predict_ores<'g>(
        &self,
        world: &mut World<'_>,
        center: ChunkPos,
        seed: i64,
        gpu: &'g horizon_gpu::OreGpu,
    ) -> Option<OrePlan<'g>> {
        let step = STEPS.iter().position(|s| *s == "underground_ores")?;
        let features = self.plan.step_features(world, center, step).ok()?;
        let origin = [
            center.x.wrapping_shl(4),
            world.min_section_y().ok()?.wrapping_shl(4),
            center.z.wrapping_shl(4),
        ];
        let mut random = WorldgenRandom::new(XoroshiroRandom::new(0));
        let decoration_seed = random.set_decoration_seed(seed, origin[0], origin[2]);
        let mut placer = FeaturePlacer::new();
        let OreBuffers { features: mut planned, mut attempts, mut spheres, mut blobs, mut sent } =
            OreBuffers::take();
        for (index, id) in features {
            let Ok(feature) = &self.features[id.0 as usize] else {
                continue;
            };
            let FeatureDefinition::Ore(ore) = &*feature.feature else {
                continue;
            };
            if !ore.plannable() || !feature.placement.iter().all(static_modifier) {
                continue;
            }
            random.set_feature_seed(decoration_seed, index as i32, step as i32);
            let (first, first_sphere) = (attempts.len(), spheres.len());
            let predicted = placer.place_with_biome_check(
                &feature.placement,
                world,
                &mut random,
                origin,
                id,
                |world, random, p| {
                    let attempt = ore.predict(world, random, p, &mut spheres)?;
                    attempts.push(attempt);
                    Ok::<_, FeatureExecutionError<RegionError>>(attempt.found)
                },
            );
            if predicted.is_err() {
                attempts.truncate(first);
                spheres.truncate(first_sphere);
                continue;
            }
            planned.push((id, first..attempts.len(), ore.size() as usize));
        }
        if planned.is_empty() {
            OreBuffers { features: planned, attempts, spheres, blobs, sent }.give();
            return None;
        }
        blobs.resize(attempts.len(), None);
        let mut next = 0;
        for (_, range, size) in &planned {
            for i in range.clone() {
                if attempts[i].found {
                    blobs[i] = Some((sent.len(), next));
                    sent.push(horizon_gpu::OreBlob {
                        first: next as u32,
                        count: *size as u32,
                        start: attempts[i].start,
                        size_xz: attempts[i].size_xz,
                        size_y: attempts[i].size_y,
                    });
                    next += size;
                }
            }
        }
        let job = (next == spheres.len()).then(|| gpu.submit(&sent, &spheres).ok()).flatten();
        let Some(job) = job else {
            OreBuffers { features: planned, attempts, spheres, blobs, sent }.give();
            return None;
        };
        sent.clear();
        OREBUFFERS.with_borrow_mut(|b| b.sent = sent);
        Some(OrePlan {
            gpu,
            features: planned,
            attempts,
            blobs,
            spheres,
            job,
            floor_writes: world.floor_writes,
            test: ores_test(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decorate(
        &self,
        data: &Arc<GeneratorData>,
        seed: i64,
        generation: WorldGenerationContext,
        worker: &mut WorkerState,
        region: &mut [ProtoChunk; 9],
        level_random: &mut dyn super::structure::legacy::PieceRandom,
        structures: Option<&super::generator_structures::StructureState>,
        ores: Option<&horizon_gpu::OreGpu>,
    ) -> Result<DecorateStats, GeneratorError> {
        let started = self.times.as_ref().map(|_| std::time::Instant::now());
        let center = region[4].position();
        for (i, chunk) in region.iter().enumerate() {
            let expected = ChunkPos::new(
                center.x.wrapping_add(i as i32 % 3 - 1),
                center.z.wrapping_add(i as i32 / 3 - 1),
            );
            if chunk.position() != expected {
                return Err(resource("decoration region is not a 3×3 neighborhood"));
            }
            if (chunk.terrain.generation_stage() as u8) < PreLightChunkStage::Terrain as u8 {
                return Err(resource("decoration requires terrain-stage chunks"));
            }
        }
        if region[4].is_decorated() {
            return Ok(DecorateStats::default());
        }
        // The full chunks of the region, before their terrain moves into it.
        let full: Vec<ChunkPos> = region
            .iter()
            .filter(|chunk| chunk.is_full())
            .map(|chunk| chunk.position())
            .collect();
        let mut chunks = Vec::with_capacity(9);
        let mut biomes = Vec::with_capacity(9);
        let (min_y, height) = (region[4].terrain.min_y(), region[4].terrain.height());
        for chunk in region.iter_mut() {
            // A one-section stand-in while the region holds the terrain: it
            // is swapped back below, and allocating the full height for nine
            // stand-ins per region cost more than some features.
            let empty = super::terrain::TerrainChunk::new(
                chunk.position(),
                min_y,
                16,
                data.registry(),
                data.block_states().clone(),
            )
            .map_err(|e| resource(e.to_string()))?;
            chunks.push(std::mem::replace(&mut chunk.terrain, empty));
            biomes.push(chunk.biomes().clone());
        }
        let terrain = TerrainRegion::new(chunks, generation)
            .and_then(|region| region.with_block_bindings(&self.bindings))
            .map(|region| region.with_world_seed(seed));
        let terrain = match terrain {
            Ok(terrain) => terrain,
            Err(e) => return Err(resource(e.to_string())),
        };
        let structures_only = worker.structures_only || !region[4].features;
        let references = region[4].structure_starts.clone();
        let mut block_entities = Vec::new();
        for chunk in region.iter_mut() {
            block_entities.append(&mut chunk.block_entities);
        }
        let mut world = World {
            block_entities,
            entities: Vec::new(),
            level_random,
            structure_mobs: worker.structure_mobs.clone(),
            starts: references.clone(),
            biome_names: data.biome_names(),
            blocks: data
                .feature_resources
                .as_ref()
                .and_then(|r| r.blocks.as_ref()),
            region: terrain,
            biomes: BiomeRegion::new(biomes, obfuscate_seed(seed))
                .map_err(|e| resource(e.to_string()))?,
            zone: GenerationWriteZone::new(center, 1),
            center,
            uniform_biomes: vec![UNKNOWN_BIOME; 9 * (height / 16).max(0) as usize],
            biome_quarts: region[4].biomes().quart_range(),
            decoration: self,
            temperatures: &mut worker.temperatures,
            behaviours: data.behaviours.as_deref(),
            // WorldGenRegion.isOutsideBuildHeight: the level's range, not the noise's.
            build: (min_y, min_y + height),
            shape_ticks: std::cell::RefCell::new(crate::level::Ticks::new(0)),
            last_tag: None,
            floor_bounds: [None; 9],
            floor_writes: 0,
            shape_capture: std::cell::RefCell::new(Vec::new()),
            full,
        };
        // The chunk's GPU ores: predicted before any of its features, their blobs worked out on
        // the GPU while the earlier steps run.
        if let (Some(t), Some(d)) = (&self.times, started) {
            t.stage(1, d.elapsed().as_nanos() as u64);
        }
        let prediction = self.times.as_ref().map(|_| std::time::Instant::now());
        let mut ore_plan = match ores {
            Some(gpu) if !structures_only => self.predict_ores(&mut world, center, seed, gpu),
            _ => None,
        };
        if let (Some(t), Some(d)) = (&self.times, prediction) {
            t.stage(2, d.elapsed().as_nanos() as u64);
        }
        let mut stats = DecorateStats::default();
        let blocks = self.blocks;
        // The blocks' facts: `BlockState.isSolidRender` and the rest, the
        // structure pieces ask them of the blocks around them.
        let tree_blocks = self
            .bindings
            .tree_blocks()
            .map_err(|e| resource(e.to_string()))?;
        // ChunkGenerator.getWritableArea: the chunk's minimum Y + 1 to its maximum Y.
        let chunk_box = BoundingBox::new(
            [center.x * 16, min_y + 1, center.z * 16],
            [center.x * 16 + 15, min_y + height - 1, center.z * 16 + 15],
        );
        let mut structure_errors = 0;
        let result = self.plan.decorate(
            &mut world,
            center,
            seed,
            STEPS.len(),
            |world, random, decoration_seed, step| {
                let Some(structures) = structures else {
                    return Ok(());
                };
                let Some(in_step) = structures.by_step.get(step) else {
                    return Ok(());
                };
                for (index, &structure) in in_step.iter().enumerate() {
                    random.set_feature_seed(decoration_seed, index as i32, step as i32);
                    let measure = self
                        .times
                        .as_ref()
                        .filter(|_| references.iter().any(|s| s.structure == structure))
                        .map(|t| (t, std::time::Instant::now()));
                    for start in references.iter().filter(|s| s.structure == structure) {
                        let mut place_feature =
                            |world: &mut World<'_>,
                             random: &mut WorldgenRandom<XoroshiroRandom>,
                             name: &str,
                             pos: [i32; 3]| {
                                match self.named_feature(name) {
                                    Some(feature) => feature
                                        .place(world, random, pos, blocks)
                                        .map_err(|e| PlacementQueryError::Failed(e.to_string())),
                                    None => Ok(false),
                                }
                            };
                        if place_in_chunk(
                            start,
                            world,
                            &structures.resources,
                            &tree_blocks,
                            random,
                            chunk_box,
                            &mut place_feature,
                        )
                        .is_err()
                        {
                            structure_errors += 1;
                        }
                    }
                    if let Some((t, d)) = measure {
                        t.structure(structure, d.elapsed().as_nanos() as u64);
                    }
                }
                Ok(())
            },
            |world, placer, random, id, origin| {
                let measure = self.times.as_ref().map(|t| (t, std::time::Instant::now()));
                let outcome = (|| -> Result<bool, RegionError> {
                    if structures_only {
                        return Ok(false);
                    }
                    let Ok(feature) = &self.features[id.0 as usize] else {
                        stats.unsupported += 1;
                        stats.skipped.insert(self.names[id.0 as usize].clone());
                        return Ok(false);
                    };
                    stats.placed += 1;
                    if let Some(plan) = ore_plan.as_mut()
                        && let FeatureDefinition::Ore(ore) = &*feature.feature
                        && let Some(placed) = plan.place(world, random, id, ore)
                    {
                        if placed.is_err() {
                            stats.failed += 1;
                            stats.skipped.insert(self.names[id.0 as usize].clone());
                        }
                        return Ok(placed.unwrap_or(false));
                    }
                    let placed = placer.place_with_biome_check(
                        &feature.placement,
                        world,
                        random,
                        origin,
                        id,
                        |world, random, p| feature.feature.place(world, random, p, blocks),
                    );
                    if placed.is_err() {
                        stats.failed += 1;
                        stats.skipped.insert(self.names[id.0 as usize].clone());
                    }
                    Ok(placed.unwrap_or(false))
                })();
                if let (Some((t, d)), Ok(placed)) = (measure, &outcome) {
                    t.feature(id.0 as usize, d.elapsed().as_nanos() as u64, *placed);
                }
                outcome
            },
        );
        if let Some(plan) = ore_plan {
            plan.gpu.finish(plan.job);
            OreBuffers {
                features: plan.features,
                attempts: plan.attempts,
                spheres: plan.spheres,
                blobs: plan.blobs,
                sent: Vec::new(),
            }
            .give();
        }
        let after = self.times.as_ref().map(|_| std::time::Instant::now());
        // `FillLayerFeature.place` for the chunk: air at the layer's Y becomes the layer.
        if result.is_ok() && !structures_only {
            for &(y, state) in &self.top_layers {
                for dz in 0..16 {
                    for dx in 0..16 {
                        let at = [center.x * 16 + dx, y, center.z * 16 + dz];
                        let current = world
                            .block(at)
                            .map_err(|e| GeneratorError::Stage("features", e.to_string()))?;
                        if data.block_states().get(current).is_air() {
                            world
                                .set_block(at, state, 2)
                                .map_err(|e| GeneratorError::Stage("features", e.to_string()))?;
                        }
                    }
                }
            }
        }
        for (pos, nbt) in std::mem::take(&mut world.block_entities) {
            let index = ((pos[2] >> 4).wrapping_sub(center.z) + 1) * 3
                + (pos[0] >> 4).wrapping_sub(center.x)
                + 1;
            if let Some(chunk) = usize::try_from(index).ok().and_then(|i| region.get_mut(i)) {
                chunk.block_entities.push((pos, nbt));
            }
        }
        // WorldGenRegion.addFreshEntity: into the chunk holding the position.
        for nbt in std::mem::take(&mut world.entities) {
            let Some(at) = block_of(&nbt) else {
                continue;
            };
            let index = ((at[2] >> 4).wrapping_sub(center.z) + 1) * 3
                + (at[0] >> 4).wrapping_sub(center.x)
                + 1;
            if let Some(chunk) = usize::try_from(index).ok().and_then(|i| region.get_mut(i)) {
                chunk.entities.push(nbt);
            }
        }
        for terrain in world.region.into_chunks() {
            let p = terrain.position();
            let index = (p.z.wrapping_sub(center.z) + 1) * 3 + p.x.wrapping_sub(center.x) + 1;
            region[index as usize].terrain = terrain;
        }
        result.map_err(|e| GeneratorError::Stage("features", e.to_string()))?;
        stats.failed += structure_errors;
        region[4]
            .terrain
            .set_generation_stage(PreLightChunkStage::Features);
        region[4].set_decorated();
        if let (Some(t), Some(d), Some(a)) = (&self.times, started, after) {
            t.stage(3, a.elapsed().as_nanos() as u64);
            t.stage(0, d.elapsed().as_nanos() as u64);
            t.chunk(&self.names, structures);
        }
        Ok(stats)
    }
}

/// `a`, and `b` when it differs.
fn span<T: PartialEq>(a: T, b: T) -> (T, Option<T>) {
    let b = (a != b).then_some(b);
    (a, b)
}
const UNKNOWN_BIOME: u32 = u32::MAX;
const MIXED_BIOMES: u32 = u32::MAX - 1;
impl World<'_> {
    /// The biome of a biome section when all its quarts hold the same one
    /// (and its chunk has biomes); `None` when mixed or outside the region.
    fn uniform_section(&mut self, cx: i32, cz: i32, section: usize) -> Option<BiomeId> {
        let (dx, dz) = (
            i64::from(cx) - i64::from(self.center.x) + 1,
            i64::from(cz) - i64::from(self.center.z) + 1,
        );
        if !(0..3).contains(&dx) || !(0..3).contains(&dz) {
            return None;
        }
        let sections = self.uniform_biomes.len() / 9;
        if section >= sections {
            return None;
        }
        let slot = (dz * 3 + dx) as usize * sections + section;
        if self.uniform_biomes[slot] == UNKNOWN_BIOME {
            let chunk = ChunkPos::new(cx, cz);
            let ready = self.region.chunk(cx, cz).is_ok_and(|c| {
                c.highest_generated_status() >= crate::chunk_status::ChunkStatus::Biomes
            });
            let values = self
                .biomes
                .chunk(chunk)
                .ok()
                .and_then(|c| c.section(section));
            self.uniform_biomes[slot] = match values {
                Some(values) if ready && values.iter().all(|b| *b == values[0]) => values[0].0,
                _ => MIXED_BIOMES,
            };
        }
        match self.uniform_biomes[slot] {
            MIXED_BIOMES => None,
            id => Some(BiomeId(id)),
        }
    }
    /// The region chunk slot of chunk `cx`, `cz` (as `uniform_section`).
    fn chunk_slot(&self, cx: i32, cz: i32) -> Option<usize> {
        let (dx, dz) = (
            i64::from(cx) - i64::from(self.center.x) + 1,
            i64::from(cz) - i64::from(self.center.z) + 1,
        );
        ((0..3).contains(&dx) && (0..3).contains(&dz)).then_some((dz * 3 + dx) as usize)
    }
    /// A block write at `[x, y, z]` may raise its chunk's heights up to y + 1.
    fn raise_floor_bound(&mut self, [x, y, z]: [i32; 3]) {
        self.floor_writes += 1;
        if let Some(slot) = self.chunk_slot(x >> 4, z >> 4)
            && let Some(bound) = &mut self.floor_bounds[slot]
        {
            *bound = (*bound).max(y.saturating_add(1));
        }
    }
    /// `floor_bounds` for chunk `cx`, `cz`, computed from its heightmap on
    /// first use; `None` outside the region or without the heightmap.
    fn floor_bound(&mut self, cx: i32, cz: i32) -> Option<i32> {
        let slot = self.chunk_slot(cx, cz)?;
        if self.floor_bounds[slot].is_none() {
            let map = self
                .region
                .chunk(cx, cz)
                .ok()?
                .heightmap(HeightmapType::OceanFloorWg)?;
            let mut bound = i32::MIN;
            for x in 0..16 {
                for z in 0..16 {
                    bound = bound.max(map.first_available(x, z).ok()?);
                }
            }
            self.floor_bounds[slot] = Some(bound);
        }
        self.floor_bounds[slot]
    }
    /// BiomeManager.getBiome without the zoom when every quart it can pick
    /// lies in biome sections of one and the same biome.
    fn uniform_biome(&mut self, [x, y, z]: [i32; 3]) -> Option<BiomeId> {
        let base = [x, y, z].map(|v| v.wrapping_sub(2) >> 2);
        let (bottom, top) = self.biome_quarts;
        let section = |q: i32| ((q.clamp(bottom, top) - bottom) >> 2) as usize;
        // The chunks and sections holding the quarts base..=base + 1.
        let (x0, x1) = span(base[0] >> 2, base[0].wrapping_add(1) >> 2);
        let (z0, z1) = span(base[2] >> 2, base[2].wrapping_add(1) >> 2);
        let (s0, s1) = span(section(base[1]), section(base[1].wrapping_add(1)));
        let found = self.uniform_section(x0, z0, s0)?;
        for cx in std::iter::once(x0).chain(x1) {
            for cz in std::iter::once(z0).chain(z1) {
                for s in std::iter::once(s0).chain(s1) {
                    if self.uniform_section(cx, cz, s)? != found {
                        return None;
                    }
                }
            }
        }
        Some(found)
    }
}
/// The ores of a chunk planned on the GPU (`HORIZON_GPU_ORES`): each planned feature's attempts as
/// predicted at the start of the chunk's decoration, and the job working out their blobs' cells.
/// The vectors of a worker's ore plans, kept from one chunk to the next: a chunk's plan holds
/// some 20,000 spheres, which growing vectors would copy again and again.
#[derive(Default)]
struct OreBuffers {
    features: Vec<(FeatureIdentity, std::ops::Range<usize>, usize)>,
    attempts: Vec<OreAttempt>,
    spheres: Vec<[f64; 4]>,
    blobs: Vec<Option<(usize, usize)>>,
    sent: Vec<horizon_gpu::OreBlob>,
}
thread_local! {
    static OREBUFFERS: std::cell::RefCell<OreBuffers> = std::cell::RefCell::default();
}
impl OreBuffers {
    /// The worker's buffers, empty.
    fn take() -> Self {
        let mut b = OREBUFFERS.take();
        b.features.clear();
        b.attempts.clear();
        b.spheres.clear();
        b.blobs.clear();
        b.sent.clear();
        b
    }
    /// Gives the buffers back for the next chunk; an empty `sent` keeps the one already there.
    fn give(self) {
        OREBUFFERS.with_borrow_mut(|b| {
            let sent = if self.sent.capacity() > 0 { self.sent } else { std::mem::take(&mut b.sent) };
            *b = OreBuffers { sent, ..self };
        });
    }
}
struct OrePlan<'g> {
    gpu: &'g horizon_gpu::OreGpu,
    /// Each planned feature, the range of its attempts and the sphere count of its blobs.
    features: Vec<(FeatureIdentity, std::ops::Range<usize>, usize)>,
    attempts: Vec<OreAttempt>,
    /// For each attempt that found a column, its blob in the job and its first sphere.
    blobs: Vec<Option<(usize, usize)>>,
    spheres: Vec<[f64; 4]>,
    job: horizon_gpu::OreJob,
    /// The world's `floor_writes` when the attempts were predicted.
    floor_writes: u64,
    /// `HORIZON_GPU_ORES=test`: each blob's cells are checked against the CPU's.
    test: bool,
}
impl OrePlan<'_> {
    /// Places the planned feature `id` from its plan; None when it is not planned, or when a
    /// column search no longer answers as predicted (the feature then runs on the CPU: nothing
    /// has been drawn or written).
    fn place(
        &mut self,
        world: &mut World<'_>,
        random: &mut WorldgenRandom<XoroshiroRandom>,
        id: FeatureIdentity,
        ore: &OreFeature,
    ) -> Option<Result<bool, FeatureExecutionError<RegionError>>> {
        let (_, range, size) = self.features.iter().find(|f| f.0 == id)?.clone();
        // Ore writes leave the heightmaps alone: the column searches can all be made first, and
        // answer as predicted when no write has moved a heightmap since.
        let moved = world.floor_writes != self.floor_writes;
        for attempt in self.attempts[range.clone()].iter().filter(|_| moved) {
            if !OreFeature::column_under(world, attempt).is_ok_and(|found| found == attempt.found) {
                self.gpu.note_cpu_fallback();
                return None;
            }
        }
        if !self.gpu.wait(&mut self.job) {
            self.gpu.note_cpu_fallback();
            return None;
        }
        for i in range.clone() {
            let Some((blob, first)) = self.blobs[i] else {
                continue;
            };
            let Some(rows) = self.gpu.rows(&self.job, blob) else {
                self.gpu.note_cpu_fallback();
                return None;
            };
            if self.test
                && OreFeature::blob_rows(&self.spheres[first..first + size], &self.attempts[i])
                    .as_deref()
                    != Some(rows)
            {
                self.gpu.note_mismatch();
                return None;
            }
        }
        let mut placed = false;
        for i in range {
            let Some((blob, _)) = self.blobs[i] else {
                continue;
            };
            let rows = self.gpu.rows(&self.job, blob)?;
            match ore.place_cells(world, random, &self.attempts[i], rows) {
                Ok(p) => placed |= p,
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(placed))
    }
}
/// `HORIZON_GPU_ORES=test`.
fn ores_test() -> bool {
    static TEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TEST.get_or_init(|| std::env::var("HORIZON_GPU_ORES").as_deref() == Ok("test"))
}
/// A placement modifier that reads nothing the chunk's features write (blocks, heightmaps): the
/// positions it gives can be worked out before they run.
fn static_modifier(modifier: &PlacementModifier) -> bool {
    match modifier {
        PlacementModifier::Biome
        | PlacementModifier::Count(_)
        | PlacementModifier::InSquare
        | PlacementModifier::HeightRange(_)
        | PlacementModifier::Rarity(_)
        | PlacementModifier::RandomChance(_)
        | PlacementModifier::Offset(_)
        | PlacementModifier::Fixed(_) => true,
        PlacementModifier::RandomlySelected(list) => list.iter().all(static_modifier),
        _ => false,
    }
}
/// Advances a feature's random stream with no block storage; every write is
/// discarded. Placement modifiers still see the real biome and build bounds.
struct OriginWorld<'a> {
    air: BlockStateId,
    generation: WorldGenerationContext,
    biome_allows: &'a mut dyn FnMut([i32; 3]) -> bool,
}
impl FeatureWorld for OriginWorld<'_> {
    type Error = ();
    fn block(&mut self, _: [i32; 3]) -> Result<BlockStateId, ()> {
        Ok(self.air)
    }
    fn set_block(&mut self, _: [i32; 3], _: BlockStateId, _: i32) -> Result<bool, ()> {
        Ok(false)
    }
}
impl PlacementContext for OriginWorld<'_> {
    fn generation(&self) -> WorldGenerationContext {
        self.generation
    }
    fn min_y(&mut self) -> Result<i32, PlacementQueryError> {
        Ok(self.generation.min_y)
    }
    fn height(&mut self, _: HeightmapType, _: i32, _: i32) -> Result<i32, PlacementQueryError> {
        Ok(self.generation.min_y + self.generation.height)
    }
    fn max_y(&mut self) -> Result<i32, PlacementQueryError> {
        Ok(self.generation.min_y + self.generation.height - 1)
    }
    fn biome_allows_feature(
        &mut self,
        position: [i32; 3],
        _: FeatureIdentity,
    ) -> Result<bool, PlacementQueryError> {
        Ok((self.biome_allows)(position))
    }
}

/// WorldGenRegion-style view: reads anywhere in the 3×3 region, writes within
/// the FEATURES step's write radius, biome lookups through the biome zoom.
struct World<'a> {
    /// The region's full chunks (`ImposterProtoChunk(chunk, false)`).
    full: Vec<ChunkPos>,
    block_entities: Vec<([i32; 3], crate::nbt::Compound)>,
    entities: Vec<crate::nbt::Compound>,
    level_random: &'a mut dyn super::structure::legacy::PieceRandom,
    structure_mobs: Option<Arc<dyn super::generator::StructureMobs>>,
    starts: Vec<Arc<super::structure::start::StructureStart>>,
    biome_names: &'a [Arc<str>],
    blocks: Option<&'a BlockBehavior>,
    region: TerrainRegion<'static, GenBlockStates>,
    biomes: BiomeRegion<BiomeId>,
    zone: GenerationWriteZone,
    center: ChunkPos,
    /// Per biome section of the nine chunks (chunk-major): its one biome id,
    /// MIXED_BIOMES, or UNKNOWN_BIOME until first asked.
    uniform_biomes: Vec<u32>,
    /// The quart Ys biome lookups clamp to (`BiomeChunk::quart_range`).
    biome_quarts: (i32, i32),
    decoration: &'a Decoration,
    temperatures: &'a mut [BiomeTemperature],
    behaviours: Option<&'a crate::level::Behaviours>,
    /// Inclusive minimum and exclusive maximum build height.
    build: (i32, i32),
    /// The tick capture `shape_update` passes to `updateShape`.
    shape_ticks: std::cell::RefCell<crate::level::Ticks>,
    /// `block_matches_tag`'s last tag and its states.
    last_tag: Option<(Box<str>, Option<Arc<BTreeSet<BlockStateId>>>)>,
    /// Per region chunk (X-major in Z rows, as `uniform_biomes`): a bound no
    /// OCEAN_FLOOR_WG first-available height in it exceeds, once asked.
    /// Heightmaps change only through this world's writes, which raise it.
    floor_bounds: [Option<i32>; 9],
    /// Writes that may have moved a heightmap (each one raises `floor_bounds`).
    floor_writes: u64,
    /// Reusable capture storage for shape updates in this decoration region.
    shape_capture: std::cell::RefCell<Vec<crate::level::ticks::CapturedTick>>,
}
/// Read-only `LevelAccess` view of the decoration region for
/// `BlockState.updateShape` (writes go through `World::set_block`).
struct ShapeLevel<'w, 'a> {
    world: &'w World<'a>,
    random: horizon_core::random::LegacyRandom,
}
impl crate::level::LevelAccess for ShapeLevel<'_, '_> {
    fn block_state(&self, pos: BlockPos) -> BlockStateId {
        self.world
            .region
            .block([pos.x, pos.y, pos.z])
            .unwrap_or(BlockStateId(0))
    }
    fn set_raw(&mut self, _: BlockPos, _: BlockStateId) -> Option<BlockStateId> {
        None
    }
    fn is_loaded(&self, pos: BlockPos) -> bool {
        self.world.region.block([pos.x, pos.y, pos.z]).is_ok()
    }
    fn min_y(&self) -> i32 {
        self.world.build.0
    }
    fn max_y(&self) -> i32 {
        self.world.build.1
    }
    fn random(&mut self) -> &mut horizon_core::random::LegacyRandom {
        &mut self.random
    }
    fn game_time(&self) -> i64 {
        0
    }
    /// Lighting has not run during FEATURES: the light engine has no data for
    /// these chunks, so sky light reads zero too (MushroomBlock.canSurvive).
    fn sky_light(&self, _: BlockPos) -> i32 {
        0
    }
}
impl World<'_> {
    /// `BlockState.updateShape` of the state at `pos` against `neighbor`.
    fn shape_update(
        &self,
        behaviours: &crate::level::Behaviours,
        state: BlockStateId,
        pos: [i32; 3],
        direction: Direction,
        neighbor_pos: [i32; 3],
        neighbor: BlockStateId,
    ) -> (BlockStateId, Vec<crate::level::ticks::CapturedTick>) {
        let mut level = ShapeLevel {
            world: self,
            random: horizon_core::random::LegacyRandom::new(0),
        };
        // The ticks updateShape schedules (LeavesBlock re-measuring its
        // distance, waterlogged blocks' fluid) go to the proto chunks. While
        // capturing, scheduling leaves the rest of `Ticks` untouched, so the
        // region keeps one instead of building one per update.
        let mut ticks = self.shape_ticks.borrow_mut();
        ticks.capture = Some(std::mem::take(&mut *self.shape_capture.borrow_mut()));
        let updated = behaviours.update_shape(
            &mut level,
            &mut ticks,
            state,
            BlockPos::new(pos[0], pos[1], pos[2]),
            direction,
            BlockPos::new(neighbor_pos[0], neighbor_pos[1], neighbor_pos[2]),
            neighbor,
        );
        (updated, ticks.capture.take().unwrap_or_default())
    }
    /// Schedule ticks a shape update requested in the region's proto chunks.
    fn schedule_captured(
        &mut self,
        behaviours: &crate::level::Behaviours,
        mut ticks: Vec<crate::level::ticks::CapturedTick>,
    ) -> Result<(), PlacementQueryError> {
        let result = (|| {
            for t in &ticks {
                let at = [t.pos.x, t.pos.y, t.pos.z];
                if t.fluid {
                    if let Some(fluid) = crate::fluid::FluidKind::from_index(t.kind as usize) {
                        self.schedule_fluid_tick(at, fluid, t.delay)?;
                    }
                } else {
                    let state = behaviours
                        .table
                        .default_state(crate::level::BlockId(t.kind as u16));
                    self.schedule_block_tick(at, state, t.delay)?;
                }
            }
            Ok(())
        })();
        ticks.clear();
        *self.shape_capture.borrow_mut() = ticks;
        result
    }
}
impl DecorationWorld for World<'_> {
    type Error = RegionError;
    fn min_section_y(&mut self) -> Result<i32, RegionError> {
        Ok(self
            .region
            .min_y()
            .map_err(|_| RegionError::InvalidChunks)?
            >> 4)
    }
    /// `BiomeRegion::present_in_chunk`: the distinct ids, ascending, from a
    /// bitset instead of sorting every quart of the column.
    fn chunk_biomes(&mut self, chunk: ChunkPos) -> Result<Vec<BiomeId>, RegionError> {
        let biomes = self.biomes.chunk(chunk).map_err(RegionError::Biome)?;
        // Ordinary registry ids fit in the stack array. Keep a spill for
        // unusually large modded registries without changing the ascending
        // iteration order of the bitset.
        let mut seen = [0u64; 128];
        let mut seen_spill = Vec::new();
        let mut last = None;
        let mut section = 0;
        while let Some(values) = biomes.section(section) {
            for &id in values {
                if last == Some(id) {
                    continue;
                }
                last = Some(id);
                let word = id.0 as usize / 64;
                if let Some(bits) = seen.get_mut(word) {
                    *bits |= 1 << (id.0 % 64);
                } else {
                    let spill_word = word - seen.len();
                    if seen_spill.len() <= spill_word {
                        seen_spill.resize(spill_word + 1, 0);
                    }
                    seen_spill[spill_word] |= 1 << (id.0 % 64);
                }
            }
            section += 1;
        }
        let count = seen
            .iter()
            .chain(seen_spill.iter())
            .map(|bits| bits.count_ones() as usize)
            .sum();
        let mut out = Vec::with_capacity(count);
        for (word, &bits) in seen.iter().chain(seen_spill.iter()).enumerate() {
            let mut bits = bits;
            while bits != 0 {
                out.push(BiomeId(word as u32 * 64 + bits.trailing_zeros()));
                bits &= bits - 1;
            }
        }
        Ok(out)
    }
    fn set_currently_generating(&mut self, _: Option<FeatureIdentity>) -> Result<(), RegionError> {
        Ok(())
    }
}
impl World<'_> {
    /// Whether `position` is in one of the region's full chunks.
    fn is_full_chunk(&self, [x, _, z]: [i32; 3]) -> bool {
        self.full.iter().any(|p| p.x == x >> 4 && p.z == z >> 4)
    }
}
impl FeatureWorld for World<'_> {
    type Error = RegionError;
    #[inline]
    fn block(&mut self, position: [i32; 3]) -> Result<BlockStateId, RegionError> {
        self.region.block(position)
    }
    fn scan_column(
        &mut self,
        position: [i32; 3],
        step: i32,
        limit: i32,
        keep: impl FnMut(BlockStateId) -> bool,
    ) -> Result<i32, RegionError> {
        self.region.scan_column(position, step, limit, keep)
    }
    fn set_block(
        &mut self,
        position: [i32; 3],
        state: BlockStateId,
        flags: i32,
    ) -> Result<bool, RegionError> {
        if !self
            .zone
            .contains_block(BlockPos::new(position[0], position[1], position[2]))
        {
            return Ok(false);
        }
        // ImposterProtoChunk(chunk, false): setBlockState, setBlockEntity,
        // removeBlockEntity and markPosForPostProcessing change nothing, and
        // WorldGenRegion.setBlock still answers true.
        if self.is_full_chunk(position) {
            return Ok(true);
        }
        let old = self.region.block(position)?;
        self.region.set_block_worldgen(position, state)?;
        self.raise_floor_bound(position);
        // ProtoChunk keeps a block entity that already exists when the same kind
        // of block is set again (a chest turning waterlogged); other changes drop it.
        let registry = self.decoration.bindings.registry();
        let same_block = old == state
            || registry.state(old).map(|s| s.name.as_str())
                == registry.state(state).map(|s| s.name.as_str());
        if !same_block {
            self.block_entities.retain(|(p, _)| *p != position);
        }
        // WorldGenRegion.setBlock: BlockState.getPostProcessPos unless UPDATE_KNOWN_SHAPE.
        if flags & 16 == 0 {
            let [x, y, z] = position;
            let mark = if self.decoration.postprocess_self.contains(&state) {
                Some(y)
            } else if self.decoration.postprocess_above.contains(&state) {
                Some(y.wrapping_add(1))
            } else {
                None
            };
            if let Some(y) = mark {
                self.region
                    .chunk_mut(x >> 4, z >> 4)?
                    .mark_postprocessing(x, y, z);
            }
        }
        Ok(true)
    }
}
fn query(e: impl std::fmt::Display) -> PlacementQueryError {
    PlacementQueryError::Failed(e.to_string())
}
/// The blocks whose block entity is a `RandomizableContainer`: every
/// `RandomizableContainerBlockEntity` subclass (chest, trapped chest, barrel,
/// dispenser, dropper, hopper, shulker box, crafter) and the decorated pot, which
/// implements the interface on a plain `BlockEntity`. `RandomizableContainer`'s
/// `setBlockEntityLootTable` hands a loot table to those and to nothing else.
const RANDOMIZABLE_CONTAINERS: [&str; 25] = [
    "minecraft:chest",
    "minecraft:trapped_chest",
    "minecraft:barrel",
    "minecraft:dispenser",
    "minecraft:dropper",
    "minecraft:hopper",
    "minecraft:shulker_box",
    "minecraft:white_shulker_box",
    "minecraft:orange_shulker_box",
    "minecraft:magenta_shulker_box",
    "minecraft:light_blue_shulker_box",
    "minecraft:yellow_shulker_box",
    "minecraft:lime_shulker_box",
    "minecraft:pink_shulker_box",
    "minecraft:gray_shulker_box",
    "minecraft:light_gray_shulker_box",
    "minecraft:cyan_shulker_box",
    "minecraft:purple_shulker_box",
    "minecraft:blue_shulker_box",
    "minecraft:brown_shulker_box",
    "minecraft:green_shulker_box",
    "minecraft:red_shulker_box",
    "minecraft:black_shulker_box",
    "minecraft:crafter",
    "minecraft:decorated_pot",
];
/// Whether the block entity of a block of this name is a `RandomizableContainer`.
/// Every worldgen caller places a chest today, so the wider set is defensive.
fn randomizable_container(name: &str) -> bool {
    RANDOMIZABLE_CONTAINERS.contains(&name)
}

/// `RandomizableContainer.setBlockEntityLootTable` for a world that keeps block
/// entities as NBT: the compound a randomizable container's block entity takes, or
/// None when the block there is not one. The seed is drawn only where vanilla draws
/// it, inside the `instanceof RandomizableContainer` branch.
fn loot_table_nbt(
    registry: &crate::registry::BlockRegistry,
    state: BlockStateId,
    table: &str,
    random: &mut impl horizon_core::random::RandomSource,
) -> Option<crate::nbt::Compound> {
    use crate::nbt::Tag;
    let name = registry.state(state)?.name.to_string();
    if !randomizable_container(&name) {
        return None;
    }
    let mut nbt = crate::nbt::Compound::new();
    nbt.insert("id".into(), Tag::String(name.as_str().into()));
    nbt.insert("LootTable".into(), Tag::String(table.into()));
    nbt.insert("LootTableSeed".into(), Tag::Long(random.next_long()));
    Some(nbt)
}
impl<'a> World<'a> {
    fn blocks(&self) -> Result<&'a BlockBehavior, PlacementQueryError> {
        self.blocks.ok_or(PlacementQueryError::Unavailable)
    }
}
/// Proto chunks hold no block entities, so dynamic shapes resolve to absent.
impl DynamicSupportWorld for World<'_> {
    type Error = PlacementQueryError;
    fn shulker_is_closed(&mut self, _: [i32; 3]) -> Result<Option<bool>, PlacementQueryError> {
        Ok(None)
    }
    fn moving_piston_shape(
        &mut self,
        _: [i32; 3],
    ) -> Result<Option<VoxelShape>, PlacementQueryError> {
        Ok(None)
    }
}
impl DynamicCollisionWorld for World<'_> {
    fn shulker_collision_state(
        &mut self,
        _: [i32; 3],
    ) -> Result<Option<ShulkerCollisionState>, PlacementQueryError> {
        Ok(None)
    }
}
impl SurvivalWorld for World<'_> {
    fn fluid_state(&mut self, position: [i32; 3]) -> Result<FluidState, PlacementQueryError> {
        self.region.fluid_state(position)
    }
    /// Unlit generation chunks report no sky or block light.
    fn raw_brightness(&mut self, _: [i32; 3], _: i32) -> Result<i32, PlacementQueryError> {
        Ok(0)
    }
    fn block_state(&mut self, position: [i32; 3]) -> Result<BlockStateId, PlacementQueryError> {
        self.region.block_state(position)
    }
}
impl PlacementContext for World<'_> {
    fn state_collision_empty(
        &mut self,
        state: BlockStateId,
        position: [i32; 3],
    ) -> Result<bool, PlacementQueryError> {
        if let Some(facts) = self.blocks()?.collision.fixed_facts(state) {
            return Ok(facts & crate::block_collision::FIXED_EMPTY != 0);
        }
        let collision = &self.blocks()?.collision;
        collision
            .collision_shape(self, state, position)
            .map(|shape| shape.is_empty())
            .map_err(query)
    }
    /// RandomizableContainer.setBlockEntityLootTable: only a container block that
    /// is actually there takes the table, and then it draws the loot seed.
    fn set_block_entity_loot_table(
        &mut self,
        position: [i32; 3],
        table: &str,
        random: &mut impl horizon_core::random::RandomSource,
    ) -> Result<(), PlacementQueryError> {
        let chunk = self.region.chunk(position[0] >> 4, position[2] >> 4)?;
        let state = chunk.block(position[0], position[1], position[2]);
        let registry = self.decoration.bindings.registry();
        let Some(nbt) = loot_table_nbt(registry, state, table, random) else {
            return Ok(());
        };
        self.set_block_entity(position, nbt)
    }
    /// BeehiveDecorator: `getBlockEntity(pos, BEEHIVE)` then
    /// `storeBee(Occupant.create(ticks))` for each bee.
    fn store_beehive_occupants(
        &mut self,
        position: [i32; 3],
        ticks_in_hive: &[i32],
    ) -> Result<(), PlacementQueryError> {
        let chunk = self.region.chunk(position[0] >> 4, position[2] >> 4)?;
        let state = chunk.block(position[0], position[1], position[2]);
        let registry = self.decoration.bindings.registry();
        if !matches!(
            registry.state(state).map(|s| s.name.as_str()),
            Some("minecraft:bee_nest" | "minecraft:beehive")
        ) {
            return Ok(());
        }
        self.set_block_entity(position, super::tree::beehive_occupants(ticks_in_hive))
    }
    /// SpawnerBlockEntity.setEntityId: the next spawn data and no potentials.
    fn configure_spawner(
        &mut self,
        position: [i32; 3],
        entity: &str,
    ) -> Result<(), PlacementQueryError> {
        let chunk = self.region.chunk(position[0] >> 4, position[2] >> 4)?;
        let state = chunk.block(position[0], position[1], position[2]);
        let registry = self.decoration.bindings.registry();
        if registry.state(state).map(|s| s.name.as_str()) != Some("minecraft:spawner") {
            return Ok(());
        }
        self.set_block_entity(position, super::cave::dungeon_spawner_nbt(entity))
    }
    fn attachment_at(
        &mut self,
        position: [i32; 3],
        direction: Direction,
    ) -> Result<bool, PlacementQueryError> {
        let blocks = self.blocks()?;
        blocks
            .survival
            .can_attach_to(self, position, direction)
            .map_err(query)
    }
    fn next_world_boolean(&mut self) -> Result<bool, PlacementQueryError> {
        Ok(self.level_random.boolean())
    }
    fn next_world_float(&mut self) -> Result<f32, PlacementQueryError> {
        Ok(self.level_random.float())
    }
    fn next_world_int(&mut self, bound: i32) -> Result<i32, PlacementQueryError> {
        Ok(self.level_random.int(bound))
    }
    fn block_survives(
        &mut self,
        state: BlockStateId,
        position: [i32; 3],
    ) -> Result<bool, PlacementQueryError> {
        if let (None, Some(behaviours)) = (self.blocks, self.behaviours) {
            let mut level = ShapeLevel {
                world: self,
                random: horizon_core::random::LegacyRandom::new(0),
            };
            let mut ticks = crate::level::Ticks::new(0);
            let pos = BlockPos::new(position[0], position[1], position[2]);
            return Ok(behaviours.can_survive(&mut level, &mut ticks, pos, state));
        }
        let blocks = self.blocks()?;
        blocks
            .survival
            .can_survive(self, state, position)
            .map_err(query)
    }
    fn face_sturdy(
        &mut self,
        position: [i32; 3],
        direction: Direction,
    ) -> Result<bool, PlacementQueryError> {
        let state = self.region.block_state(position)?;
        self.state_face_sturdy(state, position, direction)
    }
    fn state_face_sturdy(
        &mut self,
        state: BlockStateId,
        position: [i32; 3],
        direction: Direction,
    ) -> Result<bool, PlacementQueryError> {
        // Without the exported support tables, fall back to the gameplay
        // behaviour's empty-level sturdiness cache.
        if let (None, Some(behaviours)) = (self.blocks, self.behaviours) {
            return Ok(behaviours.is_face_sturdy(state, direction));
        }
        let support = &self.blocks()?.support;
        support
            .is_face_sturdy(self, state, position, direction, SupportType::Full)
            .map_err(query)
    }
    fn center_sturdy(
        &mut self,
        position: [i32; 3],
        direction: Direction,
    ) -> Result<bool, PlacementQueryError> {
        let state = self.region.block_state(position)?;
        if let (None, Some(behaviours)) = (self.blocks, self.behaviours) {
            return Ok(behaviours.is_face_sturdy_center(state, direction));
        }
        let support = &self.blocks()?.support;
        support
            .is_face_sturdy(self, state, position, direction, SupportType::Center)
            .map_err(query)
    }
    fn can_occlude(&mut self, state: BlockStateId) -> Result<bool, PlacementQueryError> {
        if let Some(behaviours) = self.behaviours {
            return Ok(behaviours.has_flag(state, crate::level::data::state_flags::CAN_OCCLUDE));
        }
        let properties = &self.blocks()?.properties;
        let name = properties
            .registry()
            .state(state)
            .ok_or(PlacementQueryError::Unavailable)?
            .name
            .clone();
        Ok(properties.definition(&name).is_some_and(|d| d.can_occlude))
    }
    fn collision_full_block(
        &mut self,
        state: BlockStateId,
        position: [i32; 3],
    ) -> Result<bool, PlacementQueryError> {
        if let Some(facts) = self.blocks()?.collision.fixed_facts(state) {
            return Ok(facts & crate::block_collision::FIXED_FULL_BLOCK != 0);
        }
        let collision = &self.blocks()?.collision;
        let shape = collision
            .collision_shape(self, state, position)
            .map_err(query)?;
        Ok(crate::block_collision::shape_fills_block(&shape))
    }
    fn add_entity(&mut self, mut nbt: crate::nbt::Compound) -> Result<(), PlacementQueryError> {
        if let Some(mobs) = self.structure_mobs.clone()
            && nbt.contains_key(&"horizon:finalize".into())
        {
            let at = block_of(&nbt).unwrap_or([0; 3]);
            let biome = self
                .biome_identity(at)
                .ok()
                .and_then(|b| self.biome_names.get(b.0 as usize))
                .map_or("", |n| &**n)
                .to_owned();
            let structures: Vec<String> = self
                .starts
                .iter()
                .filter(|s| s.pieces.iter().any(|p| p.bounding_box().is_inside(at)))
                .map(|s| s.name.clone())
                .collect();
            if mobs.finalize(&mut nbt, &biome, &structures, self.level_random) {
                nbt.remove(&"horizon:finalize".into());
            }
        }
        self.entities.push(nbt);
        Ok(())
    }
    /// SpikeFeature.placeSpike: the crystal is a fresh entity of the region,
    /// saved with the chunk.
    fn spawn_end_crystal(
        &mut self,
        position: [f64; 3],
        yaw: f32,
        beam_target: Option<[i32; 3]>,
        invulnerable: bool,
    ) -> Result<(), PlacementQueryError> {
        use crate::nbt::Tag;
        let mut nbt = crate::nbt::Compound::new();
        nbt.insert("id".into(), Tag::String("minecraft:end_crystal".into()));
        nbt.insert(
            "Pos".into(),
            Tag::List(position.iter().map(|v| Tag::Double(v.to_bits())).collect()),
        );
        nbt.insert(
            "Rotation".into(),
            Tag::List(vec![Tag::Float(yaw.to_bits()), Tag::Float(0)]),
        );
        nbt.insert("Invulnerable".into(), Tag::Byte(i8::from(invulnerable)));
        nbt.insert("ShowBottom".into(), Tag::Byte(1));
        if let Some(b) = beam_target {
            nbt.insert("beam_target".into(), Tag::IntArray(b.to_vec()));
        }
        self.add_entity(nbt)
    }
    fn set_block_entity(
        &mut self,
        position: [i32; 3],
        nbt: crate::nbt::Compound,
    ) -> Result<(), PlacementQueryError> {
        self.block_entities.retain(|(p, _)| *p != position);
        self.block_entities.push((position, nbt));
        Ok(())
    }
    /// WorldGenRegion.getBlockEntity creates the gateway's block entity when the
    /// block is an End gateway; TheEndGatewayBlockEntity.setExitPosition.
    fn configure_end_gateway(
        &mut self,
        position: [i32; 3],
        exit: [i32; 3],
        exact: bool,
    ) -> Result<(), PlacementQueryError> {
        let chunk = self.region.chunk(position[0] >> 4, position[2] >> 4)?;
        let state = chunk.block(position[0], position[1], position[2]);
        let registry = self.decoration.bindings.registry();
        if registry.state(state).map(|s| s.name.as_str()) != Some("minecraft:end_gateway") {
            return Ok(());
        }
        use crate::nbt::Tag;
        let mut nbt = crate::nbt::Compound::new();
        nbt.insert("id".into(), Tag::String("minecraft:end_gateway".into()));
        nbt.insert("Age".into(), Tag::Long(0));
        nbt.insert("exit_portal".into(), Tag::IntArray(exit.to_vec()));
        if exact {
            nbt.insert("ExactTeleport".into(), Tag::Byte(1));
        }
        self.set_block_entity(position, nbt)
    }
    fn world_seed(&mut self) -> Result<i64, PlacementQueryError> {
        self.region.world_seed()
    }
    fn update_shape_at_face(
        &mut self,
        position: [i32; 3],
        direction: Direction,
        flags: i32,
    ) -> Result<(), PlacementQueryError> {
        let Some(behaviours) = self.behaviours else {
            return Ok(());
        };
        let n = direction.normal();
        let neighbor_pos = [position[0] + n[0], position[1] + n[1], position[2] + n[2]];
        let state = self.region.block(position).map_err(query)?;
        let neighbor = self.region.block(neighbor_pos).map_err(query)?;
        let (new_state, ticks) = self.shape_update(
            behaviours,
            state,
            position,
            direction,
            neighbor_pos,
            neighbor,
        );
        self.schedule_captured(behaviours, ticks)?;
        if new_state != state {
            self.set_block(position, new_state, flags & !1)
                .map_err(query)?;
        }
        let (new_neighbor, ticks) = self.shape_update(
            behaviours,
            neighbor,
            neighbor_pos,
            direction.opposite(),
            position,
            new_state,
        );
        self.schedule_captured(behaviours, ticks)?;
        if new_neighbor != neighbor {
            self.set_block(neighbor_pos, new_neighbor, flags & !1)
                .map_err(query)?;
        }
        Ok(())
    }
    fn update_from_neighbour_shapes(
        &mut self,
        position: [i32; 3],
        flags: i32,
    ) -> Result<(), PlacementQueryError> {
        let Some(behaviours) = self.behaviours else {
            return Ok(());
        };
        let state = self.region.block(position).map_err(query)?;
        // Block.UPDATE_SHAPE_ORDER.
        let mut new_state = state;
        for direction in [
            Direction::West,
            Direction::East,
            Direction::North,
            Direction::South,
            Direction::Down,
            Direction::Up,
        ] {
            let n = direction.normal();
            let neighbor_pos = [position[0] + n[0], position[1] + n[1], position[2] + n[2]];
            let neighbor = self.region.block(neighbor_pos).map_err(query)?;
            let (updated, ticks) = self.shape_update(
                behaviours,
                new_state,
                position,
                direction,
                neighbor_pos,
                neighbor,
            );
            self.schedule_captured(behaviours, ticks)?;
            new_state = updated;
        }
        if new_state != state {
            self.set_block(position, new_state, flags).map_err(query)?;
        }
        Ok(())
    }
    fn ensure_can_write(&mut self, position: [i32; 3]) -> Result<bool, PlacementQueryError> {
        Ok(self
            .zone
            .contains_block(BlockPos::new(position[0], position[1], position[2])))
    }
    /// ProtoChunk.setBlockState: the section and the stage's heightmaps
    /// (nothing on a full chunk: ImposterProtoChunk(chunk, false)).
    fn set_chunk_block(
        &mut self,
        position: [i32; 3],
        state: BlockStateId,
    ) -> Result<(), PlacementQueryError> {
        if self.is_full_chunk(position) {
            return Ok(());
        }
        self.region
            .set_block_worldgen(position, state)
            .map_err(|e| query(format!("{e:?}")))?;
        self.raise_floor_bound(position);
        Ok(())
    }
    /// BulkSectionAccess write: section storage only, no heightmap updates.
    fn set_section_block(
        &mut self,
        [x, y, z]: [i32; 3],
        state: BlockStateId,
    ) -> Result<(), PlacementQueryError> {
        self.region
            .chunk_mut(x >> 4, z >> 4)?
            .set_block_raw(x, y, z, state)
            .map_err(|e| query(format!("{e:?}")))?;
        Ok(())
    }
    /// The trait's per-cell `replace_row`, one chunk's part of the row at a time: the write zone
    /// holds whole chunks, and the build height is the row's.
    fn replace_row(
        &mut self,
        [x, y, z]: [i32; 3],
        bits: u32,
        cell: &mut dyn FnMut([i32; 3], BlockStateId) -> Option<BlockStateId>,
    ) -> Result<bool, PlacementQueryError> {
        if self.outside_build_height(y)? {
            return Ok(false);
        }
        let mut wrote = false;
        let mut rest = bits;
        while rest != 0 {
            let first = z.wrapping_add(rest.trailing_zeros() as i32);
            // The bits up to the last cell of `first`'s chunk
            let last = (first | 15).wrapping_sub(z);
            let part = if last >= 31 {
                rest
            } else {
                rest & ((2u32 << last) - 1)
            };
            rest &= !part;
            if !self.zone.contains_block(BlockPos::new(x, y, first)) {
                continue;
            }
            wrote |= self
                .region
                .chunk_mut(x >> 4, first >> 4)?
                .replace_row_raw([x, y, z], part, |zz, state| cell([x, y, zz], state))
                .map_err(|e| query(format!("{e:?}")))?;
        }
        Ok(wrote)
    }
    fn replace_row_states(
        &mut self,
        [x, y, z]: [i32; 3],
        bits: u32,
        cell: &mut dyn FnMut(BlockStateId) -> Option<BlockStateId>,
    ) -> Result<bool, PlacementQueryError> {
        if self.outside_build_height(y)? {
            return Ok(false);
        }
        let mut wrote = false;
        let mut rest = bits;
        while rest != 0 {
            let first = z.wrapping_add(rest.trailing_zeros() as i32);
            // The bits up to the last cell of `first`'s chunk
            let last = (first | 15).wrapping_sub(z);
            let part = if last >= 31 {
                rest
            } else {
                rest & ((2u32 << last) - 1)
            };
            rest &= !part;
            if !self.zone.contains_block(BlockPos::new(x, y, first)) {
                continue;
            }
            wrote |= self
                .region
                .chunk_mut(x >> 4, first >> 4)?
                .replace_row_states([x, y, z], part, &mut *cell)
                .map_err(|e| query(format!("{e:?}")))?;
        }
        Ok(wrote)
    }
    /// Generation regions contain no entities.
    fn unobstructed_block(&mut self, _: [i32; 3]) -> Result<bool, PlacementQueryError> {
        Ok(true)
    }
    fn destroy_block_without_entity(
        &mut self,
        position: [i32; 3],
        _drop_resources: bool,
    ) -> Result<bool, PlacementQueryError> {
        let chunk = self.region.chunk(position[0] >> 4, position[2] >> 4)?;
        let state = chunk.block(position[0], position[1], position[2]);
        if chunk.block_properties(state).is_air() {
            return Ok(false);
        }
        let air = self
            .decoration
            .bindings
            .registry()
            .default_state("minecraft:air")
            .ok_or(PlacementQueryError::Unavailable)?;
        self.set_block(position, air, 3).map_err(query)
    }
    fn biome_should_freeze(
        &mut self,
        biome: BiomeId,
        position: [i32; 3],
        require_edge: bool,
    ) -> Result<bool, PlacementQueryError> {
        // The bindings live as long as the decoration: no clone of their shared sets per call.
        let decoration = self.decoration;
        let weather = decoration
            .weather
            .as_ref()
            .ok_or(PlacementQueryError::Unavailable)?;
        // A pure function of the position within the generator: computed, not cached (the
        // queries of a column do not come back).
        let mut temperature =
            BiomeTemperature::uncached(self.temperatures[biome.0 as usize].settings());
        weather.should_freeze(self, &mut temperature, position, require_edge)
    }
    fn biome_should_snow(
        &mut self,
        biome: BiomeId,
        position: [i32; 3],
    ) -> Result<bool, PlacementQueryError> {
        // The bindings live as long as the decoration: no clone of their shared sets per call.
        let decoration = self.decoration;
        let weather = decoration
            .weather
            .as_ref()
            .ok_or(PlacementQueryError::Unavailable)?;
        // A pure function of the position within the generator: computed, not cached (the
        // queries of a column do not come back).
        let mut temperature =
            BiomeTemperature::uncached(self.temperatures[biome.0 as usize].settings());
        weather.should_snow(self, &mut temperature, position)
    }
    /// Lighting has not run during FEATURES; block light reads zero.
    fn block_brightness(&mut self, _: [i32; 3]) -> Result<i32, PlacementQueryError> {
        Ok(0)
    }
    fn fluid_state(&mut self, position: [i32; 3]) -> Result<FluidState, PlacementQueryError> {
        self.region.fluid_state(position)
    }
    fn sea_level(&mut self) -> Result<i32, PlacementQueryError> {
        self.region.sea_level()
    }
    fn schedule_fluid_tick(
        &mut self,
        position: [i32; 3],
        fluid: FluidKind,
        delay: i32,
    ) -> Result<(), PlacementQueryError> {
        self.region.schedule_fluid_tick(position, fluid, delay)
    }
    fn mark_postprocessing(&mut self, position: [i32; 3]) -> Result<(), PlacementQueryError> {
        self.region.mark_postprocessing(position)
    }
    fn water_at(&mut self, position: [i32; 3]) -> Result<bool, PlacementQueryError> {
        self.region.water_at(position)
    }
    fn schedule_block_tick(
        &mut self,
        position: [i32; 3],
        block: BlockStateId,
        delay: i32,
    ) -> Result<(), PlacementQueryError> {
        self.region.schedule_block_tick(position, block, delay)
    }
    fn biome_identity(&mut self, position: [i32; 3]) -> Result<BiomeId, PlacementQueryError> {
        if let Some(biome) = self.uniform_biome(position) {
            return Ok(biome);
        }
        self.region
            .biome_at(
                &self.biomes,
                BlockPos::new(position[0], position[1], position[2]),
            )
            .copied()
            .map_err(Into::into)
    }
    fn biome_allows_feature(
        &mut self,
        position: [i32; 3],
        feature: FeatureIdentity,
    ) -> Result<bool, PlacementQueryError> {
        let biome = self.biome_identity(position)?;
        let bits = &self.decoration.biome_features[biome.0 as usize];
        let word = feature.0 as usize / 64;
        Ok(bits
            .get(word)
            .is_some_and(|w| w >> (feature.0 % 64) & 1 != 0))
    }
    fn block_state(&mut self, position: [i32; 3]) -> Result<BlockStateId, PlacementQueryError> {
        self.region.block_state(position)
    }
    /// TerrainRegion::block_matches_tag (the region holds the same tag
    /// table), with the last tag's states kept: predicates ask one tag for
    /// many positions (environment scans), and the table is keyed by name.
    fn block_matches_tag(
        &mut self,
        position: [i32; 3],
        tag: &str,
    ) -> Result<bool, PlacementQueryError> {
        let state = self.region.block(position)?;
        if self
            .last_tag
            .as_ref()
            .is_none_or(|(name, _)| **name != *tag)
        {
            let states = self.decoration.bindings.tags.get(tag).cloned();
            self.last_tag = Some((tag.into(), states));
        }
        let states = self
            .last_tag
            .as_ref()
            .and_then(|(_, states)| states.as_ref());
        Ok(states.is_some_and(|states| states.contains(&state)))
    }
    fn max_y(&mut self) -> Result<i32, PlacementQueryError> {
        self.region.max_y()
    }
    fn layer_block(&mut self, position: [i32; 3]) -> Result<LayerBlock, PlacementQueryError> {
        self.region.layer_block(position)
    }
    fn generation(&self) -> WorldGenerationContext {
        self.region.generation()
    }
    fn min_y(&mut self) -> Result<i32, PlacementQueryError> {
        self.region.min_y()
    }
    fn height(&mut self, kind: HeightmapType, x: i32, z: i32) -> Result<i32, PlacementQueryError> {
        self.region.height(kind, x, z)
    }
    /// OreFeature's search for a column under its blob, without a trait call
    /// per column. Defers to `height` when a heightmap still has to be primed.
    fn any_height_at_least(
        &mut self,
        kind: HeightmapType,
        [x0, z0]: [i32; 2],
        size: i32,
        y: i32,
    ) -> Result<Option<bool>, PlacementQueryError> {
        let (x1, z1) = (x0.wrapping_add(size), z0.wrapping_add(size));
        if size < 0 || x1 < x0 || z1 < z0 {
            return Ok(None);
        }
        // Most ore attempts in To The Max's tall range sit above every
        // column: answered from the chunks' bounds without reading columns.
        if kind == HeightmapType::OceanFloorWg {
            let mut below = true;
            for cx in x0 >> 4..=x1 >> 4 {
                for cz in z0 >> 4..=z1 >> 4 {
                    below &= self.floor_bound(cx, cz).is_some_and(|bound| bound < y);
                }
            }
            if below {
                return Ok(Some(false));
            }
        }
        for cx in x0 >> 4..=x1 >> 4 {
            for cz in z0 >> 4..=z1 >> 4 {
                if !self
                    .region
                    .chunk(cx, cz)
                    .is_ok_and(|c| c.heightmap(kind).is_some())
                {
                    return Ok(None);
                }
            }
        }
        // Any column will do, so each chunk's part of the square is read
        // from its heightmap in one go.
        for cx in x0 >> 4..=x1 >> 4 {
            let xs = [x0.max(cx << 4) & 15, x1.min((cx << 4) + 15) & 15];
            for cz in z0 >> 4..=z1 >> 4 {
                let zs = [z0.max(cz << 4) & 15, z1.min((cz << 4) + 15) & 15];
                let map = self.region.chunk(cx, cz)?.heightmap(kind).unwrap();
                if map.any_first_available_at_least(xs, zs, y) {
                    return Ok(Some(true));
                }
            }
        }
        Ok(Some(false))
    }
}

/// The block an entity's `Pos` is in.
fn block_of(nbt: &crate::nbt::Compound) -> Option<[i32; 3]> {
    let Some(crate::nbt::Tag::List(p)) = nbt.get(&"Pos".into()) else {
        return None;
    };
    Some(std::array::from_fn(|i| match p.get(i) {
        Some(crate::nbt::Tag::Double(b)) => f64::from_bits(*b).floor() as i32,
        _ => 0,
    }))
}

/// Marks `feature` in a biome's feature bitset (one bit per feature identity).
fn set_feature_bit(bits: &mut Vec<u64>, feature: FeatureIdentity) {
    let word = feature.0 as usize / 64;
    if bits.len() <= word {
        bits.resize(word + 1, 0);
    }
    bits[word] |= 1 << (feature.0 % 64);
}
#[cfg(test)]
mod tests {
    use super::loot_table_nbt;
    use crate::registry::BlockRegistry;
    use horizon_core::random::{LegacyRandom, RandomSource};

    /// Air, a chest, a dispenser and a furnace: one randomizable container the
    /// worldgen callers place, three they do not, and one plain container.
    fn registry() -> &'static BlockRegistry {
        let names = [
            "minecraft:air",
            "minecraft:chest",
            "minecraft:dispenser",
            "minecraft:hopper",
            "minecraft:shulker_box",
            "minecraft:light_gray_shulker_box",
            "minecraft:crafter",
            "minecraft:decorated_pot",
            "minecraft:furnace",
            "minecraft:ender_chest",
        ];
        let rows: Vec<serde_json::Value> = names
            .iter()
            .map(|name| {
                serde_json::json!([name, name.ends_with("air"), false, "",
                    {"id": name, "properties": {}}])
            })
            .collect();
        let defaults: serde_json::Map<String, serde_json::Value> = (names.iter().enumerate())
            .map(|(i, name)| ((*name).to_owned(), serde_json::json!(i)))
            .collect();
        let facts = serde_json::json!({"states": rows, "defaults": defaults});
        let registry = BlockRegistry::from_facts_json(facts.to_string().as_bytes()).unwrap();
        Box::leak(Box::new(registry))
    }

    fn state(name: &str) -> crate::registry::BlockStateId {
        registry().default_state(name).unwrap()
    }

    /// `RandomizableContainer.setBlockEntityLootTable` hands the table to the block
    /// entity of any `RandomizableContainer`, not only the chests the two worldgen
    /// callers place today.
    #[test]
    fn a_randomizable_container_takes_a_loot_table() {
        for name in [
            "minecraft:chest",
            "minecraft:dispenser",
            "minecraft:hopper",
            "minecraft:shulker_box",
            "minecraft:light_gray_shulker_box",
            "minecraft:crafter",
            "minecraft:decorated_pot",
        ] {
            let mut random = LegacyRandom::new(4);
            let nbt = loot_table_nbt(
                registry(),
                state(name),
                "minecraft:chests/simple_dungeon",
                &mut random,
            )
            .unwrap_or_else(|| panic!("no loot table for {name}"));
            assert_eq!(
                nbt.get(&"id".into()),
                Some(&crate::nbt::Tag::String(name.into()))
            );
            assert_eq!(
                nbt.get(&"LootTable".into()),
                Some(&crate::nbt::Tag::String(
                    "minecraft:chests/simple_dungeon".into()
                ))
            );
            // The seed is the first long of the random: same seed, same draw.
            assert_eq!(
                nbt.get(&"LootTableSeed".into()),
                Some(&crate::nbt::Tag::Long(LegacyRandom::new(4).next_long()))
            );
        }
    }

    /// A block whose block entity is not a `RandomizableContainer` keeps no table,
    /// and the seed is not drawn for it.
    #[test]
    fn a_plain_container_takes_no_loot_table() {
        for name in [
            "minecraft:furnace",
            "minecraft:ender_chest",
            "minecraft:air",
        ] {
            let mut random = LegacyRandom::new(4);
            assert!(
                loot_table_nbt(
                    registry(),
                    state(name),
                    "minecraft:chests/simple_dungeon",
                    &mut random
                )
                .is_none(),
                "{name}"
            );
            assert_eq!(
                random.next_long(),
                LegacyRandom::new(4).next_long(),
                "{name}"
            );
        }
    }
}
