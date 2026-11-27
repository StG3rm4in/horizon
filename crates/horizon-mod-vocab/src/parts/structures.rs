use super::*;
/// The structure types registered at startup, for the executable to hand to
/// the engine's `StructureRegistry`.
pub(crate) fn installed_structure_type_list()
-> &'static [crate::world::structure_type::StructureTypeInfo] {
    installed().map_or(&[], |installed| installed.structure_types.as_slice())
}

pub(crate) fn installed_structure_placement_list()
-> &'static [crate::world::structure_type::StructurePlacementInfo] {
    installed().map_or(&[], |installed| installed.structure_placements.as_slice())
}

impl dyn Registrar + '_ {
    /// Register a finite placement under an owned id (`site` or `mymod:site`).
    /// A structure set can select it with `"placement": {"type": "mymod:site", ...}`.
    /// Duplicate and invalid ids are refused at startup.
    pub fn register_structure_placement<T: crate::ModStructurePlacement>(
        &mut self,
        name: &str,
        placement: T,
    ) {
        self.add_structure_placement(name, Arc::new(placement));
    }

    /// Register structure type `name` (`tower` or `mymod:tower`), the way
    /// NeoForge's registries `STRUCTURE_TYPE` and `STRUCTURE_PIECE` register
    /// once at startup: a data pack structure with `"type": "modid:tower"`
    /// then generates by `ty`'s code — its own start and pieces, placed by the
    /// game's own spacing, biome and terrain-adaptation checks. The type's id
    /// is in a namespace owned by the calling mod's manifest (`mymod:tower`); another mod's
    /// namespace, an empty name and an id registered twice refuse the mods at
    /// startup, naming the mod. `ty`'s `load_piece` is its saved pieces'
    /// loader: keep the type for as long as the game runs.
    pub fn register_structure_type<T: crate::StructureType>(&mut self, name: &str, ty: T) {
        self.add_structure_type(name, Arc::new(ty));
    }
}

/// Registrar operations for structures.
pub trait StructuresRegistrar {
    /// Register native finite placement `name` with `placement`; mods reach it through
    /// `register_structure_placement`.
    ///
    /// Not on wasm: ModStructurePlacement is not bound by the generated worker interface. A wasm
    /// mod's structure set can name a placement a native mod registered.
    fn add_structure_placement(
        &mut self,
        name: &str,
        placement: Arc<dyn crate::ModStructurePlacement>,
    ) {
        let _ = placement;
        panic!("structure placement {name:?}: a wasm mod cannot register a structure placement");
    }

    /// Register structure type `name` with `ty`; mods reach it through
    /// `register_structure_type`.
    ///
    /// On WebAssembly, worker calls run under the linked group's store lock and call budget.
    fn add_structure_type(&mut self, name: &str, ty: Arc<dyn crate::StructureType>);
}
