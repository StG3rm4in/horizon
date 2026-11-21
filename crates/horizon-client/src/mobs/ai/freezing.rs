//! `LivingEntity.canFreeze` and the powder-snow test behind it.
use super::{Mob, MobWorld};
use horizon_core::geometry::Vec3;

pub(crate) fn in_powder_snow(m: &Mob, w: &MobWorld) -> bool {
    let b = m.bounds().deflate(Vec3::new(1e-5, 1e-5, 1e-5));
    let (min, max) = (b.min_position(), b.max_position());
    let lo = [min.x, min.y, min.z].map(|v| v.floor() as i32);
    let hi = [max.x, max.y, max.z].map(|v| v.floor() as i32);
    // No section of the box may hold powder snow: no cell of it has.
    if !super::may_contact(w, lo, hi, super::CONTACT_POWDER_SNOW) {
        return false;
    }
    for x in lo[0]..=hi[0] {
        for y in lo[1]..=hi[1] {
            for z in lo[2]..=hi[2] {
                let state = w.world.block(x, y, z);
                if super::contact_bits(w, state) & super::CONTACT_POWDER_SNOW != 0 {
                    return true;
                }
            }
        }
    }
    false
}

/// `LivingEntity.canFreeze`: not a `#freeze_immune_entity_types` mob and no
/// worn piece in `#freeze_immune_wearables`.
pub fn can_freeze(m: &Mob, w: &MobWorld) -> bool {
    // Skeleton.canFreeze: a skeleton never freezes, so powder snow builds no
    // frozen ticks on it and deals it no freeze damage. Only `Skeleton`
    // overrides it: the bogged, parched and wither skeleton extend
    // `AbstractSkeleton` and freeze; the stray is immune through the
    // `#freeze_immune_entity_types` tag below.
    if m.species() == super::Species::Skeleton {
        return false;
    }
    // SulfurCube.canFreeze: a cube carrying a block in its body does not freeze
    // either; without one it takes the tags below like any other mob.
    if m.species() == super::Species::SulfurCube && crate::mobs::sulfur_cube::has_body(m) {
        return false;
    }
    if w.data
        .entity_has("minecraft:freeze_immune_entity_types", &m.kind.id)
    {
        return false;
    }
    let immune = w
        .items
        .and_then(|i| i.item_tags.get("minecraft:freeze_immune_wearables"));
    // A wolf's armor (its body slot) is `#freeze_immune_wearables`.
    ![
        m.extra.body_id(),
        m.extra.head_id(),
        m.extra.slot_id(horizon_items::equipment::CHEST),
        m.extra.slot_id(horizon_items::equipment::LEGS),
        m.extra.slot_id(horizon_items::equipment::FEET),
    ]
    .into_iter()
    .flatten()
    .any(|item| immune.is_some_and(|t| t.contains(item)))
}

/// The player's HEAD item belongs to vanilla's gaze disguise tag.
pub fn wears_gaze_disguise(head: Option<&str>, w: &MobWorld) -> bool {
    head.is_some_and(|item| {
        w.items
            .and_then(|items| items.item_tags.get("minecraft:gaze_disguise_equipment"))
            .is_some_and(|tag| tag.contains(item))
    })
}
