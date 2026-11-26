//! Generates the native mod API, and gives the Windows executable Horizon's icon and its stack
//! (`packaging/windows/exe.rs`, shared with the official executable's build script).
#[path = "../../packaging/windows/exe.rs"]
mod windows_exe;

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let root = std::path::Path::new(&dir).join("../..");
    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("crates/horizon-client/src/game").display()
    );
    std::fs::write(
        out_dir.join("game_modules.rs"),
        horizon_game_discovery::declarations(
            &root.join("crates/horizon-client/src/game"),
            &["native_hooks", "host", "creatures"],
        )
        .expect("discovering game modules"),
    )
    .expect("writing game modules");
    let tests =
        root.join("crates/horizon-client/src/crafting/ingredient/builtin_kinds_tests/families");
    println!("cargo:rerun-if-changed={}", tests.display());
    std::fs::write(
        out_dir.join("builtin_ingredient_test_modules.rs"),
        horizon_game_discovery::folder_modules(&tests).expect("discovering ingredient tests"),
    )
    .expect("writing ingredient test modules");
    horizon_mod_api_gen::generate(&root, &out_dir, horizon_mod_api_gen::Target::Client)
        .expect("generating native mod API");
    // Whether a native mod can run in a mob's turn (`game/creatures/mob_islands.rs`). The
    // launcher adds trait packages to the workspace's members.
    println!(
        "cargo:rerun-if-changed={}",
        root.join("Cargo.toml").display()
    );
    for dir in ["native-api", "native-crates"] {
        if root.join(dir).is_dir() {
            println!("cargo:rerun-if-changed={}", root.join(dir).display());
        }
    }
    if let Some(found) = horizon_mod_api_gen::mob_turn(&root) {
        println!("cargo:rustc-env=HORIZON_MOB_TURN={found}");
    }
    if std::env::var_os("CARGO_FEATURE_UPSTREAM_PARALLEL_MOB_ISLANDS").is_some() {
        let hooks =
            horizon_mod_api_gen::mob_turn_hook_ids(&root).expect("resolving mob-turn hook IDs");
        std::fs::write(out_dir.join("mob_turn_hooks.rs"), hooks)
            .expect("writing mob-turn hook IDs");
    }
    println!(
        "cargo:rerun-if-changed={}",
        root.join(horizon_mod_api_gen::MODS_STAGE).display()
    );
    if horizon_mod_api_gen::item_turn_has_staged_code(&root) {
        println!("cargo:rustc-env=HORIZON_ITEM_TURN_STAGED=1");
    }
    if std::env::var_os("CARGO_FEATURE_UPSTREAM_PARALLEL_MOBS").is_some() {
        let (hooks, staged) = horizon_mod_api_gen::spawn_proposal_gate(&root)
            .expect("resolving natural spawn proposal hooks");
        std::fs::write(out_dir.join("spawn_proposal_hooks.rs"), hooks)
            .expect("writing natural spawn proposal hooks");
        if let Some(staged) = staged {
            println!("cargo:rustc-env=HORIZON_SPAWN_PROPOSAL_STAGE={staged}");
        }
    }
    // The SDK builds this library, not the platform executable. Carry the same executable
    // policy without asking Cargo to link a binary this package no longer owns.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("Cargo must provide the target OS");
    let target_env =
        std::env::var("CARGO_CFG_TARGET_ENV").expect("Cargo must provide the target environment");
    println!(
        "cargo:rerun-if-changed={}",
        root.join("packaging/executable_link.rs").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("crates/horizon-guest-stack/src/guest_stack_constants.rs")
            .display()
    );
    for arg in windows_exe::executable_link::arguments(&target_os, &target_env) {
        println!("cargo:horizon-executable-link-arg={arg}");
    }
    if std::env::var("CARGO_PKG_NAME").as_deref() != Ok("horizon-client") {
        windows_exe::resources("horizon-client");
    }
}
