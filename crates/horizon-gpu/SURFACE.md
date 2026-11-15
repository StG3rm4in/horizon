# Règles de surface sur GPU : plan de portage

But : calculer sur le GPU l'étape de surface de Horizon (`material_stage::build_columns`),
au bit près, et renvoyer l'état final de chaque bloc du chunk après remplissage et surface.
Mesure de départ (14/11/2026, profil avec graphe d'appels, densité et aquifères sur GPU) :
`build_surface_shared` = 19,9 % du temps CPU de `/pregen` (règles 7,2 %, blocs 5,3 %,
boucle 3,2 %, filons 3,0 %, extensions 1,0 %).

## Ce que fait la surface (code de Horizon)

Pour chaque colonne (lx, lz) du chunk, dans cet ordre (x puis z) :

1. `original_top` = premier y libre de la colonne ; biome en (x, original_top, z).
2. Badlands érodés (`material_extensions::eroded_badlands`) si ce biome est `eroded_badlands`.
3. `top` = premier y libre, pente `gx`, `gz` = différences de premier y libre entre voisins
   (bornées au chunk), `update_xz` : profondeur de surface `depth = surface_depth(x, z)`.
4. Descente de `top` à `min_y` : air → `above = 0`, `water = MIN` ; fluide → `water = y + 1` au
   premier ; solide → calcul de `bottom` (bas de la couche solide), `above += 1`, puis, dans
   les bornes d'écriture, application de la règle avec `update_y(above, y - bottom + 1, water, y)`.
   Le bloc écrit remplace le bloc par défaut (la règle n'est appliquée que là où Horizon l'applique).
5. Océans gelés : icebergs (`frozen_ocean_with_melt`) après la descente.

Les optimisations de Horizon (bandes de hauteur, variantes au-dessus ou au-dessous de la surface
préliminaire, suites de blocs, pliage par biome possible) ne changent pas le résultat : le GPU
évalue la règle générique.

## Règles et conditions

- Règles : `Block`, `Sequence` (le premier enfant qui répond), `Condition`, `Bandlands`, `OreVein`
  (densité > 0, puis `random.at(pos).next_float() > densité` → rien, richesse, espacement < 0,
  minerai brut si `next_float() < raw_ore_chance`) ; aléatoire `root.from_hash_of("minecraft:ore")`.
- Conditions (pures, caches sans effet sur le résultat) : `Hole` (depth <= 0), `Steep`
  (gx <= -4 ou gz >= 4), `Biome`/`BiomeTag`, `Temperature` (cold_enough_to_snow),
  `AbovePreliminarySurface` (y >= floor(préliminaire) + depth - 8), `NoiseThreshold`
  (`noise.get3(x, is_3d ? y : 0, z)` en f64 dans [min, max]), `YAbove`, `Water`, `StoneDepth`
  (avec le bruit secondaire), `VerticalGradient` (`root.from_hash_of(nom).fork_positional()`,
  `next_float() < 1 + (y - bas) / (haut - bas) * -1`), `Not`.

## Entrées

- Par colonne : `surface_depth` = `(int)((bruit surface * 2.75 + 3) + root.at(x, 0, z).next_double() * 0.25)`,
  bruit secondaire, plan de la surface préliminaire (floor), bande de terracotta (bruit
  clay_bands_offset, table de 192 états tirée une fois).
- Par bloc : biome (zoom des biomes sur les chunks 3 × 3, `BiomeZoom::quart_at_block`),
  température (`BiomeTemperature`, bruits de température), densités des filons (en volume).

## Jalons (chacun vérifié bit à bit contre le CPU avant le suivant)

- **S1** : entrées par colonne et tirages aléatoires (profondeur, secondaire, bandes, bruits des
  conditions, gradients verticaux, aléatoire des filons).
- **S2** : biome par bloc (zoom) et température.
- **S3** : arbre de règles traduit en CUDA et descente des colonnes ; comparaison bloc par bloc.
- **S4** : extensions (badlands érodés, icebergs).
- **S5** : intégration : le GPU renvoie les états finaux, le CPU écrit les sections ;
  sonde, parité vanilla, `/pregen`.

## Avancement

- **S1, fait (14/11/2026)** : `aleatoire.cu` (Xoroshiro, next_float, next_double, at), `generer_bruits`,
  `evaluate_noises`, `positional_draws`, `PositionalRandom::xoroshiro_state` (horizon-core).
  Contrôle `HORIZON_GPU_SURFACE_TEST=1` (generator.rs, `surface_gpu_test`) : 64 bruits sur 20 000 points,
  tirages des fabriques racine, ore, bedrock_floor, bedrock_roof, deepslate, et profondeur de surface :
  identiques au CPU.
- **S2, fait (14/11/2026)** : `surface.cu` (zoom `sf_quart`, `sf_biome_quart` sur les 3 x 3 chunks,
  simplex 2D `sf_simplex2`, `sf_temperature`), `generer_surface_s2`, `surface_s2_check` ;
  horizon-core `Noise::as_simplex`, `BiomeZoom::seed` ; horizon-world `temperature_noises()`.
  Zoom sur 20 000 positions et 20 combinaisons de température : identiques au CPU.

- **S3, fait (14/11/2026)** : la règle de surface et la descente des colonnes sur GPU, comparées bloc par
  bloc aux blocs écrits par la surface CPU (`HORIZON_GPU_SURFACE_TEST=1`, `compare_surface_gpu` dans
  generator.rs ; `HORIZON_GPU_SURFACE_BIOMES=1` ajoute les colonnes par biome).
  - Forme neutre dans horizon-gpu : `SurfaceRule`, `SurfaceCondition`, `SurfaceSetup` ; traduction de la
    `MaterialRule` par `SurfaceIr` (generator.rs ; références mémorisées par adresse d'`Arc`, ancres
    résolues sur CPU, biomes en ids, bruits des conditions par nom). 231 règles, 127 conditions, 9 bruits.
  - `MaterialRule::OreVein.roots: Option<[FunctionId; 3]>` (rempli par `MaterialRuleLibrary`, `None` pour
    les règles construites sans graphe, comme la fixture de la sonde).
  - `generer_surface` : fonctions `C_k` / `R_k`, noyau `ksurface` (un thread par colonne).
  - Surface préliminaire : `chunk_surface_level` = `interpolated` (cellules 16 × 1 × 16) autour de
    `find_top_surface` ; le CPU l'échantillonne **en volume** sur le plan y = 0 du chunk
    (`MaterialDensitySources`). Le GPU fait de même : programme de volume 16 × 1 × 16 (`niveau`), avec
    `find_top_surface` en mode volume (borne haute en volume, densité point par point).
  - Corrigé en route : une interpolation lue par une passe 2D (nœud sans Y) ajoutait une grille de coins
    jamais générée (`noyaux` génère maintenant les passes 2D dans la boucle des grilles, avec `y = MY`) ;
    tables `__device__` à initialiseur constant (températures en bits).
  - Résultat : **0 écart sur 3 668 chunks, 346 M blocs comparés dont 60,5 M changés par la surface**, et
    0 condition non évaluable. Échantillons : 5 zones de 64 chunks, puis 25 + 36 + 32 petites zones
    dispersées (script `/tmp/s3_zones2.sh N PAS DX DZ SORTIE` sur le serveur). 40 biomes en surface, dont
    badlands (bandes de terracotta), snowy_slopes, grove, meadow, ice_spikes, swamp, desert, stony_shore,
    windswept_gravelly_hills, dripstone_caves, sulfur_caves. Pas encore vus : les pics
    (jagged/frozen/stony_peaks), mangrove_swamp, mushroom_fields, cherry_grove, wooded_badlands.
    38 269 colonnes ignorées (eroded_badlands, frozen_ocean, deep_frozen_ocean : jalon S4).
  - Pas de mesure de vitesse à ce stade : un lancement et des allocations par chunk, 256 threads.
- **S4, fait (14/11/2026)** : les extensions dans `ksurface` (`sf_pilier`, `sf_iceberg` dans `surface.cu`,
  `SurfaceExtensions` dans la configuration : biomes, 6 bruits, glace tassée, neige).
  - Pilier des badlands érodés avant la descente de sa colonne (le pilier est solide pour la descente, bloc
    par défaut là où la règle ne donne rien) ; iceberg après, avec ses tirages dans l'ordre des courts-circuits
    et la fonte (`temperature(x, mer, z) > 0,1`). Les deux ne dépendent que des codes du remplissage :
    l'iceberg n'écrit que dans l'air et l'eau, jamais sur un bloc de la règle.
  - Ordre des colonnes (x puis z) : les extensions montent le premier y libre, que lisent les pentes. Trois
    hauteurs en mémoire partagée (d'origine, après pilier, finale) ; la pente lit la finale pour x − 1 et
    z − 1 (déjà traitées), l'origine pour x + 1 et z + 1, et la hauteur après pilier pour la colonne même.
    Logique recopiée de `build_columns`, mais pas vérifiée par les données : aucune condition `Steep` n'est
    évaluée à côté d'une colonne à extension dans les zones testées.
  - Résultat : **0 écart sur 1 468 chunks (144 M blocs), dont 349 000 blocs écrits par les extensions**
    (4 595 de piliers sur 13 640 colonnes de badlands érodés, 344 000 d'icebergs sur frozen_ocean et
    deep_frozen_ocean), plus wooded_badlands ; la zone des pentes enneigées reste à 0 écart.
    `HORIZON_GPU_SURFACE_SKIP_EXT=1` ignore encore ces colonnes si besoin.

- **S5, intégration (15/11/2026)** : `HORIZON_GPU_SURFACE=1` (avec `HORIZON_GPU_DENSITY`).
  - `SurfaceGpu::chunk_states` renvoie l'état final de chaque bloc (remplissage puis surface) sur 32 bits :
    l'état sur 30 bits, bit 30 si la surface y écrit un état à fluide (l'eau des marais), bit 31 si le
    remplissage planifie une mise à jour de fluide (code + 4 sur de l'eau ou de la lave).
  - Horizon : `terrain.rs::fill_noise_states` écrit ces états en une passe dans l'ordre du remplissage
    (suites d'un même état sans fluide par `fill_solid_run`, cartes de hauteur mises à jour comme
    `fill_noise_codes`), marque le post-traitement du remplissage, puis celui de la surface dans l'ordre de
    la surface (x puis z, y décroissant) : les listes de post-traitement sont ordonnées. La surface CPU est
    alors sautée pour le chunk. Repli CPU si une condition n'est pas évaluable ou en cas d'erreur GPU.
  - Contrôles au démarrage (generator.rs) : états sur 30 bits ; extensions et bloc par défaut sans fluide ;
    les états à fluide de la règle vont dans `fluid_states`.
  - Vitesse (sonde, un appel à la fois) : 2,7 ms par chunk au premier essai, **0,32 à 0,35 ms** ensuite :
    - noyau découpé en trois : `ksf_tetes` (une warp par colonne : premier y libre par vote, scalaires et
      extensions), `ksf_descente` (une warp par colonne : below, above et water par masques de 32 blocs,
      `ballot`/`popc`), `ksf_blocs` (un thread par bloc : la règle). Noyau 2,4 ms → 0,14 ms ;
    - surface préliminaire par région de 8 × 8 chunks (plan 128 × 128, cache hôte de 1 024 régions) :
      159 → ~20 µs par chunk ; sous-graphes sans Y de `find_top_surface` calculés avant la boucle ;
    - tampons, flux et mémoire verrouillée réutilisés (`RessourcesSurface`).
    Restent par chunk : filons ~86 µs (3 programmes, 30 noyaux), envois et retour ~70 µs.
    `HORIZON_GPU_SURFACE_PROFIL=1` donne le détail à l'arrêt.
  - Vérifications : sonde `chunk_dump` identique à l'octet près sur 3 zones (terres, océans gelés, badlands
    érodés, 59 chunks à structures) ; **parité vanilla à 0 différence sur 785 chunks**, dont 3 479 chunks
    surfacés par le GPU, 0 repli.
  - `/pregen`, rayon 1 600 blocs, zones neuves 230000/−250000 et −260000/270000 (script `/tmp/ab6.sh`,
    la variante gpus tourne dans la copie `gpub`) :

    | Variante | Zone 1 | Zone 2 | CPU par chunk |
    |---|---|---|---|
    | CPU seul | 684 chunks/s | 603 chunks/s | 16,3 et 16,7 ms |
    | GPU (densité, aquifères, beard) | 943 (+38 %) | 803 (+33 %) | 11,3 et 12,0 ms |
    | **GPU + surface** | **1 067 (+56 %)** | **906 (+50 %)** | **9,6 et 10,3 ms (−41 et −38 %)** |

    La surface sur GPU ajoute +13 % et retire ~1,7 ms de CPU par chunk. 0 repli sur 42 849 chunks par zone.
    Sous charge, un appel prend 0,60 à 0,63 ms (0,35 ms seul) ; GPU à 24-25 %, 2 à 3,7 Go, 120-129 W.
- **S5, réduction de la latence (15/11/2026)** :
  - **Filons par région** de 8 × 8 chunks, sur les seuls y de `positive_y` (union des filons, alignée sur
    16 ; la règle `OreVein` rend « rien » hors de ses bornes, sans tirage, comme le CPU) et gardés sur le GPU
    avec le plan de la surface préliminaire (`RegionSurface`, 48 régions, `HORIZON_GPU_SURFACE_REGIONS`).
    Plus que ~11 opérations par chunk au lieu d'une quarantaine.
  - **Surface sur le flux des codes** : `GpuFinalDensity::chunk_blocks_surface` lance la surface juste après
    le noyau des codes (crochet `suite` de `copier_codes` et du chemin à structures) ; les codes ne
    remontent plus au GPU et il n'y a qu'une attente. `SurfaceGpu::chunk_states` (mode de comparaison)
    garde son propre flux.
  - **Attente bloquante** (`HORIZON_GPU_SYNC=block`) : moins de CPU mais des appels de 0,87-0,92 ms au lieu
    de 0,51-0,53 ms, débit en baisse (969 / 999 contre 1 001 / 1 019 chunks/s) : on garde l'attente active.
  - `/pregen` (copies `gpu` et `gpub`, scripts `/tmp/ab7.sh` et `/tmp/ab8.sh`, gain de la surface sur le GPU
    sans surface, zones neuves à chaque passe) :

    | Étape | Zones | GPU sans surface | GPU + surface | Gain |
    |---|---|---|---|---|
    | Filons par région | 270000/−290000, −300000/310000 | 863, 871 | 1 001, 1 019 | +16 %, +17 % |
    | + flux des codes | 310000/−330000, −340000/350000 | 979, 814 | **1 125, 954** | +15 %, +17 % |

    CPU par chunk avec la surface : 9,2 à 9,6 ms (16-18 % de moins que sans). Le flux des codes est neutre
    dans la mesure, gardé pour la simplicité des échanges. Sonde identique et parité vanilla à 0 différence
    après chaque étape (3 479 chunks surfacés par le GPU, 0 repli).
  - Reste du temps CPU : écritures des sections, éclairage, décorations, compression ; la surface ne pèse
    plus que l'appel GPU (~0,5-0,7 ms d'attente active sous charge).

## Conception de S3 (faite, gardée pour mémoire)

- Source : `Pipeline.rule` (`MaterialRule` non compilé, références en `Arc`), traduit en CUDA par un
  générateur (fonction par sous-arbre référencé, mémorisée par adresse de l'`Arc`, pour éviter
  l'explosion du code). `Block` → état ; `Sequence` → premier enfant qui répond ; `Condition` ;
  `Bandlands` (bruit `clay_bands_offset` et table de 192 états) ; `OreVein` (densité et richesse en
  mode **volume**, calculées avec la région comme la densité finale ; espacement en mode **point** ;
  aléatoire `root.from_hash_of("minecraft:ore")` ; il faut les FunctionId des racines, via
  `MaterialRuleLibrary`).
- Conditions : biomes → masque de bits sur l'id ; `BiomeTag` → membres ; `Temperature` → table par
  biome (base, gelé) ; `YAbove` et `VerticalGradient` : ancres résolues sur CPU
  (`VerticalAnchor::resolve(generation)`) ; `NoiseThreshold` → bruit `pipeline.noises[nom]` en `get3` ;
  `AbovePreliminarySurface` → `floor(surface_level(x, 0, z)) + depth - 8`.
- Noyau : un thread par colonne (256 par chunk), descente comme `build_columns` (air, fluide, solide,
  `bottom`, `above`, `water`), bornes d'écriture, règle appliquée aux blocs solides ; entrées : codes du
  remplissage du chunk (0/8 solide, 1 air, 2/3 fluide), biomes des 3 x 3 chunks, colonnes
  (profondeur, secondaire, pente, préliminaire). À lire avant : `first_available` et `Column::kind`
  (material_stage.rs), les bornes d'écriture.
- Vérification : comparer bloc par bloc aux blocs écrits par la surface CPU (sonde `chunk_dump`,
  `HORIZON_PHASE=terrain`).
