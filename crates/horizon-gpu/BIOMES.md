# Biomes sur GPU

## Ce qui coûte

`biome_chunk_or_busy` (horizon-world, generator.rs) construit le `BiomeChunk` d'un chunk : 4 × 4 × 96
quarts dans l'overworld (1 536). Pour chacun, les six champs du climat (température, humidité,
continentalité, érosion, profondeur, bizarrerie), échantillonnés en volume sur le CPU, puis la
recherche du biome le plus proche dans l'arbre du climat (`ClimateIndex`). Profil du 15/11, tout le
reste sur GPU : **8,6 % du CPU**, dont 2,0 % dans `distance_until` et 0,7 % dans `search`.

## Ce que fait le GPU

`HORIZON_GPU_BIOMES=1` : pour une source multi-noise, le GPU calcule une région de R × R chunks
(`HORIZON_GPU_BIOMES_REGION`, 8 par défaut) d'un coup, au premier chunk demandé de la région.

1. `kbi_climat`, un thread par colonne de quarts : les six racines du climat du routeur en mode point
   (`ClimateSampler::sample` : position = quart << 2), par une fonction générée
   (`Gen::colonne_points`, traduction.rs). Comme dans la descente de la surface, les sous-graphes sans
   Y (`domain_axes`) sont calculés une fois avant la boucle sur les quarts Y, et partagés entre les six
   racines : cinq des six champs, et la partie de la profondeur qui n'est pas le gradient en Y.
2. `kbi_recherche`, un thread par quart : la quantification `(v * 10000f) as i64` (saturée, NaN → 0),
   puis la recherche dans l'arbre **sans point de départ** : parcours en profondeur, enfants dans
   l'ordre, un nœud n'est visité que si sa distance est strictement sous la meilleure. Le noyau garde
   la première feuille au minimum, et ce minimum.
3. `kbi_ordre`, un bloc par chunk : la recherche du CPU part du résultat précédent du chunk
   (`ClimateSearch::last`, un résolveur neuf par chunk). Elle rend ce résultat précédent si sa distance
   vaut le minimum du quart, sinon la première feuille au minimum, celle d'une recherche sans point de
   départ : toutes les feuilles vues avant elle sont plus loin, et ses ancêtres sont au plus à sa
   distance. Le bloc applique donc `last` quart par quart, dans l'ordre du remplissage (sections de bas
   en haut, puis x, y, z), par tuiles de 128 quarts en mémoire partagée, et écrit l'identifiant du biome.

L'arbre vient de `ClimateIndex::tree` (index.rs) : par nœud, les six intervalles et le carré de la
distance du décalage ; et les liens (premier enfant et nombre d'enfants ; une feuille : son entrée et
0). La distance est celle du CPU, en arithmétique 64 bits modulaire ; l'arrêt anticipé de
`distance_until` ne change aucune comparaison.

Côté CPU (`gpu_biome_chunk`) : une région par `OnceLock`, dans un cache des 128 dernières régions
(`GpuBiomeRegions`). Le premier worker qui en a besoin la calcule ; les autres attendent, ou rendent
« occupé » quand `biome_chunk_or_busy` est appelé sans attente. Chaque `BiomeChunk` est rempli depuis
la région, dans l'ordre de `fill`.

Repli sur le CPU :

- chunk qui se fond avec un ancien monde (blender non vide) ;
- entrée sans identifiant de biome (le CPU lève l'erreur) ;
- région en erreur (pile du parcours, recherche sans résultat, erreur CUDA) ;
- source qui n'est pas multi-noise ;
- aléatoire legacy (le Nether), comme pour la densité.

## Variables

- `HORIZON_GPU_BIOMES=1` : biomes sur GPU.
- `HORIZON_GPU_BIOMES=test` : les deux chemins, comparés section par section ; le CPU fait foi. Des
  lignes `GPU biomes test:` sont écrites dans latest.log aux puissances de deux, avec les 16 premiers
  écarts.
- `HORIZON_GPU_BIOMES_REGION` : côté d'une région en chunks (8).
- `HORIZON_GPU_DUMP_BIOMES=fichier` : le source CUDA généré.

## Vérifications

- Sonde `chunk_dump` (graine 12345, phase full, tout sur GPU) sur 3 zones de 256 chunks : identique à
  l'octet près au CPU, en `test` comme en `1` ; 484 chunks par zone comparés en mode test, 0 écart.
- Parité vanilla : overworld en mode test, 627 chunks, **0 différence**, 5 473 chunks de biomes
  comparés, 0 écart ; complète, tout sur GPU, **0 différence sur 785 chunks**. Le Nether (aléatoire
  legacy) reste sur le CPU : 0 différence sur 113 chunks.
- `/pregen` en mode test (rayon 800, zones neuves 2600000/−2620000 puis 2650000/−2670000) : plus de
  8 192 chunks comparés, 0 écart.

## Mesures

Temps GPU d'une région de 8 × 8 chunks (nsys, sonde sur 32 × 32 chunks) :

| Version | Climat | Recherche | Ordre | Région entière |
|---|---|---|---|---|
| 1 : climat et recherche par quart, ordre par un thread par chunk | 3,46 ms (avec la recherche) | | 0,97 ms | ~5 ms |
| 2 : climat par colonne, ordre par tuiles | 0,11 ms | 0,33 ms | 0,23 ms | ~1,0 ms (max 1,4) |

Le bruit est en f64, que la 3080 calcule lentement : en version 1, c'est le climat évalué en chaque
quart qui coûtait. Avec des régions de 16 × 16, une région prend 2,0 ms : moins par chunk, mais une
attente plus longue pour le premier chunk demandé ; 8 reste la valeur par défaut.

Bancs `/tmp/ab18.sh` (release, rayon 1 600 blocs, zones neuves, tout le reste sur GPU dont les
sections) :

| Zone | Biomes CPU | Biomes GPU | CPU par chunk |
|---|---|---|---|
| 2700000/−2720000 | 1 669 chunks/s | 1 648 (−1,2 %) | 5,43 → 5,29 ms (−2,6 %) |
| −2730000/2740000 | 1 435 chunks/s | 1 485 (+3,5 %) | 6,46 → 6,05 ms (−6,3 %) |
| 2750000/−2770000 | 1 415 chunks/s | 1 393 (−1,5 %) | 6,56 → 6,39 ms (−2,6 %) |

Le débit ne bouge pas : `/pregen` n'est pas limité par le CPU (8,4 à 9,2 cœurs occupés sur 12). Le
CPU par chunk baisse de 2,6 à 6,3 % selon la zone.

Profil (zone 2800000/−2820000, 1 714 chunks/s avec les binaires de profil) :

- `biome_chunk_or_busy` passe de 5,7 % du CPU à 0,1 %, et le chemin GPU (`gpu_biome_chunk`,
  `BiomeGpu::region`) coûte 0,18 % ;
- la catégorie « biomes et climat » passe de 8,6 % à 2,8 %. Le reste vient des appels hors des
  chunks de biomes : `fold` 1,4 %, `carver_biome` 0,4 %, `biome_id` 0,3 %.
- Les plus gros postes sont maintenant :

| Poste | Part du CPU |
|---|---|
| Décorations | 29,6 % |
| Sauvegarde | 17,9 % |
| Attente et pilote du GPU | 11,2 % |
| zlib | 9,9 % |
| Thread principal | 7,8 % |
| Génération (reste) | 6,3 % |
| Structures | 5,1 % |
