# Minerais sur GPU (`HORIZON_GPU_ORES`)

## Ce qui coûte

Profil `/pregen` du 15/11 (densité et surface sur GPU, 1 023 chunks/s) : `ore_feature` représente
10,5 % du CPU en coût inclusif. Ventilation par ligne de `ore_feature.rs` :

| Poste | Part du CPU |
|---|---|
| Écriture des blocs (`set_section_block`) | 1,9 % |
| Lecture des blocs (`section_block`) | 1,6 % |
| Géométrie des blobs : élagage des sphères, carrés par axe, `pass_run`, boucle des cellules, bitset `tested` | environ 5 % |
| Recherche de colonne, règles, divers | le reste |

Les lectures et écritures dépendent de l'état du monde au moment de l'étape des minerais : elles
restent sur le CPU. La géométrie ne dépend que des tirages aléatoires : c'est elle qui passe sur le
GPU.

## Pourquoi la géométrie peut être calculée à l'avance

La décoration d'un chunk dépend de l'ordre : les features des voisins décorés avant lui écrivent dans
sa région. Les minerais lisent donc un monde qu'on ne connaît qu'à l'étape 6 (`underground_ores`).
Mais pour un minerai *planifiable* :

- `discard_chance_on_air_exposure` = 0 : aucun tirage par cellule, aucun voisin lu ;
- des règles de cible pures (`heights` de Horizon : sans tirage) ;
- des modificateurs de placement qui ne lisent aucun bloc (`count`, `in_square`, `height_range`,
  `rarity_filter`, `biome`, `random_chance`, `offset`, `fixed`, `randomly_selected` de ceux-là).

Chaque feature est réensemencée par `set_feature_seed`. Les tirages d'une tentative (position, puis
direction et hauteurs des extrémités, puis `size` × `next_double` des sphères) ne dépendent alors du
monde que par la recherche de colonne (`OCEAN_FLOOR_WG` sous le blob), qui décide si les tirages des
sphères ont lieu. En minecraft vanilla 26.3, cela couvre dirt, gravel, granite, diorite, andesite,
tuff, coal_upper, iron (×3), copper (×2), redstone (×2), lapis, emerald, gold_extra, infested et clay.
Sont exclus : coal_lower, gold, gold_lower et les diamants, avec une chance strictement entre 0 et 1,
ainsi que lapis_buried et diamond_buried, à 1.

Avec une chance nulle, chaque cellule du blob est lue une fois, et seule sa propre écriture en dépend.
L'ordre de visite des cellules d'un blob ne change donc rien : le GPU renvoie un ensemble, pas une
liste ordonnée. Les blobs d'une feature restent appliqués dans l'ordre des tentatives, puisqu'un blob
lit ce que le précédent a écrit.

## Déroulement

1. **Début de la décoration du chunk** (`Decoration::predict_ores`). Pour chaque minerai planifiable
   de l'étape 6, dans l'ordre et sur la même sélection de biomes que `FeatureDecorationPlan::decorate`
   (`step_features`), le CPU rejoue la chaîne de placement sur un aléatoire à part, avec
   `OreFeature::predict`. La recherche de colonne est faite sur la région telle qu'elle est à ce
   moment. Les sphères des tentatives qui trouvent une colonne sont envoyées au GPU (`OreGpu::submit`).
2. **GPU** (`minerais.cu`, `kmin_blobs`, un bloc de 128 threads par blob) :
   - élagage des sphères dans l'ordre exact du CPU, avec la boucle `i2` découpée en paquets de 32 dans
     une warp. Le premier `i2` qui retire `i1` arrête la boucle, car un `i1` retiré ne retire plus rien ;
   - carrés par axe `((v + 0.5) − c) / r` au carré, sur les bornes `max(floor(c ∓ r), start)` ;
   - pour chaque rangée (x, y), le masque des z tels que `(xd² + yd²) + zd² < 1`, réunis par `atomicOr`
     en mémoire partagée.

   Tout est calculé en double avec les intrinsics `__d*_rn`, sans contraction (`--fmad=false`). En
   sortie, `size_xz × size_y` masques par blob, plus un drapeau d'erreur si une cellule sort de la
   boîte : le CPU aurait alors levé `WorkLimit`.
3. **Étapes 0 à 5 sur le CPU** pendant ce temps.
4. **Étape 6, feature planifiée** (`OrePlan::place`) :
   - si une écriture a pu déplacer une heightmap depuis la prédiction (compteur `floor_writes` du
     monde, incrémenté avec `raise_floor_bound`), le CPU refait toutes les recherches de colonne de la
     feature. Les écritures de minerais ne touchent pas aux heightmaps, on peut donc les faire d'abord ;
   - si l'une diffère de la prédiction (les étapes 0 à 5 ont pu élever le sol), la feature passe par
     le chemin CPU normal. Rien n'a été tiré ni écrit, et l'aléatoire de la feature est intact ;
   - sinon, après attente du lot (une fois par chunk), chaque blob est appliqué par
     `OreFeature::place_cells`, avec le même test par cellule que `do_place` (`place_cell`, partagé).

Les caches que la prédiction remplit tôt ne changent pas les réponses :
- les bornes de sol `floor_bounds` restent des majorants, relevés à chaque écriture ;
- les heightmaps amorcées puis mises à jour donnent le même résultat qu'un amorçage tardif.

## Mesures

**Version 1** : cellules appliquées une à une par `place_cell`, lot envoyé en 3 copies, retour en 2.
- Bancs `/pregen` (`/tmp/ab9.sh`, rayon 1 600 blocs) :
  - zone 410000/−430000 : 981 → 986 chunks/s, CPU 10,49 → 10,59 ms par chunk ;
  - zone −440000/450000 : 1 018 → 1 043 chunks/s, CPU 9,10 → 8,93 ms par chunk.
- Profil (`/tmp/sous_ore3.py`) : les minerais passent de 10,5 % à 9,1 % du CPU.
  - étape 6 planifiée : 5,7 %, dont environ 5 % de travail par cellule (zone d'écriture, lecture,
    mémo des règles, écriture) ;
  - prédiction : 2,2 %, dont 0,6 % d'envoi au GPU et 0,5 % de tirages des sphères ;
  - minerais restés sur le CPU : 1,15 %.

  La géométrie n'était donc qu'environ 2,4 points, pas 5 : la première ventilation par ligne comptait
  dans la boucle de visite du travail qui reste par cellule.

**Version 2** :
- application par rangée :
  - `PlacementContext::replace_row` traite toute une rangée, cellule par cellule par défaut ;
  - `World` le redéfinit pour traiter un chunk à la fois : zone d'écriture et hauteur vérifiées une
    fois, puis `TerrainChunk::replace_row_raw` ;
  - les intervalles de Y des règles sont calculés une fois par rangée ;
- lot envoyé en une seule copie, retour en une seule.

Bancs de la version 2 :
- zone 500000/−520000 : 937 → 965 chunks/s (+3,0 %), CPU 10,84 → 10,77 ms par chunk ;
- zone −530000/540000 : 1 026 → 1 048 chunks/s (+2,1 %), CPU 8,97 → 8,91 ms par chunk.

Sonde identique, mode test à 0 écart, parité vanilla à 0 différence (1 775 lots, 0 repli).

Profil de la version 2 (zone 560000/−580000, 1 143 chunks/s avec les binaires de profil) : les
minerais représentent **7,2 % du CPU** (contre 10,5 % sans GPU), répartis ainsi :
- étape 6 planifiée : 3,7 %, dont 2,4 % de lectures et écritures par rangée ;
- prédiction : 2,4 %, dont 0,5 % de tirages des sphères, 0,5 % de chaînes de placement, 0,5 %
  d'envoi et 0,3 % de recherches de colonne ;
- minerais restés sur le CPU (chance d'abandon à l'air entre 0 et 1) : 1,0 %.

En contrepartie, l'attente et le pilote du GPU passent de 5,5 à 6,7 %. Le pilote CUDA fait trois
appels par chunk de plus, et les noyaux des minerais partagent le GPU avec ceux de la densité et de la
surface. Le débit gagne 2 à 3 %, et le temps CPU par chunk baisse moins que la part des minerais.

## Variables

- `HORIZON_GPU_ORES=1` : minerais sur GPU, indépendamment de `HORIZON_GPU_DENSITY`.
- `HORIZON_GPU_ORES=test` : pour chaque blob, compare aussi les masques GPU à ceux du CPU
  (`OreFeature::blob_rows`). En cas d'écart, la feature passe par le CPU et l'écart est compté.

À l'arrêt, une ligne `GPU minerais :` donne le nombre de lots, de blobs par lot, le temps d'envoi et
d'attente, les replis CPU et les écarts.
