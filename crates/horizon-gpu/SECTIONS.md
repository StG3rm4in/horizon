# Sections compactes sur GPU (`HORIZON_GPU_SECTIONS`)

## Ce qui coûte

Profil `/pregen` du 15/11, tout sur GPU (`/tmp/sous_etats.py`) : `fill_noise_states`, qui écrit les
états finaux du GPU (remplissage puis surface) dans le `TerrainChunk`, pèse **9,1 % du CPU** :
`fill_solid_run` 4,9 %, la boucle elle-même 2,7 % (dont 0,85 % pour relire les 98 304 états à la
recherche du bit 30), `set_block_raw` 0,7 %, les cartes de hauteur 0,5 %. Le GPU renvoyait pour cela
les 98 304 états en u32 (393 Ko par chunk).

## Pourquoi le GPU peut rendre la section exacte

Une section de génération (`BlockSection`) garde une palette dynamique : ordre d'arrivée des états,
compteur par case, cases libres, largeur des mots qui ne fait que grandir. Sa sauvegarde de proto chunk
(`BlockSection::to_nbt`) écrit cette palette telle quelle, cases libres comprises : pour rester
identique à l'octet près, le GPU doit rendre exactement ce que `set` aurait laissé.

`fill_noise_states` écrit chaque cellule non air une seule fois, dans une section neuve (uniforme air),
colonne après colonne (z puis x), de haut en bas ; `fill_run` équivaut à des `set` dans le même ordre.
Le résultat ne dépend donc que de cette suite d'écritures :

- s'il reste de l'air : palette [air, états dans l'ordre de leur première écriture], leurs compteurs,
  pas de case libre ;
- plus d'air et un seul état : section uniforme ;
- plus d'air, et l'état de la dernière écriture était déjà dans la palette : l'air garde sa case, à 0,
  qui est libre ;
- plus d'air, et l'état de la dernière écriture est nouveau : il prend la case de l'air ;
- la largeur des mots est `bits_for_palette(longueur, 4)` de la palette finale.

Le compte non air suit les deux chemins d'écriture : +1 par cellule d'une suite (`fill_solid_run`),
+1 si l'état n'est pas de l'air pour une cellule écrite seule (fluide ou marquée). Les cartes
`OCEAN_FLOOR_WG` et `WORLD_SURFACE_WG` valent 1 + la plus haute cellule opaque. Les marques de
post-traitement gardent leur ordre : bit 31 des cellules écrites, colonnes en z puis x, y descendant ;
puis bit 30 de toutes les cellules, colonnes en x puis z, y descendant.

## Déroulement

1. Après `ksf_blocs` (les états finaux sur le GPU), sur le même flux (`SurfaceGpu::enfiler`) :
   - `ksec_hauteurs` : un bloc par colonne, les deux cartes et le nombre de marques ;
   - `ksec_marques` : les rangs des colonnes, puis les marques des seules colonnes qui en ont ;
   - `ksec_sections` : un bloc par section ; une table de 128 états (première position d'écriture,
     compteur), leurs rangs, les règles de la dernière écriture, les mots compactés.
2. Le GPU renvoie 1 285 mots u32 par section, les hauteurs et les marques (environ 125 Ko par chunk au
   lieu de 393 Ko).
3. `fill_noise_sections` (terrain.rs) pose chaque section par `BlockSection::from_parts`, ses compteurs
   non air, les cartes (`Heightmap::set_first_available`) et les marques.

Plus de 128 états dans une section ou plus de 2 048 marques d'une sorte : les états bruts reviennent et
`fill_noise_states` les écrit comme avant. Seul un chunk neuf de toute la hauteur prend ce chemin
(`TerrainChunk::is_fresh`).

## Variables

- `HORIZON_GPU_SECTIONS=1` : sections compactes du GPU (avec `HORIZON_GPU_SURFACE=1`).
- `HORIZON_GPU_SECTIONS=test` : les états bruts reviennent aussi ; `fill_noise_states` les écrit, puis
  `check_noise_sections` compare chaque section (palette, compteurs, cases libres, mots), les compteurs
  non air, les cartes et les marques.

La ligne `GPU surface :` donne les chunks posés, les replis par les états bruts et les écarts.

## Vérifications

- Sonde `chunk_dump` (`/tmp/sec1.sh`), 3 zones de 256 chunks (0,0 ; −1250,940 ; badlands érodés
  −320,−295) : identique au CPU à l'octet près en mode test et avec les sections ; 0 écart et 0 repli
  sur 1 200 chunks en mode test.
- Parité vanilla : **0 différence sur 785 chunks** avec les sections ; overworld en mode test :
  3 479 chunks, 0 écart, 0 repli.
- Tests unitaires de horizon-world (chunk, heightmap, terrain) : 34 passent.

## Mesures

La première version (un seul bloc pour les cartes et les marques, lectures non coalescées) allongeait
l'appel de surface de 140 µs (537 → 676 µs, sonde sans contention). Avec un bloc par colonne pour les
cartes, les marques relues seulement dans les colonnes qui en ont et des lectures coalescées dans
`ksec_sections`, le surcoût tombe à 48 µs (498 → 546 µs).

Bancs `/pregen` (`/tmp/ab16.sh`, release, rayon 1 600 blocs, zones neuves, tout sur GPU) :

| Zone | Sans sections | Avec | CPU par chunk |
|---|---|---|---|
| 2000000/−2020000 | 1 574 chunks/s | **1 690 (+7,3 %)** | 5,96 → 5,31 ms (−10,9 %) |
| −2030000/2040000 | 1 524 chunks/s | **1 665 (+9,3 %)** | 6,34 → 5,60 ms (−11,7 %) |
| 2050000/−2070000 | 1 303 chunks/s | **1 387 (+6,5 %)** | 7,48 → 6,93 ms (−7,4 %) |

(Première version : +3,0 / +5,2 / +6,1 % ; CPU −6,8 / −11,4 / −9,2 %.)
