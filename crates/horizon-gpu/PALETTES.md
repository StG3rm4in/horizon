# Palettes des chunks finis

## Ce qui coûte

`block_column` (horizon-client, vanillagen.rs) convertit les sections de génération (`BlockSection`)
en sections de chunk (`PalettedContainer` de horizon-core). Elle sert aux vues des chunks voisins pour
la lumière et à la colonne finale d'un chunk. Profil du 15/11, tout sur GPU : **5,7 % du CPU**, dont
3,9 % dans `PalettedContainer::from_values` et 1,1 % dans `BlockSection::fill_states`.

Le chemin passait par les valeurs : décompacter les 4 096 états de la section, puis reconstruire une
palette dans l'ordre de première apparition par index (`palette_ids`) et recompacter tous les ids.

## Le chemin direct

`TerrainChunk::section_container` → `BlockSection::to_container` :

1. les cases vivantes de la palette de génération, rangées par leur première cellule (deux cases du
   même état fusionnent) : c'est l'ordre de première apparition que `from_values` trouve ;
2. chaque id renuméroté vers cette palette, directement dans les mots
   (`PalettedContainer::from_mapped`). De 4 à 4 bits, le cas courant, une table de 256 octets traduit
   les deux entrées d'un octet d'un coup ; sinon entrée par entrée. La palette globale (plus de 256
   états) reçoit les valeurs, comme dans `from_values`.

Pour des mots de 4 bits, la recherche des premières cellules lit un mot à la fois : le masque des ids
qu'il contient, puis la position seulement pour un id vu pour la première fois.

Le résultat est le même conteneur, mot pour mot (`to_wire` : largeur, palette, mots), pas seulement les
mêmes valeurs.

## Variables

- `HORIZON_COLUMN=values` : l'ancien chemin (banc).
- `HORIZON_COLUMN_TEST=1` : les deux chemins, comparés par `to_wire` ; une ligne `column test:` dans
  latest.log aux puissances de deux.

## Vérifications

- Test unitaire `to_container_matches_from_values` (horizon-world, chunk.rs) : sections aléatoires de
  1 à 300 états, avec des cases mortes et réutilisées, jusqu'à la palette globale.
- `/pregen` en mode test (zone 2100000/−2120000, rayon 800, tout sur GPU) : **262 144 sections, 0
  différence**.
- La sonde et le harnais de parité vanilla ne passent pas par `block_column`.

## Mesures

Première version (lecture des cellules une à une pour les premières apparitions), bancs `/tmp/ab17.sh`
(release, rayon 1 600 blocs, zones neuves, tout sur GPU dont les sections) :

| Zone | Par les valeurs | Directe | CPU par chunk |
|---|---|---|---|
| 2200000/−2220000 | 1 432 chunks/s | 1 410 (−1,6 %) | 6,59 → 6,45 ms (−2,1 %) |
| −2230000/2240000 | 1 527 chunks/s | 1 530 (+0,2 %) | 6,10 → 5,95 ms (−2,5 %) |
| 2250000/−2270000 | 1 590 chunks/s | 1 697 (+6,7 %) | 5,80 → 5,37 ms (−7,4 %) |

Profil de cette version (zone 2300000/−2320000, 1 593 chunks/s avec les binaires de profil) :
`block_column` tombe à 3,4 % du CPU, dont 3,1 % dans `to_container`. `fill_noise_states` est à 0 %
(sections GPU). Les plus gros postes sont maintenant les décorations (31,5 %), la sauvegarde (15,7 %),
zlib (8,9 %), l'attente et le pilote du GPU (8,7 %), les biomes (8,6 %) et le thread principal (7,2 %).

Version avec recherche des premières apparitions par mot (mêmes vérifications : test unitaire,
`/pregen` en mode test sur la zone 2400000/−2420000 : 262 144 sections, 0 différence) :

| Zone | Par les valeurs | Directe | CPU par chunk |
|---|---|---|---|
| 2500000/−2520000 | 1 544 chunks/s | **1 584 (+2,6 %)** | 6,06 → 5,80 ms (−4,3 %) |
| −2530000/2540000 | 1 524 chunks/s | **1 548 (+1,5 %)** | 6,29 → 6,05 ms (−3,8 %) |
| 2550000/−2570000 | 1 422 chunks/s | **1 521 (+7,0 %)** | 6,47 → 6,16 ms (−4,8 %) |
