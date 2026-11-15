# Où passe le CPU des décorations

Mesures du 15/11. Tout le reste est sur GPU (densité, aquifères, beardifier, surface, minerais,
grottes, lumière, sections, biomes). Le débit de `/pregen` est limité par les 6 cœurs physiques
(DEBIT.md) : chaque pour-cent de CPU gagné par chunk se retrouve à peu près en débit.

Les décorations (`Decoration::decorate`, generator_decoration.rs) prennent **34 à 35 % du CPU**, soit
**2,5 à 3,0 ms par chunk**.

## Par feature

`HORIZON_FEATURE_TIMES=1` mesure le temps de chaque feature placée, des structures (par structure),
de la prédiction des minerais GPU et du reste de `decorate`. Le classement s'écrit dans latest.log
tous les 10 000 chunks (`FeatureTimes`, generator_decoration.rs). C'est un temps mural, mesuré
pendant que les autres workers tournent. Script `/tmp/ft1.sh "x z" ...` (binaires de profil, copie
gpu, rayon 1 600 blocs).

Par famille, en part du temps de décoration :

| Famille | 3150000/−3170000 (2 963 µs par chunk) | −3200000/3210000 (2 535 µs par chunk) |
|---|---|---|
| Minerais (prédiction GPU comprise) | ~31 % | ~35 % |
| Arbres | ~23 % | ~19 % |
| Sculk du deep dark | 7,5 % | 4,6 % |
| Géode d'améthyste | 5,8 % | 6,1 % |
| Grottes luxuriantes, lichen, lianes | 5,9 % | 5,7 % |
| Spéléothèmes | 4,9 % | 5,1 % |
| Neige et glace (`freeze_top_layer`) | 4,5 % | 5,4 % |
| Structures | 3,3 % | 4,8 % |
| Salles à monstres | 1,7 % | 1,8 % |
| Préparation de `decorate` | 1,6 % | 1,8 % |

Les postes les plus lourds de la première zone :

| Feature | Part | Par appel | Appels | Placés |
|---|---|---|---|---|
| `trees_birch_and_oak_leaf_litter` | 8,4 % | 909 µs | 10 924 | 71 % |
| prédiction des minerais | 8,3 % | 246 µs | 40 000 | |
| `sculk_patch_deep_dark` | 7,1 % | 1 059 µs | 7 946 | 7 % |
| `ore_dirt` | 6,1 % | 180 µs | 40 002 | 96 % |
| `amethyst_geode` | 5,8 % | 172 µs | 40 002 | 4 % |
| `trees_jungle` | 5,0 % | 1 539 µs | 3 820 | 69 % |
| `freeze_top_layer` | 4,5 % | 134 µs | 40 000 | 100 % |
| `pointed_dripstone` | 3,3 % | 251 µs | 15 726 | 70 % |
| `dark_forest_vegetation` | 3,2 % | 1 558 µs | 2 451 | 75 % |
| `ore_coal_lower` | 2,5 % | 72 µs | 40 001 | 98 % |
| `glow_lichen` | 2,1 % | 63 µs | 40 001 | 76 % |
| granite, diorite, tuf, andésite | ~2 % chacun | ~60 µs | 40 002 | 99 % |

Le monument coûte 25 à 29 ms par chunk qui en a un, mais ces chunks sont rares (0,4 à 1,6 %).

## Dans les features (profil perf)

Profils `/tmp/profil_b1.sh` (zones 2800000/−2820000 et 3250000/−3270000), analysés par
`/tmp/sous_deco.py` et `/tmp/sous_filtre.py` (fonctions en temps inclus et en temps propre sous un
cadre donné). Parts du CPU total :

- **Minerais, ~10 %** :
  - écriture des cellules que le GPU a calculées : `OrePlan::place` 5,8 %, dont
    `TerrainChunk::replace_row_raw` 4,4 %. Par cellule, il y a une lecture (`TerrainBlocks::block`),
    le test de la cible, puis `BlockSection::set`, qui recalcule la position et cherche l'état dans la
    palette.
  - prédiction : `predict_ores` 3,8 %, avec la chaîne de placement et le test de biome 2,5 %, les
    sphères 0,9 % et l'envoi au GPU 0,7 %.
- **Géode, 2,6 %**, dont 1,8 % pour le bruit : `NoiseStack::get3` est évalué pour chaque bloc de la
  boîte de 33 × 33 × 33, soit environ 4 ms par géode placée.
- **Neige et glace, 2,2 à 3,5 %** : la température du biome coûte 1,2 %, dont 0,9 % rien que pour son
  cache (hachage SipHash et éviction imitant le cache LRU de vanilla). S'y ajoutent les tests
  « doit neiger » et « doit geler » (biome par position), la survie de la neige et les écritures.
- **Sculk, 3,1 % là où il y a du deep dark** : `spread_all` 2,5 %, dont `can_attach_to` 1,1 % et les
  lectures de blocs 0,8 %.
- **Arbres, 2,9 %** : un coût éparpillé entre les mises à jour de forme (0,8 %), les écritures, les
  feuillages, l'ensemble des feuilles en ordre Java et les cartes de hauteur.

## Pistes, de la plus rentable à la moins

1. **Écriture des cellules de minerais** (~6 % du CPU). Remplacer une rangée directement dans la
   section : une recherche de section, une position par cellule pour la lecture et l'écriture, et la
   mémoïsation du test de cible par id de palette plutôt que par état. Mêmes écritures, dans le même
   ordre. Gain estimé : 2 à 3 % du CPU.
2. **Géode** (~2 %). Le bruit ne fait que décaler une somme qu'on compare à `outer_crust`, et un bloc
   sous ce seuil est ignoré sans tirage aléatoire. Avec une borne rigoureuse du bruit, inutile de
   l'évaluer là où la somme ne peut pas atteindre le seuil, c'est-à-dire presque toute la boîte. Il
   faut écrire cette borne : Horizon n'a pas l'équivalent du `maxValue` de vanilla. Gain estimé :
   environ 1,5 %.
3. **Neige et glace** (1,7 à 3,5 %). La température est une fonction pure ; son cache ne sert qu'à
   éviter des recalculs, et coûte plus que le calcul lui-même. Un calcul direct, et une seule recherche
   de biome par colonne. Gain estimé : 1 à 2 %.
4. **Prédiction des minerais** (~3 %) : tests de biome et sphères. Gain incertain, de l'ordre de 1 %.
5. **Sculk** (~2 % là où il y en a) : `can_attach_to` mis en cache par état et par face.
6. **Arbres** (~7 %) : pas de point chaud unique.

Les pistes 1 à 3 donneraient ensemble environ 5 % du CPU par chunk, donc de débit. Toutes peuvent rester
exactes, et se vérifient par la sonde, la parité vanilla et le mode test des minerais.

## Ce qui a été fait (15/11)

Originaux dans `/tmp/deco-avant/` sur le serveur.

1. **Neige et glace.** `BiomeTemperature::uncached` (biome_temperature.rs) calcule la température
   sans cache. `World::biome_should_freeze` et `biome_should_snow` (generator_decoration.rs) s'en
   servent, et lisent les liaisons météo par la référence à la décoration au lieu de les cloner à
   chaque appel (deux `Arc` partagés par tous les workers). `freeze_top_layer` passe de 134-136 à
   **48-49 µs par chunk (−64 %)**.
2. **Cellules de minerais.** `BlockSection::replace_with` (chunk.rs) lit l'emplacement une fois, passe
   l'état au test et écrit au même emplacement, avec la logique de `set` (déplacée dans `set_slot`).
   `TerrainChunk::replace_row_raw` (terrain.rs) s'en sert pour les sections qui ont des blocs non-air,
   et garde la lecture de `block` ailleurs (`air` dans une section d'air, `void_air` hors de la
   hauteur). Les gros blobs (terre, granite, diorite, tuf, andésite, gravier) gagnent 7 à 13 %, soit
   5 à 20 µs par chunk chacun.
3. **Géode : abandonné.** La borne rigoureuse du bruit est trop large pour sauter quoi que ce soit.
   Pour une couche de Perlin, `range` donne 2 ; une analyse de l'interpolation donne au mieux 1,5. Le
   `NormalNoise` de la géode a deux couches d'amplitude ~1,11, multipliées par 0,05 et par les 3 ou 4
   points de la géode : la marge dépasse 0,67, alors que le seuil `outer_crust` vaut environ 0,48.
   Aucun bloc n'était sauté, et la somme ajoutée rendait la géode 13 % plus lente. Le changement est
   retiré. Reste possible : mettre en cache, par cellule du réseau de Perlin, les gradients des huit
   coins, pour les blocs qui la partagent le long d'une rangée (exact, environ 0,5 % du CPU).

Vérifications :

- tests unitaires de chunk, terrain, température, minerais et neige : 19 passent ;
- sonde identique à l'octet près aux sorties d'avant sur 4 zones (dont une zone de snowy_taiga, de
  snowy_plains et de frozen_river), en chemin CPU comme en chemin tout GPU ;
- parité vanilla à 0 différence sur 785 chunks, tout sur GPU comme sur CPU.

Mesures. Temps par feature (`/tmp/ft1.sh`, mêmes zones que ci-dessus, copie gpub) : la décoration
passe de 2 963 à 2 921 µs par chunk et de 2 535 à 2 376 µs (géode encore incluse). Banc release
`/tmp/ab20.sh` (binaires d'avant dans `target/avant-deco/`, copie gpub, contre ceux d'après, copie gpu,
sans la géode) :

| Zone | Débit avant → après | CPU par chunk |
|---|---|---|
| 3400000/−3420000 | 1 584 → 1 593 chunks/s (+0,6 %) | 5,69 → 5,61 ms (−1,4 %) |
| −3430000/3440000 | 1 836 → 1 779 (−3,1 %) | 4,81 → 4,81 ms (0 %) |
| 3450000/−3470000 | 1 756 → 1 865 (+6,2 %) | 4,94 → 4,78 ms (−3,2 %) |

Environ −1,5 % de CPU par chunk : moins que les 3 à 5 % estimés, parce que la géode est abandonnée et
que l'écriture des cellules de minerais ne gagne qu'une dizaine de pour-cent, pas un tiers. Le débit
reste dans le bruit des zones.
