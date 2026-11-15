# Grottes sur GPU (`HORIZON_GPU_CAVES`)

## Ce qui coûte

Profil `/pregen` du 15/11 (densité, surface et minerais sur GPU, 1 143 chunks/s avec les binaires de
profil, `/tmp/sous_grottes.py`) : les grottes représentent **12,4 % du CPU**.

| Poste | Part du CPU |
|---|---|
| Masque (`generate_carving_mask`) | 4,65 % |
| dont tracé des tunnels (`CaveCarver::carve`) | 3,42 % |
| dont biome des origines (`carver_biome`) | 0,72 % |
| dont canyons | 0,24 % |
| Application du masque (`apply_carving_mask_shared`) | 5,51 % |
| dont aquifère (`compute_substance`) | 2,35 % |
| dont matériau sous un bloc d'herbe creusé (`top_material_cell`) | 1,19 % |
| dont mise en place du moteur de règles | 1,02 % |
| dont écritures | 0,49 % |
| Construction de l'aquifère CPU (`NoiseBasedAquifer::new`), qui ne servait plus qu'aux grottes | 2,29 % |

Le masque d'un chunk réunit les carvers des 17 × 17 chunks d'origine autour de lui : chaque origine
est retracée pour chacun des 289 chunks qu'elle peut toucher.

## Pourquoi le masque peut être calculé ailleurs

- Les tirages d'un appel de carver (`WorldCarver::carve` après son test de départ) ne dépendent pas
  du chunk cible : nombre de départs, positions, salles, paramètres et graines des tunnels, canyon et
  sa table de largeurs. Seuls le tracé (`can_reach` arrête un tunnel) et la rasterisation des
  ellipsoïdes dépendent de la cible.
- Le masque est une union de cellules : l'ordre des ellipsoïdes ne compte pas.
- La substance d'une cellule creusée (`compute_substance` avec une densité 0) ne dépend que de sa
  position : les statuts des cellules d'aquifère sont des fonctions pures, déjà calculées sur le GPU
  pour toute la région.

## Déroulement

1. **Planification, une fois par origine** (`WorldCarver::plan`, cache `carver_plans` partagé et par
   worker) : les mêmes tirages que `carve`, dans le même ordre, rangés en salles, tunnels (graine,
   position, épaisseur, lacet, tangage, fin, échelle verticale, multiplicateurs du run, plancher) et
   canyons (départ, état de leur flux après la table des largeurs, table). `carve` n'est pas modifié :
   la sonde de vanilla vérifie aussi l'état de l'aléatoire après une erreur de sortie.
2. **Début du chunk** (`Pipeline::submit_caves`) : les éléments des 289 origines qui peuvent atteindre
   le chunk (un chemin avance d'au plus un bloc par pas) partent sur le GPU avec un compteur par appel
   (le budget de 2^24 cellules testées de Horizon).
3. **GPU, `kgr_trace`** (`grottes.cu`, un bloc de 128 threads par élément) : le thread 0 trace le
   tunnel et ses branches (pile), ou le canyon, exactement comme `CaveRun::tunnel` et
   `CanyonCarver::trace` (LegacyRandom, table de sinus de Horizon, float et double `_rn`, sans
   contraction). Il garde les ellipsoïdes qui passent `can_reach` et le test de portée. Le bloc les
   rasterise ensuite comme `carve_ellipsoid_with_budget` (axes précalculés par ellipsoïde) dans un
   masque en mémoire partagée, réuni au masque du chunk par `atomicOr`.
4. **Remplissage du chunk** : dès que la région du chunk est sur le GPU, `kgr_substance` (module des
   aquifères) calcule la substance de chaque cellule du masque, rangée dans l'ordre des bits, et le
   résultat est copié, toujours sur le flux du lot, hors du chemin critique du worker.
5. **Étape des grottes** : le CPU attend le lot (déjà fini), reconstruit le `CarvingMask` à partir des
   mots et l'applique avec `apply_carving_mask_with`, qui lit les substances au lieu de l'aquifère. Le
   moteur de règles du matériau sous l'herbe n'est construit que si un chunk en a besoin.
   L'aquifère CPU n'est plus construit que si le chunk passe par le remplissage CPU ou par la
   substance CPU.

Le chunk repasse par le CPU, sans changer le résultat, si le GPU signale une erreur (capacité,
borne aléatoire, fournisseur de rayon non pris en charge), si un appel dépasse son budget (le CPU lève
alors l'erreur lui-même), ou si le masque a plus de cellules que les codes copiés.

## Variables

- `HORIZON_GPU_CAVES=1` : grottes sur GPU. La substance des cellules demande en plus la densité et
  les aquifères sur GPU (`HORIZON_GPU_DENSITY`) ; sans eux, seul le masque vient du GPU.
- `HORIZON_GPU_CAVES=test` : compare chaque masque à `generate_carving_mask` et chaque substance à
  l'aquifère CPU, construit alors pour chaque chunk. En cas d'écart, le chunk passe par le CPU.

À l'arrêt, une ligne `GPU grottes :` donne le nombre de lots, d'éléments et de cellules par lot, les
temps d'envoi et d'attente, les replis et les écarts.

## Vérifications

- Test unitaire `tests/carver_plan.rs` : pour les grottes, les grottes du Nether et les canyons de
  vanilla, 120 graines et 85 cibles chacune, `plan` puis `execute` creusent les mêmes cellules que
  `carve`.
- Sonde `chunk_dump` (`/tmp/gr1.sh`), 3 zones de 256 chunks (0,0 ; −1250,940 ; 750,−1270) : sortie
  identique au CPU à l'octet près avec le masque GPU seul en mode test, tout GPU en mode test, et tout
  GPU. 0 écart de masque, 0 écart de substance, 0 repli. 80 à 120 éléments et 850 à 1 700 cellules
  creusées par chunk en moyenne (10 903 au plus).
- Parité vanilla (`tools/test_chunk_generation_parity.py --seeds 1`) : **0 différence sur 785
  chunks**, avec `HORIZON_GPU_CAVES=1` comme avec `test`. Par dimension, en mode test :
  - overworld (627 chunks comparés, 64 sites de structures) : 3 479 lots, dont 3 378 avec l'aquifère
    GPU (les autres ont un masque vide), 0 écart de masque, 0 écart de substance ;
  - Nether (113 chunks) : 611 lots, masque GPU seul (pas de densité GPU), 0 écart, dont un chunk de
    16 384 cellules creusées.

## Mesures

**Version 1** : un plan par origine (289 recherches par chunk), trois tampons GPU par lot.
- Bancs `/pregen` (`/tmp/ab10.sh`, rayon 1 600 blocs, densité + surface + minerais sur GPU dans les
  deux variantes) :
  - zone 600000/−620000 : 1 010 → 1 040 chunks/s (+3,0 %), CPU 10,09 → 9,72 ms par chunk ;
  - zone −630000/640000 : 1 036 → 1 095 chunks/s (+5,7 %), CPU 8,92 → 8,38 ms par chunk.
- Profil (zone 700000/−720000, 1 096 chunks/s avec les binaires de profil, `/tmp/sous_grottes2.py`) :
  les grottes passent de 12,4 % à environ 4,9 % du CPU.
  - préparation de l'envoi (`submit_caves`) : 1,77 %, dont 0,9 % de recherches dans les caches de
    plans (289 par chunk, sous verrou quand le cache du worker se vide) et 0,4 % de pilote CUDA ;
  - application du masque : 2,76 %, dont 1,59 % pour le matériau sous l'herbe creusée, lui-même à
    1,34 % dans la surface préliminaire (`find_top_surface` interpolé), recalculée à chaque cellule ;
  - aquifère GPU enfilé : 0,23 % ; attente du lot : 0,13 % ; aquifère CPU : 0,04 %.
  - L'attente et le pilote du GPU passent de 6,7 à 8,0 %.

**Version 2** :
- plans par blocs de 8 × 8 origines : 9 recherches par chunk ;
- un seul tampon GPU par lot : une copie à l'aller (éléments, largeurs et zéros de la sortie), une au
  retour (sortie et substances) ; 5 appels au pilote par chunk au lieu de 8 ;
- `MaterialDensitySources::retarget` garde la surface préliminaire tant que la colonne ne change pas
  (le même plan donne les mêmes valeurs).

Vérifications de la version 2 : sonde identique sur les 3 zones dans les 3 modes ; parité vanilla à
0 différence (complète, et overworld en mode test) ; mode test à 0 écart de masque et 0 écart de
substance (overworld 3 479 lots, Nether 611 lots). Envoi d'un lot : 9 à 12 µs au lieu de 13 à 18 µs.

Bancs de la version 2 (zones neuves) :

| Zone | Sans grottes GPU | Avec | CPU par chunk |
|---|---|---|---|
| 800000/−820000 | 957 chunks/s | 996 chunks/s (+4,1 %) | 10,73 → 10,15 ms (−5,4 %) |
| −830000/840000 | 1 119 chunks/s | 1 185 chunks/s (+5,9 %) | 8,41 → 7,74 ms (−8,0 %) |
| 850000/−870000 | 1 066 chunks/s | 1 111 chunks/s (+4,3 %) | 9,69 → 9,08 ms (−6,3 %) |

Le GPU passe de 27-31 % à 33-38 % d'utilisation. Le harnais de parité n'affiche que les 5 premières
lignes d'erreur de la sonde : avec `HORIZON_GPU_ORES` en plus, la ligne `GPU grottes` n'y apparaît
pas ; il faut lancer le mode test sans les minerais pour la lire.
