# Attente du GPU

## Ce qui coûte

Profil `/pregen` du 15/11 avec tout sur GPU (densité, aquifères, beardifier, surface, minerais,
grottes, lumière ; `/tmp/sous_gpu.py`) : l'étape « attente et pilote du GPU » pèse **12,7 % du CPU**,
dont 8,9 % dans libcuda et cudarc. Par appelant :

| Appelant | Part du CPU |
|---|---|
| `chunk_codes` (attente du remplissage et de la surface d'un chunk) | 4,6 % (+ 0,4 % avec structures) |
| `SurfaceGpu::enfiler` (lancements) | 0,7 % |
| envois de la lumière, des grottes et des minerais | 1,2 % |
| `OreGpu::rows` (`as_slice` à chaque blob) | 0,55 % |
| calcul des régions | 0,5 % |
| attentes des minerais et des grottes | 0,4 % |

`PinnedHostSlice::as_slice` de cudarc appelle `cuEventSynchronize` à chaque fois, même sans suivi
d'événements.

## Ce qui a été essayé

Bancs `/pregen` (`/tmp/ab12.sh` et `/tmp/ab13.sh`, release, rayon 1 600 blocs, zones neuves, tout sur
GPU) :

| Variante | 1200000/−1220000 | −1230000/1240000 | 1250000/−1270000 |
|---|---|---|---|
| sans priorités (`HORIZON_GPU_PRIORITY=0`) | 1 376 chunks/s, 6,76 ms | 1 335, 6,26 ms | 1 386, 6,73 ms |
| flux prioritaires | 1 381, 6,74 ms | 1 329, 6,28 ms | 1 386, 6,75 ms |
| flux prioritaires et `HORIZON_GPU_SYNC=yield` | 1 369, 6,84 ms | 1 333, 6,35 ms | 1 376, 6,83 ms |

(chunks/s, puis temps CPU par chunk.) Ni les priorités ni `yield` ne changent quoi que ce soit : le
débit n'est plus limité par l'attente du GPU ni par le CPU (8 à 9 cœurs occupés sur 12). Il l'est par
l'avancée de `/pregen`, qui balaie la zone par bandes d'au moins 1 024 chunks et n'avance une bande que
lorsque ses premières rangées sont éclairées.

L'attente active ne coûte donc que du CPU. Le mode bloquant (le thread dort), écarté quand le CPU était
saturé (appels de 0,9 ms au lieu de 0,5), ne freine plus rien :

| Variante | 1300000/−1320000 | −1330000/1340000 |
|---|---|---|
| attente active (défaut d'alors) | 1 307 chunks/s, 7,04 ms | 1 247, 6,58 ms |
| `HORIZON_GPU_SYNC=block` | 1 303, 6,62 ms (−6,0 %) | 1 256, 6,30 ms (−4,3 %) |

## Ce qui est gardé

- **Attente bloquante par défaut** : `HORIZON_GPU_SYNC` vaut `block` sauf demande contraire (`spin`,
  `yield`, `auto` pour l'heuristique de CUDA, souvent active).
- Flux du chemin critique (remplissage, surface, régions) à la plus haute priorité du GPU, les lots
  asynchrones à la priorité par défaut. Sans effet mesuré aujourd'hui ; `HORIZON_GPU_PRIORITY=0` les
  remet à égalité.
- `OreGpu::wait` lit le tampon de retour une fois ; `rows` ne rappelle plus le pilote à chaque blob.
  Sonde des minerais en mode test : identique, 0 écart.

Confirmation avec le nouveau défaut, sans variable (`/tmp/at2.sh`, zones neuves) :

| Variante | 1400000/−1420000 | −1430000/1440000 |
|---|---|---|
| `HORIZON_GPU_SYNC=auto` (ancien défaut) | 1 200 chunks/s, 7,64 ms | 1 239, 6,56 ms |
| défaut (bloquant) | 1 179, 7,26 ms (−5,0 %) | 1 262, 6,15 ms (−6,3 %) |

Le débit varie de −1,7 % et +1,9 % : c'est le bruit d'une zone à l'autre.

## Correction du 15/11

Le débit était déjà limité par le CPU : « 8 à 9 cœurs sur 12 » compte des threads logiques, et le
10600K n'a que 6 cœurs physiques. Voir DEBIT.md.


## Appels et copies autour du GPU (15/11 au soir)

Profil en lz4, avec l'écriture NBT directe (`/tmp/sous_gpu2.py` : par fonction de horizon-gpu et par
sorte de feuille). Le poste « attente et pilote du GPU » contient aussi le travail CPU du crate
horizon-gpu, pas seulement les appels du pilote :

| Origine | Part du CPU |
|---|---|
| Envoi de la lumière (`gpu_light_submit` → `LightGpu::submit`) | 3,7 % |
| Recopies de la surface (`chunk_blocks_surface`) | 1,3 % |
| `chunk_codes`, dont `memchr` 0,9 % | 1,4 % |
| Envois des minerais et des grottes | ~0,9 % |
| Lancements de la surface et de la substance | ~0,9 % |
| Pilote hors horizon-gpu (réveils, interruptions) | 1,1 % |

### Ce qui a été changé

- **Le contrôle des codes** (`chunk_codes`, `chunk_codes_beard`, lib.rs). Les 98 304 codes qu'écrit le
  GPU étaient lus deux fois depuis la mémoire : `contains(&-1)`, puis la copie. Ils sont maintenant
  copiés d'abord, puis contrôlés dans le cache. En cas d'erreur, `out` revient à zéro. Le générateur
  ne lit ces codes que si le GPU a réussi (`from_codes`). Le `memchr` passe de 0,92 % à 0,19 % du
  CPU.
- **Les cartes de hauteur de la sauvegarde** (SAUVEGARDE.md). `Heightmap::prime_sections`
  (horizon-world heightmap.rs, trait `HeightmapSections`) calcule les mêmes hauteurs section par
  section. La vue de la sauvegarde donne pour chaque cellule un masque des cartes qu'elle couvre,
  tiré de la palette de la section par `unpack_ids_mapped`. On ne cherche donc plus la section ni
  l'état de chaque bloc. Le coût passe de 1,51 % à 0,93 % du CPU.

Vérifications :

- tests unitaires `prime_sections_matches_prime` (colonnes aléatoires) et
  `heightmaps_by_sections_match_cells` (horizon-client-lib, états du jeu) ;
- `HORIZON_HEIGHTMAP_TEST=1` calcule chaque carte des deux façons et compare. `/pregen` en mode test
  (rayon 800, zone 4150000/−4170000) : **plus de 16 384 chunks, 0 écart** ;
- sonde identique sur 4 zones en tout GPU, et parité vanilla à 0 différence sur 785 chunks.

Banc `/tmp/ab24.sh` (binaires d'avant dans `target/avant-gpu/`, copie gpub ; lz4) :

| Zone | Débit avant → après | CPU par chunk |
|---|---|---|
| 4200000/−4220000 | 2 207 → 2 225 chunks/s (+0,8 %) | 3,71 → 3,65 ms (−1,6 %) |
| −4230000/4240000 | 1 984 → 1 979 (−0,3 %) | 4,39 → 4,37 ms (−0,5 %) |
| 4250000/−4270000 | 2 009 → 1 999 (−0,5 %) | 4,18 → 4,17 ms (−0,2 %) |

Environ −0,8 % de CPU par chunk, dans le bruit du banc. Le profil y voit −1,3 %.

### Ce qui reste

Profil après ces changements (zone 4300000/−4320000, 2 082 chunks/s avec les binaires de profil) : le
poste pèse 14,6 % du CPU (pilote 5,0 %, libc 4,3 %, code 2,7 %, noyau 2,3 %).

- **Envoi de la lumière : 4,1 %.** Chaque chunk éclairé copie les mots des sections de ses 9 chunks
  (voisinage 3 × 3, ~430 Ko) dans le tampon d'envoi : chaque chunk est recopié 9 fois. Le remède
  serait un cache des sections sur le GPU. Il serait indexé par l'identité des `Arc<Section>`, qu'il
  garderait vivantes, de sorte qu'une modification passe par une copie sur écriture. Il faudrait des
  emplacements recyclés sans libérer ceux qu'un travail en vol utilise, et une synchronisation entre
  flux pour qu'un travail ne lise pas une section encore en cours d'envoi. Gain estimé : 2 à 3 %.
- **Recopies de la surface : 1,9 %.** Les sections compactes et les codes sont recopiés du tampon
  épinglé vers des `Vec`. Les lire en place demanderait de garder le tampon jusqu'au remplissage du
  chunk.
