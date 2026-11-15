# Lumière sur GPU (`HORIZON_GPU_LIGHT`)

## Ce qui coûte

Profil `/pregen` du 15/11 (densité, surface, minerais et grottes sur GPU, `/tmp/sous_lumiere.py`) :
la lumière représente **21 % du CPU**.

| Poste | Part du CPU |
|---|---|
| `light_chunk` (tâche `Job::Light`) | 16,3 % |
| dont décodage des palettes des 9 chunks (`unpack_ids_mapped`) | environ 3,3 % |
| dont recopie des états dans la région (`states_of`) | environ 1,5 % |
| dont remplissage en largeur du ciel et des blocs (`spread`) | environ 7 % |
| dont colonnes de ciel | environ 0,8 % |
| dont mise en forme du chunk central (`pack_lighting`, `Light::from_slice`) | 1,55 % |
| Vues des chunks voisins (`light_view`, `block_column`) | 4,6 % |

`light_chunk` éclaire un chunk sur ses 3 × 3 voisins (48 × 48 × h) : chaque chunk est décodé neuf fois,
une fois par tâche de lumière de ses voisins.

## Pourquoi le GPU donne la même lumière

Les valeurs finales du moteur de vanilla (et du remplissage de Horizon) sont le **point fixe** du
remplissage : pour chaque cellule, le maximum sur tous les chemins depuis une source de
`source − Σ max(1, atténuation)`, sans traverser une face occultée (`block_masks`). Ce point fixe est
unique : un remplissage par vagues parallèles, dans n'importe quel ordre, l'atteint aussi. Les entiers
n'ont pas d'arrondi : il n'y a rien d'autre à reproduire que les règles.

## Déroulement

1. **Tâche de lumière** (`streaming.rs`, `Job::Light`) : `light::gpu_light_submit` calcule la hauteur
   (`top_section`, comme `light_chunk`) et copie, pour les 9 chunks, les sections telles qu'elles sont
   stockées : palette (traduite en infos de lumière par `LightTables::info`) et mots compactés. Pas de
   cache sur le GPU, donc rien à invalider. La tâche envoie le lot et rend la main tout de suite.
2. **GPU** (`lumiere.cu`, sur le flux du lot) :
   - `klu_etats` : un bloc par (chunk, section), décode les états et l'émission, et range les cellules
     qui émettent dans les semis ;
   - `klu_ciel` : une colonne par thread, descend jusqu'à la première arête occultée
     (`packed_edge_occluded`) et écrit la colonne du ciel ;
   - `klu_remplir` : un bloc de 1 024 threads, le ciel puis les blocs, par vagues jusqu'à ce qu'aucune
     cellule ne monte. Une cellule monte par CAS sur le mot de 32 bits qui contient son octet ; les
     semis du ciel sont ceux de `light_chunk` (sources entre le fond de leur colonne et le plus profond
     de leurs voisines) ;
   - `klu_sortie` : un bloc par section du chunk central, qui range sa lumière comme
     `Light::from_slice` (uniforme, ciel seul, bloc seul ou complet, nibbles déjà empaquetés).
3. **Fil `horizon-gpu-light`** (un par dimension) : attend chaque lot sur un événement à attente
   bloquante (le fil dort), construit les `Light` et envoie `Done::Lit` à la boîte aux lettres, comme
   la tâche l'aurait fait.

Un lot que le GPU ne prend pas (32 lots déjà en cours, colonne d'une autre hauteur, erreur GPU) est
éclairé sur le CPU comme avant. Chaque lot réserve 14 Mo de mémoire GPU (24 sections).

## Variables

- `HORIZON_GPU_LIGHT=1` : lumière sur GPU.
- `HORIZON_GPU_LIGHT=test` : le fil compare chaque chunk à `light_chunk` (même genre de stockage,
  mêmes 4 096 valeurs par section) ; en cas d'écart, la lumière du CPU est gardée et l'écart compté.

Une ligne `GPU lumiere :` est journalisée (latest.log) à chaque puissance de deux de chunks : chunks,
octets envoyés, temps d'envoi (sur le worker) et d'attente (sur le fil), lots en cours au plus,
erreurs, replis CPU et écarts.

## Vérifications

- Mode test (`/tmp/lu1.sh`, serveur compilé dans parity-target) : zone 900000/−920000 (rayon 400) puis
  1100000/−1120000 (rayon 1 200) : **16 384 chunks éclairés par le GPU et comparés à `light_chunk`,
  0 écart**, 0 erreur. En mode test, le fil recalcule chaque chunk sur le CPU : il devient le goulot,
  les 32 lots sont pris et une partie des chunks est éclairée directement sur le CPU (replis), ce qui ne
  change pas le résultat.
- Mode normal : 0 repli, 21 lots en cours au plus ; envoi 122 à 129 µs par chunk sur le worker
  (environ 180 Ko copiés), attente du GPU 250 µs sur le fil.

## Mesures

Bancs `/pregen` (`/tmp/ab11.sh`, release, rayon 1 600 blocs, zones neuves ; densité, surface,
minerais et grottes sur GPU dans les deux variantes) :

| Zone | Sans lumière GPU | Avec | CPU par chunk |
|---|---|---|---|
| 1000000/−1020000 | 1 117 chunks/s | **1 333 chunks/s (+19,3 %)** | 9,09 → 6,95 ms (−23,5 %) |
| −1030000/1040000 | 1 124 chunks/s | **1 303 chunks/s (+16,0 %)** | 8,24 → 6,33 ms (−23,2 %) |
| 1050000/−1070000 | 1 173 chunks/s | **1 405 chunks/s (+19,8 %)** | 8,84 → 6,74 ms (−23,8 %) |

Le GPU passe de 28-42 % à 63-68 % d'utilisation, et de 4,9 à 5,3 Go de mémoire. Le serveur n'occupe
plus que 8,2 à 9,3 cœurs sur 12 (10 avant) : le débit n'est plus limité seulement par le CPU des
workers.

Profil avec la lumière sur GPU (zone 1150000/−1170000, 1 337 chunks/s avec les binaires de profil) :
la lumière tombe de 16,6 % à **0,9 % du CPU**. Les plus gros postes sont maintenant les décorations
(23,3 %), l'attente et le pilote du GPU (12,7 %, contre 8,0 %, dont 8,3 % dans libcuda : les appels
synchrones de la densité et de la surface attendent plus longtemps un GPU plus chargé), la sauvegarde
(12,5 %), l'écriture des états finaux (9,1 %), le thread principal (8,8 %), les biomes (7,6 %) et la
compression zlib (7,5 %).

Tests unitaires de la lumière de horizon-client-render (`cargo test -p horizon-client-render --lib
light`) : 89 passent.


## Cache des sections sur le GPU (15/11 au soir)

### Pourquoi

Un lot de lumière lit les 9 chunks du voisinage 3 × 3. L'envoi recopiait les mots et les palettes de
toutes leurs sections non uniformes : environ 180 Ko par chunk, et chaque chunk recopié 9 fois. Le
profil en lz4 met cet envoi à **4,1 % du CPU** (copie, conversion des palettes, pilote ; GPU_ATTENTE.md).

### Comment

`LightGpu::submit_sections` (lumiere.rs) reçoit les sections d'un lot section par section
(`LightInput` : absente, uniforme, ou compacte avec son `Arc<Section>`) ; `gpu_light_submit` (light.rs)
les décrit comme le remplissage d'avant.

- **Identité** : la clé d'une section est l'adresse de son `Arc` (`Arc::as_ptr`). L'emplacement garde
  un `Weak` vers elle. Tant qu'il vit, aucune autre allocation ne prend cette adresse, et
  `Arc::make_mut` déplace la section au lieu de la changer en place : un `Arc` encore à cette adresse
  a donc le même contenu. Toutes les écritures de sections passent par `Arc::make_mut`. Un `Arc` fort
  aurait forcé chaque écriture de lumière à cloner la section ; avec le `Weak`, elle ne fait que la
  déplacer (quelques centaines d'octets).
- **Emplacements** : 16 384 de 1 152 mots sur le GPU (151 Mo), qui contiennent les infos de la palette
  puis les mots. Une palette globale de plus de 1 024 mots reste en clair.
- **Descripteurs** : les genres 4 (palette) et 5 (globale) de `klu_etats` lisent dans le cache
  (`CACHE`) au lieu de l'entrée.
- **Remplissage** : une section absente du cache est envoyée en clair dans l'entrée, comme avant, puis
  `klu_cache` la recopie dans son emplacement, sur le flux du lot. L'événement `rempli` de la ressource
  est enregistré juste après.
- **États d'un emplacement** :
  - *réservé* : le remplissage n'est pas encore enfilé, et les autres lots envoient la section en
    clair ;
  - *en cours* : l'événement est enregistré, et les autres lots l'attendent (`stream.wait`). Si la
    ressource enregistre plus tard le même événement, c'est plus loin sur le même flux, donc encore
    après ce remplissage ;
  - *prêt* : le lot qui l'a rempli est fini.
- **Épinglage et éviction** : un lot épingle les emplacements qu'il lit ou remplit, jusqu'à `finish`.
  L'éviction (horloge à seconde chance) ne prend qu'un emplacement prêt et non épinglé. Si le flux d'un
  lot échoue, ses remplissages sont oubliés.
- **Repli** : sans emplacement libre, ou pour une section que remplit un autre lot, la section passe
  en clair, comme avant.

### Variables

- `HORIZON_GPU_LIGHT_CACHE` : `0` sans cache, un nombre d'emplacements, `test`. Par défaut, 16 384.
- `HORIZON_GPU_LIGHT_CACHE=test` : chaque section trouvée en cache est aussi envoyée en clair, et
  `klu_etats` compare les deux cellule par cellule (drapeau `LU_E_CACHE`, compté dans le bilan). Se
  combine avec `HORIZON_GPU_LIGHT=test`, qui compare la lumière du chunk à celle du CPU.

### Vérifications

- Les 89 tests unitaires de la lumière passent.
- `/pregen` de rayon 800, zone 4400000/−4420000, cache en mode test et lumière en mode test :
  **0 écart du cache, 0 écart au CPU** (53 % des sections trouvées).
- `/pregen` de rayon 800, zone 4450000/−4470000, cache réel et lumière en mode test : **0 écart au
  CPU**.
- Les lots « repassés par le CPU » de ces essais viennent du mode test : le thread de complétion
  recalcule chaque chunk sur le CPU, ne suit plus, et les 32 lots en vol sont pris.

### Mesures

Banc `/tmp/ab25.sh` (release, tout sur GPU, lz4 ; même binaire, `HORIZON_GPU_LIGHT_CACHE=0` dans la
copie gpub contre le cache dans la copie gpu ; zones neuves) :

| Zone | Débit sans → avec | CPU par chunk |
|---|---|---|
| 4500000/−4520000 | 1 928 → 1 999 chunks/s (+3,7 %) | 4,33 → 4,20 ms (−3,0 %) |
| −4530000/4540000 | 1 956 → 2 055 (+5,1 %) | 4,32 → 4,16 ms (−3,7 %) |
| 4550000/−4570000 | 2 001 → 2 070 (+3,5 %) | 4,19 → 4,01 ms (−4,3 %) |

**−3,7 % de CPU par chunk, +4,1 % de débit.**

| Mesure | Sans cache | Avec cache |
|---|---|---|
| Sections trouvées en cache (rayon 1 600) | — | 73 % |
| Envoyé par chunk | 180 Ko | 48 Ko |
| Envoi | 124 µs par chunk | 75 µs par chunk |
| Mémoire GPU | — | +224 Mo |
