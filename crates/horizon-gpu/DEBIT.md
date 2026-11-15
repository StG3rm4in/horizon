# Ce qui limite `/pregen`

Mesures du 15/11, tout sur GPU (densité, aquifères, beardifier, surface, minerais, grottes, lumière,
sections, biomes), sur l'i5-10600K du serveur : 6 cœurs physiques, 12 threads, une seule barrette de
16 Go à 3 600 MT/s.

## Pendant un `/pregen`

Script `/tmp/pg1.sh x z` (analyse `/tmp/pg1_analyse.py`). Binaires de profil, zone neuve
2900000/−2920000, 1 450 chunks/s.

CPU par thread, en cœurs :

| Threads | Cœurs | Par thread |
|---|---|---|
| 11 workers | 5,22 | 0,47 |
| 6 threads de compression des sauvegardes | 1,69 | 0,28 |
| thread principal | 0,69 | |
| écriture des régions | 0,59 | |
| GPU (événements, lumière) | 0,10 | |
| **total** | **8,29** | |

Occupation : 73 % des 12 CPU logiques, de 71 à 79 % chacun ; les deux threads de chaque cœur
physique sont aussi chargés l'un que l'autre.

`perf stat` sur 10 s : IPC 1,15. Le contrôleur mémoire lit 6,5 Gio/s et écrit 4,3 Gio/s, soit
environ 40 % de ce qu'un seul canal DDR4-3600 permet.

Temps hors CPU des workers (`offcputime`, 10 s, piles utilisateur) : 49 s sur 110 s. Les causes :

| Cause | Temps |
|---|---|
| file de travail vide (`jobs::work`) | 16,8 s |
| attente du remplissage GPU d'un chunk (`Moteur::chunk_codes`) | 10,5 s |
| verrou des départs de structures (`StructureState::build_starts`) | 4,0 s |
| `OnceLock` partagés | 3,1 s |
| attente des minerais GPU | 2,3 s |
| attente des grottes GPU | 1,5 s |
| lecture des régions | 1,0 s |
| régions de densité | 0,5 s |
| régions de biomes | 0,3 s |

Trace (`HORIZON_PREGEN_TRACE=1`) :

| Étape | Valeur |
|---|---|
| générations en cours | 28 à 31 (limite 33, trois par worker) |
| décorations | 8 à 18 (limite 22) |
| lumières | 8 à 20 |
| file vide | 2 à 9 % des mises à jour |

## Selon le nombre de workers

Script `/tmp/ab19.sh "copie workers x z" ...` (release, rayon 1 600 blocs, zones neuves).
`HORIZON_WORKER_THREADS=N` donne N − 1 workers. Chaque nombre passe une fois dans chaque copie :

| Zone | 5 workers | 11 (défaut) | 15 workers |
|---|---|---|---|
| 3000000/−3020000 | 1 440 chunks/s (−15,9 %) | 1 711 | 1 603 (−6,3 %) |
| 3050000/−3070000 | 1 437 (−9,1 %) | 1 581 | 1 606 (+1,6 %) |
| 3100000/−3120000 | 1 386 (−8,9 %) | 1 521 | 1 547 (+1,7 %) |
| CPU par chunk | 4,05 à 4,42 ms | 5,18 à 5,83 ms | 5,65 à 6,11 ms |
| Cœurs occupés | 5,8 à 6,1 | 8,6 | 9,0 à 9,4 |

## Conclusion

**La limite, ce sont les 6 cœurs physiques.** « 8 à 9 cœurs occupés sur 12 » compte des threads
logiques : chaque cœur physique a presque toujours au moins un thread actif. Avec 5 workers, un par
cœur environ, `/pregen` garde 89 % de son débit. Les 6 workers de plus, qui partagent les mêmes cœurs
en hyperthreading, n'ajoutent que 12 % de débit, pour 30 % de CPU par chunk en plus. Au-delà, rien.

Rien d'autre n'est saturé :

- bande passante mémoire : ~40 % ;
- GPU : ~60 % ;
- thread principal : 0,69 cœur ;
- écriture des régions : 0,59 cœur ;
- threads de compression : 0,28 cœur chacun.

Les attentes des workers (GPU, structures, file vide) laissent le cœur à l'autre thread : les
supprimer ne rendrait que la part de l'hyperthreading.

Les conclusions de PREGEN.md et GPU_ATTENTE.md (« débit pas limité par le CPU, 8 à 9 cœurs sur 12 »)
se trompaient pour la même raison : le CPU était déjà la limite, par ses cœurs physiques.

## Ce qui ferait aller plus vite

Uniquement moins de CPU par chunk. Les postes, d'après le profil du 15/11 avec les biomes sur GPU
(BIOMES.md) :

| Poste | Part du CPU | Remarque |
|---|---|---|
| Décorations | 29,6 % | |
| Sauvegarde | 17,9 % | encodage NBT et allocations, dont `mi_page_malloc_zero` 3 % |
| Attente et pilote du GPU | 11,2 % | |
| Compression zlib | 9,9 % | niveau 6, comme vanilla |
| Thread principal | 7,8 % | |

Pour zlib, Horizon lit déjà `region-file-compression` de server.properties (`deflate`, `lz4`, `gzip`,
`none`, comme vanilla depuis 1.20.5). Les copies de banc sont en `deflate`.


## Remesure en lz4 (15/11 au soir)

Avec les régions en lz4 (copies de banc), les palettes sans recompactage (SAUVEGARDE.md), la neige et
les minerais optimisés (DECORATIONS.md). Script `/tmp/lim1.sh` : profil, diagnostic par thread, banc
des workers. `/tmp/pg1.sh` lance maintenant perf stat avec `LC_ALL=C`, sinon la virgule décimale de la
locale casse la lecture des compteurs.

Selon le nombre de workers (`/tmp/ab19.sh`, release, zones neuves) :

| Zone | 5 workers | 11 (défaut) | 15 workers |
|---|---|---|---|
| 3950000/−3970000 | 1 422 chunks/s (−16,4 %) | 1 701 | 1 730 (+1,7 %) |
| −3980000/3990000 | 1 686 (−14,2 %) | 1 965 | 2 014 (+2,5 %) |
| CPU par chunk | 3,06 à 3,67 ms | 4,39 à 5,19 ms | 4,61 à 5,48 ms |

Pendant un `/pregen` (binaires de profil, zone 3900000/−3920000, 1 768 chunks/s) :

| Threads | Cœurs |
|---|---|
| 11 workers | 5,31 (0,48 chacun) |
| 6 threads de compression | 0,82 (0,14 chacun) |
| thread principal | 0,67 |
| écriture des régions | 0,36 |
| **total** | **7,34** (65 % des CPU logiques, contre 73 % en deflate) |

- Mémoire : 6,7 Gio/s en lecture et 4,4 en écriture (~40 %) ; IPC 1,08.
- Temps hors CPU des workers : 39 % du temps. File vide 11 %, remplissage GPU du chunk
  (`chunk_codes`) 10 %, verrou de `build_starts` 5 %.

**La limite reste le CPU, par ses 6 cœurs physiques.** 5 workers gardent 85 % du débit, 15 workers
n'ajoutent que 2 %. Le thread principal (0,67 cœur) n'est pas un goulot ; la compression et l'écriture
des régions non plus.

Profil (`/tmp/profil_b1.sh`, zone 3850000/−3870000, 1 861 chunks/s avec les binaires de profil) :

| Poste | Part du CPU |
|---|---|
| Décorations | 36,2 % |
| Sauvegarde (arbre NBT, écriture) | 14,3 % |
| Attente et pilote du GPU | 12,5 % |
| Thread principal | 8,4 % |
| Génération (reste) | 6,7 % |
| Structures | 4,8 % |
| Finalisation des chunks | 4,3 % |
| Biomes et climat | 3,2 % |
| Grottes (CPU) | 2,6 % |
| Compression (lz4) | 2,3 % |

- La fonction la plus coûteuse en temps propre est `_mi_page_malloc_zero` (4,6 %), le chemin
  d'allocation de mimalloc. `/tmp/appelants_alloc.py` montre qu'il est éparpillé ; le plus gros
  appelant est l'arbre NBT de la sauvegarde (chaînes `JavaString::from`, compounds), environ 1,5 %.
- Le pilote CUDA (`/tmp/sous_gpu.py`) pèse 5,2 % : les lancements de la surface (`enfiler`) 1,4 %, et
  les envois de la lumière, des grottes et des minerais 2,1 %.
