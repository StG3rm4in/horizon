# Horizon GPU

La génération du monde sur le GPU et un tick serveur accéléré, pour Horizon.

[English](README.md)

Ce dépôt est un fork d'[Horizon](https://github.com/horizon-rust/horizon), une réimplémentation en Rust de Minecraft: Java Edition 26.3 avec un client, un serveur et un lanceur. La branche `parallel-mobs` ajoute en option les îlots de mobs et l'apparition naturelle sur des threads auxiliaires à la génération sur GPU et au travail sur le tick de `gpu-worldgen`.

1. **La génération du monde sur le GPU.** Dans l'Overworld, NVIDIA CUDA peut calculer la densité finale, les aquifères, le beardifier, les règles de surface, une partie de la génération des minerais, les cellules des géodes d'améthyste, le creusement des grottes, la lumière des chunks générés, les sections compactes et les biomes multi-bruit. Le Nether prend aussi en charge sur GPU la densité, le creusement, les règles de surface et les biomes. L'End prend en charge la densité sur GPU. Dans ces deux dimensions, le sélecteur global de fluides remplace les aquifères actifs. Les biomes de l'End et les sections compactes du Nether restent sur le CPU. Voir les [notes sur le Nether et l'End](../crates/horizon-gpu/NETHER_END.md).
2. **Un tick du serveur dédié plus rapide.** `exact` partage le travail compatible tout en conservant les flux aléatoires en série. Le mode optionnel d'îlots peut regrouper les mobs admissibles et exécuter l'apparition naturelle en couloirs. Ces chemins peuvent utiliser leurs propres flux aléatoires, ce qui peut produire des tirages différents de vanilla.

## Résultats

Le serveur de mesure est équipé d'un Intel i5-10600K (6 cœurs, 12 threads), de 32 Go de mémoire en double canal et d'une NVIDIA RTX 3080.

### Génération du monde (`/pregen`, serveurs neufs)

La comparaison du 17 novembre 2026 utilise le même build du fork pour les passages CPU seul et CPU + GPU, avec deux répétitions.

| Dimension | CPU seul | CPU + GPU | Rapport de débit |
|---|---:|---:|---:|
| Overworld | 853 chunks/s, 12,92 ms CPU/chunk | **2 287 chunks/s, 3,37 ms CPU/chunk** | ×2,68 |
| Nether | 1 340 chunks/s | 3 183 chunks/s | ×2,38 |
| End | 3 586 chunks/s | 6 879 chunks/s | ×1,92 |

Dans les passages GPU profilés sur l'Overworld, le débit est limité par les 6 cœurs physiques du CPU plutôt que par la saturation du GPU.

### Tick du serveur dédié, 100 bots

Chaque chiffre est la durée moyenne du tick sur un passage. À 20 ticks par seconde, le budget est de 50 ms par tick.

| Build et mode | Date | Moyenne en ms par tick |
|---|---|---:|
| Référence historique | 16 nov. 2026 | 241 (~4 TPS) |
| `gpu-worldgen`, natif, `exact` | 23 nov. 2026 | 42,6 / 42,9 |
| Build du fork, PGO frais, `exact` | 24 nov. 2026 | 37,8 / 38,1 |
| `parallel-mobs` (`508f77de87`), natif, îlots et couloirs d'apparition activés | 24 nov. 2026 | 33,5 / 33,3 |
| `parallel-mobs`, PGO frais, îlots et couloirs d'apparition activés | 24 nov. 2026 | **30,1 / 30,2** |

Les mesures des 23 et 24 novembre utilisent 100 bots avec le comportement `mixed`, des comparaisons A–B–B–A et un CPU à 4,9 GHz pendant les passages chronométrés. Le gain depuis la référence initiale comprend à la fois des changements du fork et des fusions de l'amont.

## Comment l'exactitude est vérifiée

- **Génération du monde.** `tools/test_chunk_generation_parity.py` compare des chunks échantillonnés à un oracle Java qui utilise le JAR du serveur vanilla. Le projet a relevé zéro différence sur les chemins CPU et GPU pour un échantillon de 785 chunks couvrant 27 biomes et 74 sites de structures, avec la graine 12345 et un site de régression dans le Nether. `tools/gpu_worldgen.py probe` compare les sorties CPU et GPU sur des zones échantillonnées dans les trois dimensions, et la densité fait aussi l'objet d'un autotest au démarrage en plus des modes de comparaison par étape. Ces contrôles vérifient les chunks et zones échantillonnés, et non une équivalence universelle avec vanilla ni la reproductibilité des mondes sauvegardés d'une exécution à l'autre. Voir le [guide des bancs et des vérifications](../crates/horizon-gpu/BENCHMARKS.md).
- **Tick du serveur.** Le moteur du serveur dédié intègre un harnais d'empreintes (`tools/tick_digest.py`, piloté par `HORIZON_TICK_DIGEST`). Les passages de validation couvrent 600 ticks et 100 joueurs simulés sur deux scènes — vol en créatif de jour dans l'Overworld, et survie en difficulté élevée la nuit avec combats, générateurs, dangers et passages dans le Nether et l'End — avec des vérifications complètes des blocs et de la lumière des sections tous les 50 ticks. L'empreinte hache une sélection d'état des niveaux qui tickent et des entités suivies : état des générateurs aléatoires, écritures de blocs enregistrées, ticks planifiés de blocs et de fluides, chunks chargés, NBT normalisé des block entities, données de sauvegarde des joueurs suivis et paquets normalisés de la scène. **Il s'agit d'un contrôle de régression entre builds d'Horizon dans le même mode (série contre série, `exact` contre `exact`), pas d'une comparaison avec vanilla** ; la parité vanilla est vérifiée sur la génération du monde avec l'oracle Java ci-dessus.
- **Îlots de mobs.** Lorsque les îlots sont compilés et que des groupes admissibles sont disponibles, `HORIZON_PARALLEL_TICK=verify` compare plusieurs groupes à une référence à un seul groupe avec graines rejouées. Le projet a relevé zéro différence sur 200 ticks groupés vérifiés le 23 novembre 2026. Cela vérifie le regroupement dans ce mode, distinct du chemin série normal et de vanilla.

## Ce qui a été optimisé

- Les étapes GPU de la génération passent par le CPU lorsque le moteur CUDA est indisponible ou lorsque le travail sort du domaine pris en charge. En dehors de la génération des chunks, le GPU peut aussi traiter des recherches de biomes et des rayons d'exposition aux explosions.
- La génération et la sauvegarde sur CPU bénéficient de chemins accélérés pour les décorations, de l'écriture NBT directe des chunks et du stockage en ligne des petites listes d'identifiants des palettes.
- Le tick utilise des chemins rapides pour les mobs immobiles sur des blocs pleins, les contacts avec les blocs et certains tests de proximité des block entities. Il utilise aussi des masques compacts pour les ticks aléatoires, réutilise des calculs au sein des passes d'apparition naturelle et partage les corps de collision des joueurs entre les tours des mobs.
- **Les ticks aléatoires** peuvent préparer des masques de section compacts sur des threads auxiliaires lorsqu'il y a suffisamment de travail. Le thread du serveur ticke ensuite les chunks actifs en ordre série, en utilisant le générateur aléatoire du niveau et le flux de positions du chemin série.
- **L'apparition naturelle en couloirs** utilise un flux aléatoire par chunk. Les couloirs proposent des apparitions par rapport aux plafonds au début de la passe, et le serveur valide les résultats dans l'ordre de la passe. L'admissibilité dépend du mode, des couloirs disponibles, du nombre de chunks, des hooks d'apparition et du rejeu des graines.
- **Les îlots de mobs** regroupent les mobs admissibles pour des threads auxiliaires. Les tours des mobs peuvent utiliser des flux aléatoires par mob. Le code se replie sur le travail en série lorsque le plan de regroupement ou les conditions du tick l'exigent, notamment s'il y a trop peu de couloirs disponibles ou en cas de rappels vers le jeu.
- Le build peut utiliser l'optimisation guidée par profil (PGO), et sous Linux le thread du tick dédié peut être épinglé.

## Branches

| Branche | Contenu |
|---|---|
| `gpu-worldgen` (par défaut) | La génération sur GPU et les optimisations du tick en mode `exact` |
| `parallel-mobs` | Le code commun, avec les îlots de mobs et l'apparition naturelle en couloirs sur des threads auxiliaires en option |
| `main` | Horizon de l'amont, suivi par avance rapide |

## Utilisation

Suivre le [README de l'amont](../README.md) pour l'installation. Le build du jeu active `cuda` par défaut ; pour un build du moteur dédié avec `--no-default-features`, ajouter `--features cuda` pour la prise en charge du GPU, et le pilote CUDA ainsi que NVRTC sont chargés à l'exécution. Pour compiler les chemins `parallel-mobs`, compiler avec `horizon-client/upstream-parallel-mob-islands` (qui implique `upstream-parallel-mobs`) pour les îlots de mobs et les couloirs d'apparition, ou `horizon-client/upstream-parallel-mobs` pour les couloirs d'apparition seuls.

| Variable | Valeurs | Prérequis / Priorité | Effet |
|---|---|---|---|
| `HORIZON_GPU_DENSITY` | `1`, `wait` | Moteur d'exécution CUDA | Active la densité finale prise en charge sur GPU (avec aquifères et beardifier) ; `1` compile en arrière-plan avec repli CPU entre-temps, `wait` attend l'initialisation |
| `HORIZON_GPU_SURFACE` | `1` | Remplissage GPU | Active les règles de surface prises en charge sur GPU |
| `HORIZON_GPU_SECTIONS` | `1` | Chemin de surface GPU (`HORIZON_GPU_SURFACE`) | Active les sections compactes sur GPU |
| `HORIZON_GPU_CAVES` | `1` | `HORIZON_GPU_DENSITY` pour la matière des cellules creusées | Active le creusement pris en charge sur GPU (les masques peuvent tourner sans densité GPU) |
| `HORIZON_GPU_ORES` | `1` | Moteur d'exécution CUDA | Active les travaux pris en charge sur les amas et filons souterrains de minerais sur GPU |
| `HORIZON_GPU_GEODES` | `1`, `test` | Moteur d'exécution CUDA | Active le classement sur GPU des cellules des géodes d'améthyste (décoration `minecraft:amethyst_geode` seulement ; les autres géodes restent sur le CPU) ; `test` compare chaque cellule au CPU |
| `HORIZON_GPU_LIGHT` | `1` | Moteur d'exécution CUDA | Active le calcul de lumière sur GPU pour les chunks nouvellement générés |
| `HORIZON_GPU_BIOMES` | `1` | Moteur d'exécution CUDA | Active les biomes multi-bruit sur GPU pour les sources admissibles |
| `HORIZON_GPU_LOCATE` | `1` | Moteur d'exécution CUDA | Active les recherches de biomes prises en charge sur GPU (dont `/locate biome`) |
| `HORIZON_GPU_TNT` | `1` | Moteur d'exécution CUDA | Laisse un modèle de coût choisir CUDA pour les rayons d'exposition aux explosions |
| `HORIZON_PARALLEL_TICK` | `1`, `exact`, `0`, `verify` | fonctionnalité Cargo `parallel-mobs` pour `1`/`verify` ; laisser `[engine] parallel_tick` à false pour passer `exact`/`0` sous le superviseur | `1` active les îlots et couloirs d'apparition compilés lorsqu'ils sont admissibles (hors `exact`) ; `exact` conserve les flux aléatoires en série avec les îlots et couloirs inactifs ; `0` sélectionne le mode série ; `verify` vérifie les ticks à plusieurs groupes admissibles face à une référence à un seul groupe avec graines rejouées |
| `HORIZON_TICK_CPU` | `<cpu>` | Serveur dédié Linux ; `HORIZON_SERVER_CORE=1` est prioritaire | Épingle le thread du tick dédié sur un cœur CPU |

Comme alternative à `HORIZON_PARALLEL_TICK=1`, définir `parallel_tick = true` sous `[engine]` dans le fichier `server.toml` du superviseur dédié (qui démarre le moteur avec `HORIZON_PARALLEL_TICK=1`), ou choisir Parallel Tick **On** ou un réglage **Auto** actif en solo. Les deux chemins d'activation en option nécessitent un tick parallèle actif hors `exact` et appliquent toujours leurs contrôles d'admissibilité.

## Notes

- Horizon et ce fork sont sous licence LGPL-3.0-or-later. Horizon prépare les ressources Minecraft à partir de votre propre exemplaire du jeu.
- Ce dépôt est un fork communautaire non officiel.
- Les notes du projet attribuent des contributions à des travaux menés avec Claude, Codex et Gemini sous la direction du propriétaire du dépôt.
