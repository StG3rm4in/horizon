# Horizon GPU

La génération du monde sur le GPU et un tick serveur accéléré, pour Horizon.

[English](README.md)

Ce dépôt est un fork d'[Horizon](https://github.com/horizon-rust/horizon), une réécriture complète en Rust de
Minecraft: Java Edition 26.3 (client, serveur et lanceur), vérifiée contre le jeu vanilla. Le fork ajoute deux
choses, avec la même règle que l'amont : **rien ne doit différer du vanilla**.

1. **La génération du monde sur le GPU.** La densité, les aquifères, le beardifier, les règles de surface, les
   minerais, les grottes, la lumière, les sections et les biomes tournent sur un GPU NVIDIA (CUDA) dans
   l'Overworld. Le Nether et l'End n'ont pas d'aquifères ; le GPU y fait la densité et les grottes, et dans le
   Nether aussi les règles de surface et les biomes ; les biomes de l'End et les sections compactes du Nether
   restent sur le CPU (`crates/horizon-gpu/NETHER_END.md`). Chaque chunk sort identique, bloc par bloc, au chemin
   CPU, lui-même identique au vanilla.
2. **Un tick du serveur dédié plus rapide à 100 joueurs.** Chaque optimisation garde le tick identique au bit
   près au code d'avant, ce que vérifie un harnais tick par tick sur une scène fixe (voir plus bas).

## Résultats

Mesurés sur une seule machine : un Intel i5-10600K (6 cœurs, 12 threads) à 4,9 GHz, 32 Go de mémoire en double
canal, une NVIDIA RTX 3080.

### Génération du monde (`/pregen`, zones neuves)

| Dimension | CPU seul | CPU + GPU |
|---|---|---|
| Overworld | 853 chunks/s, 12,9 ms de CPU par chunk | **2 287 chunks/s (×2,7)**, 3,4 ms de CPU par chunk |
| Nether | — | ×2,4 |
| End | — | ×1,9 |

Ce qui limite maintenant, ce sont les 6 cœurs du CPU : le GPU travaille environ deux tiers du temps.

### Tick du serveur, 100 joueurs (millisecondes par tick en moyenne ; 20 ticks par seconde demandent moins de 50 ms)

| Build | ms par tick |
|---|---|
| Premier banc, 16/11/2026 | 241 (environ 4 ticks par seconde) |
| Mode exact, build natif, 22/11/2026 | 55,0 |
| **Mode exact, build PGO** | **44,2** |
| **Mode exact, PGO, thread du tick épinglé sur un cœur** | **42,1** |
| Îlots de mobs (option, non exact), PGO | 34,7 |
| Îlots de mobs, PGO, épinglé | 33,5 |

Le banc, ce sont 100 bots qui volent dans un même monde, chacun chargeant sa propre zone ; chaque chiffre est la
moyenne de passages faits dans l'ordre A B B A. La première ligne inclut aussi les progrès de l'amont depuis,
que le fork intègre régulièrement.

## Comment l'exactitude est vérifiée

- **Génération du monde :** les chunks générés sont comparés à ceux du serveur vanilla à la graine 12345
  (785 chunks, 27 biomes, 74 sites de structures : 0 différence, sur le GPU comme sur le CPU), et une sonde compare
  les chemins CPU et GPU dans les trois dimensions. Une seule graine est comparée au vanilla ; les étapes GPU ont
  aussi des modes de test qui comparent chaque résultat à celui du CPU.
- **Tick du serveur :** un harnais fait tourner deux scènes fixes, 600 ticks avec 100 joueurs simulés : celle
  par défaut (des joueurs en créatif qui volent dans l'Overworld, de jour) et une scène de survie (la nuit, en
  difficile, une arène de générateurs et de dangers, des mobs qui combattent les joueurs, des joueurs dans le
  Nether et l'End). À chaque tick, il hache l'aléa du niveau et sa gaussienne, les écritures de blocs, les ticks
  planifiés, le mouvement de chaque entité et ses champs lus tels quels (santé, absorption, temps de blessure et
  de mort, feu, gel, air, effets, cible et joueur auquel elle est liée, orientations, coups…), l'aléa propre de
  chaque entité, les compteurs, le NBT des block entities qui tickent, les données de chaque joueur, chaque
  paquet envoyé à chaque joueur, chaque niveau qui tick et les chunks chargés (et tous les 50 ticks les blocs et
  la lumière de chaque section). Deux builds doivent donner les mêmes lignes, en série et avec le tick parallèle
  (`HORIZON_PARALLEL_TICK=exact`).

  Cela montre qu'une optimisation ne change rien par rapport au build d'avant, sur ces scènes ; ce n'est pas une
  comparaison avec le vanilla. Le harnais est dans chaque build du serveur, mais ne tourne qu'avec
  `HORIZON_TICK_DIGEST` (que `tools/tick_digest.py` définit).
- **Îlots de mobs :** `HORIZON_PARALLEL_TICK=verify` fait tourner les mobs à la fois en groupes et en série, et
  compare les deux.

## Ce qui a été optimisé

- Les étapes GPU ci-dessus, avec un repli sur le CPU pour chaque étape et pour tout ce qu'une étape GPU ne
  couvre pas.
- La génération et la sauvegarde sur le CPU : les décorations, un écrivain NBT direct pour les chunks
  sauvegardés, des palettes de sections stockées en ligne.
- Le tick : des raccourcis à la manière de Lithium (un mob immobile sur un bloc plein, les contacts avec les
  blocs par une table de bits, les block entities sans joueur à portée), des masques compacts pour les ticks
  aléatoires, une apparition naturelle qui garde le travail par chunk et par passe, les corps des joueurs
  partagés entre les tours, moins de recherches et d'allocations dans les chemins chauds.
- Le build : l'optimisation guidée par un profil (PGO, environ −20 % en mode exact) et l'épinglage optionnel du
  thread du tick sur un CPU (`HORIZON_TICK_CPU`).
- Les îlots de mobs (branche `parallel-mobs`) : les mobs qui ne peuvent pas s'influencer sont tickés en groupes
  sur des threads d'aide, et le résultat est vérifié égal à celui de la boucle en série.

## Branches

| Branche | Contenu |
|---|---|
| `gpu-worldgen` (par défaut) | la génération sur GPU et les optimisations exactes du tick |
| `parallel-mobs` | `gpu-worldgen` plus les îlots de mobs |
| `main` | Horizon de l'amont, sans changement |

## Utilisation

Compiler et lancer comme l'amont (voir le [README de l'amont](../README.md)). Les options du fork :

- `HORIZON_GPU_<ÉTAPE>=1` pour `DENSITY`, `SURFACE`, `ORES`, `CAVES`, `LIGHT`, `SECTIONS`, `BIOMES` : fait
  tourner cette étape sur le GPU (CUDA). Le jeu se replie sur le CPU quand aucun GPU n'est disponible.
- `HORIZON_PARALLEL_TICK=exact` : le tick parallèle, gardé exact.
- `HORIZON_TICK_CPU=<cpu>` (Linux) : épingle le thread du tick du serveur dédié sur un CPU.
- Îlots de mobs : un build de `parallel-mobs` avec la fonctionnalité `upstream-parallel-mob-islands`, et
  `parallel_tick = true` dans `server.toml`.

## Notes

- Licence : LGPL-3.0-or-later, comme l'amont. Aucun fichier du jeu Minecraft n'est inclus ; Horizon les prépare
  à partir de votre propre exemplaire du jeu.
- Ce n'est pas un produit officiel Minecraft, ni approuvé par Mojang ou Microsoft, ni associé à eux.
- Le code du fork a été écrit avec Claude (Anthropic) sous la direction du propriétaire du dépôt, et chaque
  changement est vérifié par les harnais ci-dessus avant d'être intégré.
