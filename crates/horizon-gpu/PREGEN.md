# Avancée de `/pregen`

## Le constat

Avec tout sur GPU (densité, aquifères, beardifier, surface, minerais, grottes, lumière), `/pregen`
n'occupe plus que 8 à 9 cœurs sur 12. La trace `HORIZON_PREGEN_TRACE=1` (moyenne de chaque étape sur
une seconde dans latest.log, `game/commands/pregen.rs`) montre, sur une zone de 201 × 201 chunks :

| Mesure | Valeur |
|---|---|
| File des workers vide | 28 à 57 % des mises à jour |
| Décorations en cours | 5,5 en moyenne (limite : 2 par worker, 24) |
| Générations en cours | 15 à 20 (limite : 3 par worker, 36) |
| Lumières en cours | 4 à 8 |

## Pourquoi

`/pregen` balaie la zone par bandes de colonnes (jusqu'à 256 de large) sur quelques rangées (au moins
1 024 chunks : 6 rangées ici). `Streaming::set_band` donne aux chunks de la bande leur rang de
décoration rangée par rangée, colonne après colonne, pour que l'ordre ne dépende pas du minutage.

Or un chunk n'est décoré qu'une fois décorés tous les chunks de rang inférieur à moins de deux de lui
(`DecorationOrder::blocked` : leurs régions d'écriture se recouvrent). Colonne après colonne, chaque
chunk attend celui de sa gauche : une rangée se décore en série, et seules les 5 ou 6 rangées de la
bande avancent ensemble.

## Le changement

Par défaut depuis le 15/11 (`HORIZON_PREGEN_ORDER=rows` pour l ancien ordre) : chaque rangée en trois passes, une colonne sur trois à chaque passe.
Deux chunks à trois colonnes l'un de l'autre n'ont pas de région d'écriture commune, ni de chunk de
leurs 3 × 3 en commun : une passe décore jusqu'à 67 chunks à la fois sur 201 colonnes. L'ordre reste
fixé par la géométrie de la bande, indépendant du minutage.

Le monde ne change pas de qualité, mais pas non plus à l'identique : là où les décorations de chunks
voisins se recouvrent (arbres, filons, sculk aux bords), l'ordre décide de ce qui passe en premier.
Aucun des deux ordres n'est celui de vanilla, qui n'a pas de prégénération (l'ordre de vanilla autour
d'un joueur, `DecorationOrder::reassign`, n'est pas touché).

## Mesures

Trace (serveur compilé dans parity-target, même zone neuve 1550000/−1570000) :

| Ordre | Débit | File vide | Décorations | Générations |
|---|---|---|---|---|
| colonne après colonne | 1 260 chunks/s | 28 à 39 % | 5,5 | 18 |
| trois passes | 1 380 chunks/s (+9,5 %) | 4 à 8 % | 6 à 12 | 29 à 30 |

Bancs release (`/tmp/ab14.sh`, rayon 1 600 blocs, zones neuves, tout sur GPU) :

| Zone | Colonne après colonne | Trois passes | Cœurs occupés |
|---|---|---|---|
| 1600000/−1620000 | 1 305 chunks/s | 1 386 (+6,2 %) | 8,6 → 9,4 |
| −1630000/1640000 | 1 386 chunks/s | 1 635 (+18,0 %) | 7,6 → 9,4 |
| 1650000/−1670000 | 1 303 chunks/s | 1 344 (+3,2 %) | 8,7 → 9,3 |

Le temps CPU par chunk monte de 4 à 6 % : plus de cœurs travaillent en même temps sur une machine à une
seule barrette de mémoire. Plus de workers (`HORIZON_WORKER_THREADS=14` ou `16`, `/tmp/ab15.sh`)
n'apporte rien de net (−2,6 à +2,7 %).

## Reproductibilité

`/tmp/cmp_regions.py` et `/tmp/cmp_blocs.py` comparent les chunks que deux mondes ont sauvegardés.
Même zone de 51 × 51 chunks, générée deux fois (12 puis 5 workers) :

| Ordre | Chunks aux blocs différents | Aux biomes différents |
|---|---|---|
| colonne après colonne | 3 sur 2 601 | 0 |
| trois passes | 1 sur 2 601 | 0 |

Les quelques écarts, et ceux des entités de bloc (graines de butin, compteurs) et des ticks planifiés,
existaient déjà : `/pregen` fait tourner les ticks du serveur pendant la génération, et les fluides
planifiés s'écoulent plus ou moins selon le temps que les chunks restent chargés.

## Correction du 15/11

Le débit était déjà limité par le CPU : « 8 à 9 cœurs sur 12 » compte des threads logiques, et le
10600K n'a que 6 cœurs physiques. Voir DEBIT.md.
