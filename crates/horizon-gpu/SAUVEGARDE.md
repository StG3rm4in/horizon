# Sauvegarde des chunks

Le débit de `/pregen` est limité par les 6 cœurs physiques (DEBIT.md). Dans le profil du 15/11 avec
tout sur GPU (biomes compris), la sauvegarde prend **17,9 % du CPU** (encodage NBT, écriture des
régions) et la compression zlib **9,9 %**. Les deux tournent dans les 6 threads du pool de
compression (`save/compression_pool.rs`) : `ChunkRecord::from_nbt` encode le NBT, puis
`Compression::encode` le compresse.

## La compression

Horizon écrit les régions comme vanilla :
- `region-file-compression` de server.properties choisit `deflate` (par défaut), `lz4`, `gzip` ou
  `none` (`Compression::configure`, comme `RegionFileVersion.configure`) ;
- deflate au niveau par défaut de `Deflater`, 6.

`HORIZON_ZLIB_LEVEL` (0 à 9, lu une fois, `compression.rs`) change ce niveau pour les bancs. N'importe
quel lecteur relit un flux deflate, quel qu'en soit le niveau.

Banc `/tmp/ab21.sh "copie variante x z" ...` (release, tout sur GPU, rayon 1 600 blocs, 40 401
chunks par zone, zones neuves). Chaque variante passe une fois dans chaque copie (carré latin). La
taille est celle des fichiers `region/`, `entities/` et `poi/` de la zone (`/tmp/taille_zone.py`) :

| Zone | deflate 6 (défaut) | deflate 1 | lz4 |
|---|---|---|---|
| 3500000/−3520000 | 1 645 chunks/s, 5,46 ms CPU, 386 Mio | +4,3 %, −6,0 % CPU, +27,7 % disque | +7,2 %, −9,3 % CPU, +46,6 % disque |
| −3530000/3540000 | 1 672 chunks/s, 5,16 ms, 370 Mio | +11,0 %, −8,3 %, +26,8 % | +11,8 %, −9,5 %, +44,8 % |
| 3550000/−3570000 | 1 748 chunks/s, 4,94 ms, 358 Mio | +8,2 %, −6,9 %, +25,3 % | +10,0 %, −9,9 %, +43,8 % |

En moyenne :

| Variante | Débit | CPU par chunk | Disque par chunk |
|---|---|---|---|
| deflate 6 | — | — | 9,4 Kio |
| deflate 1 | +7,8 % | −7,1 % | 11,9 Kio (+27 %) |
| lz4 | +9,7 % | −9,6 % | 13,7 Kio (+45 %) |

lz4 supprime presque tout le coût de zlib. Ses régions ne se relisent que par vanilla 1.20.5 et plus
récent, et par les outils qui connaissent ce format (type 4, le bloc de lz4-java). Deflate 1 garde le
format de toujours.

Pour une prégénération de 10 000 blocs de rayon (1,56 million de chunks) : 14,0 Gio en deflate 6,
17,7 Gio en deflate 1, 20,4 Gio en lz4.

### L'option du niveau deflate

`[world] region_deflate_level` dans server.toml (0 à 9, 6 par défaut comme vanilla) :

1. Le superviseur l'écrit dans server.properties sous `horizon-region-deflate-level`
   (`config::TYPED`, bornes dans `settings.rs`).
2. Le moteur la passe à `Compression::configure_deflate_level` au démarrage (`dedicated.rs`, juste
   après `region-file-compression`). Une valeur invalide est signalée dans le journal, et 6 reste.
3. `HORIZON_ZLIB_LEVEL` sert encore aux bancs quand rien n'est configuré.

Le niveau ne compte que pour `deflate` et `gzip`.

Vérifié par :
- un test unitaire de compression (relecture aux niveaux 1 et 9, refus de 10) ;
- les tests de horizon-server : référence `docs/api/server-reference.md` régénérée (`HORIZON_WRITE_DOCS=1
  cargo test -p horizon-server reference`), golden de l'API mis à jour (`testdata/api-v1-config.json`),
  `defaults_are_vanilla` ;
- un serveur réel : même zone (rayon 400), 26,1 Mio au niveau 6, 32,7 Mio au niveau 1 (+25 %).

Depuis le 15/11, les copies de banc (`/data/horizon-ab/{gpu,gpub,cpu}`) écrivent en **lz4**
(`region-file-compression = "lz4"` dans le `[properties]` de leur server.toml). Les mesures qui suivent
ne se comparent donc plus directement aux précédentes, faites en deflate 6. `/tmp/ab21.sh`, qui
changeait server.properties directement, n'a plus d'effet : le serveur le réécrit depuis server.toml.

## L'encodage

Profil (`/tmp/profil_b1.sh`, zone 3600000/−3620000, deflate 6, `/tmp/sous_filtre.py`) : la sauvegarde
pèse 27 % du CPU en tout, compression comprise.

- `encode_saved` (le chunk en arbre NBT) : 12,9 %, dont
  - `PalettedContainer::pack` 6,1 % (`palette_ids` 2,7 %, `pack_words` 1,9 %, `unpack` 1,1 %) ;
  - les chaînes 2,7 % (`JavaString` est un `Vec<u16>` : chaque chaîne est transcodée en UTF-16 et
    allouée) ;
  - les ticks planifiés 2,3 % (un `BTreeMap` et sept chaînes par tick) ;
  - les compounds 2,3 %.
- `nbt::write_named` (l'arbre en octets) : 2,2 %.
- La compression : 9,7 %.

### `pack` sans recompactage

`PalettedContainer::pack` (paletted.rs, `PalettedContainer.pack` de vanilla) décompacte les 4 096
cellules, refait la palette dans l'ordre de première apparition, puis recompacte les ids. Seules les
sections uniformes avaient un raccourci.

Or un conteneur que `from_values` ou `from_mapped` vient de construire est déjà sous cette forme :
- sa palette liste chaque valeur dans l'ordre de première apparition, sans case morte ;
- `build` a encodé les ids à la largeur que `pack` choisit, avec le même `pack_words`.

C'est le cas des sections de chunk que `block_column` construit (`BlockSection::to_container`).
L'indicateur `packed` du conteneur le note. `set`, seule écriture, l'efface. `pack` rend alors la
palette et les mots tels quels. Le calcul complet reste dans `pack_full`.

Vérifications :

- test unitaire `packed_containers_pack_as_in_full` : blocs et biomes, de 2 à 300 valeurs, par
  `from_values`, `from_mapped` et après une écriture ;
- `HORIZON_PACK_TEST=1` calcule chaque `pack` des deux façons et compare. `/pregen` en mode test
  (rayon 800, zone 3650000/−3670000) : **plus de 131 072 conteneurs, 0 écart** ;
- les régions des bancs, comparées entre l'ancien et le nouveau binaire (`/tmp/cmp_blocs.py` et
  `/tmp/nbt_diff.py`, qui liste les chemins des tags qui diffèrent) : 15 chunks sur 40 401 aux blocs
  différents. C'est le bruit connu d'un passage à l'autre : fluides, âge des plantes, baies, lianes,
  des ticks qui tournent pendant `/pregen`. Le NBT entier diffère pour ~5 % des chunks, mais par des
  champs qui changent déjà d'un passage à l'autre :
  - l'ordre de la liste `block_entities` ;
  - les graines de butin ;
  - l'ordre des `fluid_ticks`.

  Aucune section aux blocs identiques n'est écrite autrement.

Banc `/tmp/ab22.sh` (release, tout sur GPU, deflate 6 ; binaires d'avant dans `target/avant-pack/`,
copie gpub ; ceux d'après dans la copie gpu ; zones neuves) :

| Zone | Débit avant → après | CPU par chunk |
|---|---|---|
| 3700000/−3720000 | 1 704 → 1 795 chunks/s (+5,3 %) | 5,14 → 4,85 ms (−5,6 %) |
| −3730000/3740000 | 1 405 → 1 516 (+7,9 %) | 6,41 → 6,03 ms (−5,9 %) |
| 3750000/−3770000 | 1 676 → 1 745 (+4,1 %) | 5,23 → 4,93 ms (−5,7 %) |

**−5,7 % de CPU par chunk, +5,8 % de débit.**

### L'écriture directe du NBT

Profil en lz4 (zone 3850000/−3870000) : construire l'arbre (`encode_saved`) coûtait 9,1 % du CPU, et
l'écrire (`write_named`) 2,7 %. `JavaString` est un `Vec<u16>` : chaque chaîne est transcodée en
UTF-16 et allouée, puis retranscodée à l'écriture. Chaque tick planifié devenait un `BTreeMap` et sept
chaînes.

Désormais, `encode_saved_bytes` (save.rs) écrit les octets que `write_named(encode_saved(...))`
écrirait, sans l'arbre des grosses parties :

- `encode_root` est `encode_with`, mais avec les sections et les ticks laissés en listes vides. Le
  reste de la racine reste en tags : entités de bloc, structures, cartes de hauteur, clés d'un chunk
  chargé.
- `nbt::RootWriter` (horizon-world nbt.rs) écrit la racine entrée par entrée avec le même `Writer`.
  On garde donc l'UTF-8 modifié, la chaîne trop longue écrite vide, le NaN canonique et la limite
  d'octets. Il n'existe que sous des limites sans plafond de nœuds (`Limits::saved`), puisque ce
  qu'il écrit directement n'est pas compté.
- Les entrées de la racine sont parcourues dans l'ordre du `BTreeMap`. `sections`, `block_ticks` et
  `fluid_ticks` sont écrits directement (`write_sections`, `write_ticks`), dans l'ordre où leurs
  compounds trieraient leurs clés :
  - section : `BlockLight`, `SkyLight`, `Y`, `biomes`, `block_states` ;
  - `biomes` et `block_states` : `data`, `palette` ;
  - tick : `i`, `p`, `t`, `x`, `y`, `z`.
- Les palettes d'états (`write_state_palette`) écrivent le nom d'un état par défaut, ou le compound
  d'un autre. Ce compound est sérialisé une fois par l'écrivain actuel (`nbt::payload_bytes`) et mis
  en cache par thread. Une palette qui mélange chaînes et compounds devient, comme dans
  `Writer::payload`, une liste de compounds où les chaînes sont enveloppées dans `{"": …}`.
- Le pool de compression appelle `Record::encoded`, puis `ChunkRecord::from_bytes`. Les autres
  enregistrements (protos, POI, tags) passent par l'arbre comme avant.
- `HORIZON_NBT=tree` reprend l'ancien chemin. `HORIZON_NBT_TEST=1` écrit les deux, compare les
  octets, garde ceux de l'arbre en cas d'écart, et écrit des lignes `NBT test:` dans latest.log.

Vérifications :

- test unitaire `direct_chunk_bytes_match_the_tree` (horizon-client-lib, avec
  `HORIZON_TEST_RESOURCES`) : états tirés dans tout le registre, donc des palettes mixtes, avec et sans
  lumière, avec et sans tables ; les 240 tests de sauvegarde passent ;
- `/pregen` en mode test (rayon 800, zone 4000000/−4020000) : **plus de 8 192 chunks, 0 écart octet
  par octet**.

Banc `/tmp/ab23.sh` (release, tout sur GPU, lz4 ; même binaire, `HORIZON_NBT=tree` dans la copie
gpub contre l'écriture directe dans la copie gpu ; zones neuves) :

| Zone | Débit arbre → direct | CPU par chunk |
|---|---|---|
| 4050000/−4070000 | 1 852 → 2 004 chunks/s (+8,2 %) | 4,70 → 4,20 ms (−10,6 %) |
| −4080000/4090000 | 2 029 → 2 093 (+3,1 %) | 4,07 → 3,82 ms (−6,1 %) |
| 4100000/−4120000 | 1 844 → 1 956 (+6,1 %) | 4,77 → 4,38 ms (−8,2 %) |

**−8,3 % de CPU par chunk, +5,8 % de débit** en moyenne.

### Ce qui reste

- Les cartes de hauteur sont recalculées depuis les blocs à chaque sauvegarde. Depuis le 15/11 au
  soir, c'est fait section par section (`Heightmap::prime_sections`, voir GPU_ATTENTE.md) : 1,51 % →
  0,93 % du CPU.
- Les entités de bloc et le reste de la racine passent encore par l'arbre ; c'est peu par chunk.
- Les protos (bord de `/pregen`) s'écrivent encore par l'arbre.
