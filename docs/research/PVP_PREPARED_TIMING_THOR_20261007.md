# Chronométrage résident préparé sur Thor — 7 octobre 2026

La campagne diagnostique complète donne des rapports médians appariés vec4/packed7 entre 1,329 et 1,680 pour les six géométries. Ce sont des résultats bruts de temps hôte encodage/soumission/attente, sans admission de performance indépendante. Aucun gain de modèle ni de bout en bout n'est établi.

Source exécutée : `e04c980817cac889e759fb2bb33cf231ef737e73`. Rust 1.89.0 release, Vulkan, NVIDIA Tegra NVIDIA Thor, pilote 580.00. Binaire copié immuable SHA256 `953dab3555d3ea2326229e0c265c55dddaee05eae09e9e8195e66057fb3baa11`. Le protocole prospectif est [PVP_PREPARED_TIMING_PROTOCOL.md](PVP_PREPARED_TIMING_PROTOCOL.md).

Trois processus frais, six géométries, banques ANF boundary/sparse4/dense32, cinq variantes préparées, cinq échauffements et vingt répétitions par cas. Ordre cyclique et miroir équilibré. Total : **10 800 observations chronométrées**, plus 2 700 échauffements. Les 60 750 comparaisons complètes source/aller/retour, doubles lectures et oracle, sont exactes : zéro mot erroné, zéro géométrie rejetée, trois sorties 0. Chaque résultat est vérifié hors du chronomètre, avec padding et quatre mots de garde.

Les banques ANF sont converties vers les dispositions physiques packed ou vec4. Le passage au plan préparé ne crée aucun buffer/bind group pendant l'encodage. Les cinq variantes utilisent cette même règle.

## Aller : temps et rapports bruts

Chaque ligne regroupe trois processus × trois banques × vingt répétitions (180 paires). Les temps sont les médianes des observations ; le rapport est la médiane des ratios appariés, donc ne correspond pas nécessairement au quotient des deux médianes. L'intervalle est le minimum/maximum des neuf médianes processus × banque, pas un intervalle de confiance.

| K | G | vec4 médiane (ms) | packed7 médiane (ms) | Rapport apparié | Médianes par cas |
|---:|---:|---:|---:|---:|---:|
| 4096 | 128 | 0.323 | 0.195 | 1.668 | 1.625–1.708 |
| 16384 | 129 | 0.632 | 0.391 | 1.632 | 1.617–1.661 |
| 16384 | 2048 | 1.154 | 0.844 | 1.374 | 1.265–1.499 |
| 65536 | 129 | 0.989 | 0.588 | 1.680 | 1.590–1.699 |
| 65536 | 512 | 1.251 | 0.944 | 1.349 | 1.060–1.379 |
| 262144 | 128 | 1.364 | 1.028 | 1.329 | 1.279–1.337 |

Toutes les variantes sont conservées. Un rapport inférieur à 1 signifie un ralentissement.

| K | G | fused2 | tile8 | packed1 | packed7 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 128 | 1.051 | 1.121 | 1.003 | 1.668 |
| 16384 | 129 | 1.048 | 1.063 | 1.053 | 1.632 |
| 16384 | 2048 | 1.033 | 0.875 | 0.868 | 1.374 |
| 65536 | 129 | 0.993 | 0.936 | 1.126 | 1.680 |
| 65536 | 512 | 0.995 | 0.870 | 0.878 | 1.349 |
| 262144 | 128 | 0.949 | 0.859 | 0.887 | 1.329 |

Les retours sont mesurés séparément, disponibles dans les CSV. Les percentiles 95, extrema, comptes et médianes par processus/banque/phase/variante figurent dans summary.csv. Aucune moyenne globale des différentes tailles ne sert de preuve.

## Limites et restauration

Le temps comprend création de l'encodeur, encodage, soumission et attente explicite de fin. Upload/reset, conversions, préparation et lectures de vérification sont exclus et disposent de lignes séparées. Les timestamps GPU ne sont pas demandés. La latence de bout en bout n'est pas mesurée.

L'observateur fournit 37 snapshots observed et 44 partial, avec 40 observations incomplètes pendant les mesures. L'occupation reste inconnue pour l'admission ; **zéro observation de confirmation de performance est acceptée**. Le verrou coopératif et la pause des services ne prouvent pas l'inactivité GPU. Les workflows CI physiques étaient également en cours ; leur état seul ne démontre aucune contention effective.

La campagne a tourné de 09:28:43 à 09:30:40 UTC. Les quatre services memorithm-clm, memorithm-clm-encoder, memorithm-viggle-lan et rustdesk sont rétablis actifs avec leurs politiques initiales. Inspection indépendante réussie à 09:30:55 UTC : gardes retirées, trace retirée, watchdog inactif, verrou disponible, contrôleur sorti 0. Le binaire historique PVP3d est inchangé (SHA256 `7aef67026ad088504478885cf9f2b7c864218f6fb3b0792bd8609fd907d3f414`).

Avant toute exécution GPU, un premier contrôle Clippy a rejeté option_env.expect ; il a été corrigé avant de figer la révision exécutée. Une préparation anticipée a échoué sur un binaire non encore copié, sans modifier les services ni lancer de test GPU. Le dossier partiel campaign.QVSQ0xC6 reste conservé dans l'archive ; aucun processus de mesure échoué n'a été remplacé.

## Données reproductibles

Dossier [pvp-timing-thor-20261007](evidence/pvp-timing-thor-20261007/) : analyse, résumé complet, ratios et inspection. L'archive conserve tous les logs natifs, toutes les observations, traces, sources du contrôleur et manifeste SHA256.

Archive : `pvp-timing-thor-20261007.tar.xz`
SHA256 : `b8733b8cf3f49e6863acad759a64d15f8c1d5faa9963c44b7db44548213a7a93`
Git blob : `9b2428a60e749393054ddd455db861997b945fd5`
Le Git blob publié est identique au git hash-object calculé sur Thor.
