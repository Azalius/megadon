# Organisation du repo
## Fichiers Matlab
Les fichiers Matlab liés à la classification sont de la forme '[dl|vec][Titre|Corps|Combo].m'. Le fichier dlTitre.m par exemple traite du classificateur Deep Learning(dl) sur le titre(Titre). Combo correspond à la concaténation du titre et du corps. L'architecture de tous ces fichiers est similaire.
Le fichier 'test' permet de tester un modèle entrainer sur des données arbitraires.
La fonction preprocessText est stockée dans sous propre fichier afin d'etre réutilisable.
## Autres fichiers
Les csv sont extraits de Kaggle et ont permis de générer les .xlsx sont utilisés pour notre étude.
