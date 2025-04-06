# QRT-ENS-DataChallenge-
Prédiction de Survie Globale de patients atteints de Leucémie Myéloïde par QRT


Contexte
Au cours des dernières années, le secteur médical a de plus en plus adopté des méthodes basées sur l’analyse des données de santé en grande quantité, notamment dans le domaine du pronostic et du traitement de maladies complexes telles que le cancer. Les modèles prédictifs en santé ont transformé les soins aux patients, permettant des stratégies de traitement bien plus adaptées et efficaces. Ces avancées sont particulièrement précieuses en oncologie, où des modèles de prédictions précis peuvent significativement améliorer la qualité et le timing des décisions thérapeutiques.

But
En partenariat avec l’Institut Gustave Roussy, le Data Challenge de QRT de cette année se concentre sur la prédiction du risque de décès pour les patients diagnostiqués avec un cancer du sang, plus précisément un sous-type de leucémie myéloïde adulte. Pour ces patients, l’évaluation du risque est mesurée par la survie globale — la période allant du diagnostic initial jusqu’au décès du patient ou jusqu’au dernier suivi enregistré.

Pourquoi est-ce important ? Estimer le pronostic d’un patient est essentiel pour adapter son approche thérapeutique. Les patients identifiés comme ayant un profil à faible risque peuvent recevoir des thérapies de soutien visant à améliorer les paramètres sanguins et la qualité de vie globale, tandis que les patients identifiés comme à haut risque peuvent être prioritaires pour des options de traitement plus intensives, telles que la greffe de cellules souches hématopoïétiques.

Des prédictions précises des risques pourraient donc conduire à de meilleures décisions cliniques, une meilleure qualité de vie des patients, et à une utilisation plus efficace des ressources au sein des établissements de santé.

Ce challenge offre aux participants une occasion unique de travailler avec des données réelles provenant de 24 centres cliniques et de contribuer à une application concrète de la science des données au domaine médical.

Description des données
Le jeu de données est structuré en deux fichiers ZIP : X_train.zip et X_test.zip, ainsi qu’un fichier CSV : Y_train.csv. Le jeu d’entraînement contient des données sur 3 323 patients, et le jeu de test contient des données sur 1 193 patients. Les fichiers ZIP contenant les données d’entrée sont séparés en deux ensembles : Données Cliniques et Données Moléculaires.

La colonne ID est l’identifiant unique du patient qui relie les Données Cliniques, les Données Moléculaires et Y_train.

Objectif
L’objectif du Data Challenge est de prédire la survie globale (OS) des patients diagnostiqués avec un cancer du sang. Deux résultats clés sont fournis pour chaque patient dans Y_train :

OS_YEARS : Le temps de survie global en années depuis le diagnostic.
OS_STATUS : Indicateur de l’état de survie, où 1 indique un décès et 0 indique que le patient était vivant lors du dernier suivi.
La sortie attendue est un fichier CSV contenant les IDs des patients et une prédiction du risque de décès. Comme la métrique expliquée plus bas l’indique, l’échelle des prédictions n’importe pas, uniquement l’ordre. Pour deux patients i et j, une prédiction de risque plus faible pour i que pour j indique que l’on estime que le patient i survivra plus longtemps que le patient j.

Le fichier soumis devra avoir en index la colonne nommée “ID”, avec les ID des patients, et une colonne nommée “risk_score”, contenant la prédiction du risque de décès.
Un exemple de fichier de soumission contenant des prédictions aléatoires est fourni dans la section Fichiers.

Métrique : IPCW-C-index (Concordance Index for Right-Censored Data with IPCW)
Indice de Concordance (C-index)
Le C-index mesure à quel point un modèle prédictif peut ordonner correctement les temps de survie. Il calcule la proportion de toutes les paires comparables d’individus pour lesquelles les prédictions des risques de décès du modèle sont dans le bon ordre par rapport à leurs temps de survie réels.

Par exemple un patient j qui est décédé à un temps T_j ne peut pas former de paire “comparable” avec un patient i qui a survécu jusqu’à un temps T_i<T_j, car on ne sait pas si ce patient i a survécu jusqu’à T_j.

En revanche, on peut déterminer que ce patient j a survécu plus longtemps que tous les patients i décédés avant T_j (T_i<T_j). On peut aussi déterminer que le patient j est décédé avant tous les patients qui ont survécu ou qui sont décédés à un temps plus tard que T_j (T_i>T_j).

Pour une paire “comparable” de patients i et j avec T_i<T_j, une paire est dite concordante si les risques de décès prédits par le modèle sont dans le bon ordre R_i>R_j.
​​ 

Pour un modèle avec des prédictions de survie, le C-index varie entre 0 et 1 :

1 : Concordance parfaite (classement idéal).
0,5 : Modèle aléatoire (aucun pouvoir prédictif).
IPCW-C-index
L’IPCW-C-index étend le C-index traditionnel pour mieux gérer les données censurées à droite en appliquant des poids inverses basés sur une probabilité de censure (IPCW). La censure à droite est définie par le fait que pour certains patients, les données ne contiennent pas le temps de survie total mais uniquement un temps de survie partiel (c’est le cas des patients encore au vie au moment de la dernière observation). Cette approche modifie le calcul du C-index en attribuant des poids différents aux paires de données, en fonction de la probabilité que chaque observation ait été observée (non censurée).

La métrique est importée depuis la librairie scikit-survival (documentation scikit-survival), est montrée dans le benchmark, et est tronquée à 7 ans.

Données Cliniques (Une Ligne par Patient)
Chaque patient est associé à un identifiant unique et à des informations cliniques détaillées :

ID : Identifiant unique pour chaque patient.
CENTER : Le centre clinique où le patient est traité.
BM_BLAST : Pourcentage de blastes dans la moelle osseuse, indiquant la proportion de cellules sanguines anormales trouvées dans la moelle osseuse des patients.
WBC : Nombre de globules blancs, mesuré en Giga/L (paramètre sanguin).
ANC : Nombre absolu des neutrophiles dans le sang, en Giga/L (paramètre sanguin).
MONOCYTES : Nombre de monocytes dans le sang, en Giga/L (paramètre sanguin).
HB : Niveau d’hémoglobine, mesuré en g/dL (paramètre sanguin).
PLT : Nombre de plaquettes, en Giga/L (paramètre sanguin).
CYTOGENETICS :Description détaillée du caryotype du patient, représentant les anomalies chromosomiques détectées dans les cellules cancéreuses sanguines. Les notations suivent la norme ISCN, avec des marqueurs typiques comme 46,XX pour un caryotype féminin normal ou 46,XY pour un caryotype masculin normal. Des anomalies comme la monosomie 7 (perte du chromosome 7) indiquent une maladie à haut risque.
Données Moléculaires Génétiques (Une Ligne par Mutation Somatique par Patient)
Les mutations somatiques (acquises) sont des mutations spécifiques aux cellules cancéreuses, non trouvées dans les cellules normales du patient. Ces données capturent les mutations génétiques au niveau cellulaire :

ID : Identifiant unique pour chaque patient.
CHR, START, END : Position chromosomique de la mutation dans le génome humain.
REF, ALT : Nucléotides de référence et alternatifs (mutants).
GENE : Le gène affecté par la mutation.
PROTEIN_CHANGE : Impact de la mutation sur la protéine produite par le gène.
EFFECT : Classification générale de l’impact de la mutation sur la fonction du gène.
VAF : Fraction allélique variante, représentant la proportion de cellules portant la mutation.
Description du benchmark
Le benchmark fourni montre deux modèles :

Un modèle simple LightGBM (Light Gradient-Boosting Machine) utilisant uniquement des données cliniques, sans tenir compte de la censure.
Un modèle de risques proportionnels de Cox, qui prend en compte à la fois les données cliniques et des informations (limitées) sur les mutations génétiques.
Le premier benchmark (modèle LGBM) est fourni à titre d’exemple. C’est le score obtenu avec le deuxième benchmark (modèle de Cox) qui est retenu pour établir le score du benchmark. Un notebook Benchmark, avec le code utilisé pour générer les benchmarks, est fourni dans la section Fichiers.

À propos du modèle de risques proportionnels de Cox
Un modèle de risques proportionnels de Cox est une technique statistique, proche de la régression linéaire, utilisée en analyse de survie pour examiner l’effet de plusieurs variables sur le temps jusqu’à un événement spécifique. Il modélise la fonction de risque, qui représente le risque instantané de l’événement à un moment donné, en supposant que l’effet de chaque prédicteur est multiplicatif et reste constant dans le temps (d’où le terme “risques proportionnels”). Le modèle est semi-paramétrique, ce qui signifie qu’il ne suppose pas de fonction de risque de base spécifique, le rendant ainsi flexible pour analyser des données de temps jusqu’à événement, notamment avec des observations censurées. (Voir Wikipedia)

N’hésitez pas à créer vos propres features, à vous inspirer du benchmark existant et à construire vos propres modèles.

Bonne chance pour ce Data Challenge !
