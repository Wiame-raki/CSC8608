## Question 2.g:
On calcule les métriques séparément sur `train_mask`, `val_mask` et `test_mask` pour respecter un protocole d’évaluation propre et exploitable en pratique.
Le **train_mask** permet de suivre la dynamique d’apprentissage (est-ce que la loss baisse, est-ce que le modèle apprend réellement le signal ?).
Le **val_mask** sert à piloter les décisions d’ingénierie : tuning d’hyperparamètres, early stopping, choix d’architecture.
Le **test_mask** reste isolé pour estimer la performance finale sans fuite d’information ni biais d’optimisation.
Ici, on évalue à chaque epoch car Cora est petit ; sur un graphe industriel beaucoup plus large, on limiterait la fréquence d’évaluation pour réduire le coût compute.

## Question 3.e:
| Modèle | test_acc | test_f1 | Temps total train (s) |
| ------ | -------- | ------- | --------------------- |
| GCN    | 0.8040   | 0.7940  | 1.51                  |

## Question 3.f:
Sur Cora, le **GCN** peut surpasser le **MLP** car il combine les **features des nœuds** avec le **signal topologique** du graphe. Grâce au **message passing**, chaque nœud intègre l’information de ses voisins, ce qui produit un **lissage des représentations** et renforce la cohérence des classes dans les zones à forte **homophilie**.
Si les **features brutes** sont déjà très discriminantes, le gain du GCN reste limité : un MLP peut suffire à séparer les classes.
En revanche, dès que la topologie apporte un signal pertinent (ex. papiers citant des articles du même sujet), le GCN améliore la généralisation sur validation et test en fusionnant information locale et structure globale.
Sur Cora, petit et fortement homophile, le GCN est à la fois efficace et rapide. Sur des graphes plus bruyants ou faiblement connectés, l’avantage peut disparaître et le MLP redevenir compétitif.
