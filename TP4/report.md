## Question 2.g:
On calcule les métriques séparément sur `train_mask`, `val_mask` et `test_mask` pour respecter un protocole d’évaluation propre et exploitable en pratique.
Le **train_mask** permet de suivre la dynamique d’apprentissage (est-ce que la loss baisse, est-ce que le modèle apprend réellement le signal ?).
Le **val_mask** sert à piloter les décisions d’ingénierie : tuning d’hyperparamètres, early stopping, choix d’architecture.
Le **test_mask** reste isolé pour estimer la performance finale sans fuite d’information ni biais d’optimisation.
Ici, on évalue à chaque epoch car Cora est petit ; sur un graphe industriel beaucoup plus large, on limiterait la fréquence d’évaluation pour réduire le coût compute.
