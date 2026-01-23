# TP1 — Segment Anything (SAM)

## Dépôt du projet
Lien du dépôt Git :
https://github.com/<username>/<repo>  


## Arboresence TP1:
```markdown
TP1/
├── data/
│   └── images/
├── src/
│   ├── app.py
│   ├── sam_utils.py
│   ├── geom_utils.py
│   └── viz_utils.py
├── outputs/
│   ├── overlays/
│   └── logs/
├── report/
│   └── report.md
├── requirements.txt
└── README.md
```
## Environnement d’exécution
Le TP a été exécuté sur un **nœud GPU via SLURM**.

- Machine : `arcadia-slurm-node-1`
- Accès GPU : via allocation SLURM (`srun --gres=gpu:1`)

## Environnement logiciel
- Gestionnaire d’environnement : **conda (Miniforge)**
- Environnement activé :
```bash
torch_env
```
- Version :
```bash
cpu = _conversion_method_template(device=torch.device("cpu"))
torch 2.5.1
cuda_available True
device_count 1
```
## `segment_anything` fonctionne
![PCA](./img/ok.png)

## Streamlit:
- Port choisi : `8511`
![PCA](./img/streamlit.png)
**UI accessible via SSH tunnel : oui**

## Question 2:
*images récupérées via recherche web* 
![PCA](./img/img.png)

1. **`mockup-graphics-_mUVHhvBYZ0-unsplash.jpg`** (Feuilles sur fond blanc)
> C'est l'archétype parfait de l'image **“simple”** avec un sujet organique unique et un fond blanc uni, idéal pour évaluer la précision des contours.


2. **`adrian-rosco-stef-KGw1AOyBTQM-unsplash.jpg`** (Cheveux roux)
> Elle représente la catégorie **“difficile”** par excellence, mettant au défi la gestion des textures fines (cheveux) et des reflets lumineux complexes.


3. **`yoav-aziz-tKCd-IWc4gI-unsplash.jpg`** (Ruelle avec lanternes)
> Une superbe image **“chargée”** qui cumule les difficultés : perspective profonde, nombreux objets qui se chevauchent (occlusion) et éclairage varié.


4. **`jezael-melgoza-_noSmX8Kgoo-unsplash.jpg`** (Rue néons violets)
> Sélectionnée pour son fond **“chargé”** urbain moderne, elle teste la capacité à gérer des sources de lumière artificielles intenses et une foule en mouvement.


5. **`brian-patrick-tagalog-s0FBvCk9-DU-unsplash.jpg`** (Prise électrique)
> Une image **“simple”** intéressante car, contrairement aux feuilles, elle n'est pas sur fond blanc : elle teste la gestion des ombres portées et du faible contraste sur le mur bleu.

### Cas simple:
![PCA](./img/mockup-graphics-_mUVHhvBYZ0-unsplash.jpg)
### Cas complexe:
![PCA](./img/adrian-rosco-stef-KGw1AOyBTQM-unsplash.jpg)

## Question 3 :

**Modèle choisi :** `vit_h`
**Checkpoint utilisé :** `sam_vit_h_4b8939.pth`

**Sortie du test rapide :**

![PCA](./img/quicktest.png)

**Premier constat :**
Le modèle fonctionne correctement et détecte le masque principal. L’inférence est un peu lente sur des images de très haute résolution (ici 5472×3648). On remarque que certains détails fins ne sont pas parfaitement segmentés, mais globalement le masque est cohérent. Ce test permet de vérifier que l’intégration de SAM avec le SamPredictor est opérationnelle.
