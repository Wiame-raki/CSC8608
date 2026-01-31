# TP2
Nom: RAKI Wiame

## Question 1:
![PCA](../outputs/smoke.png)

## Question 2:

![Baseline Image](../outputs/baseline.png)

### Configuration Utilisée

*   **Modèle**: `runwayml/stable-diffusion-v1-5`
*   **Scheduler**: `DDIM`
*   **Seed**: `42`
*   **Prompt**: `A beautiful cat, photorealistic, 4k`
*   **Negative Prompt**: `ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft`
*   **Pas d'inférence (num_inference_steps)**: `30`
*   **Échelle de guidage (guidance_scale)**: `7.5`
*   **Dimensions**: `512x512`

## Expérimentations text2img


**Paramètres des expériences**

*   **Prompt**: "A professional e-commerce photograph of a luxury blue suede shoe, studio lighting, plain background"
*   **Seed**: 42

| Run ID           | Scheduler | Steps | Guidance (γ) | Objectif de l'expérience                     |
| :--------------- | :-------- | :---- | :----------- | :------------------------------------------- |
| `run01_baseline` | EulerA    | 30    | 7.5          | Référence de base                            |
| `run02_steps15`  | EulerA    | 15    | 7.5          | Évaluer l'effet de `steps` insuffisants      |
| `run03_steps50`  | EulerA    | 50    | 7.5          | Évaluer l'effet de `steps` élevés            |
| `run04_guid4`    | EulerA    | 30    | 4.0          | Évaluer l'effet d'une `guidance` faible      |
| `run05_guid12`   | EulerA    | 30    | 12.0         | Évaluer l'effet d'une `guidance` forte       |
| `run06_ddim`     | DDIM      | 30    | 7.5          | Comparer l'impact du `scheduler`             |



### Grille de Comparaison Visuelle

| Run 1: Baseline <br> (EulerA, 30 steps, γ=7.5) | Run 2: Steps bas <br> (15 steps) | Run 3: Steps hauts <br> (50 steps) |
| :---: | :---: | :---: |
| ![Baseline](../outputs/t2i_run01_baseline.png) | ![Steps 15](../outputs/t2i_run02_steps15.png) | ![Steps 50](../outputs/t2i_run03_steps50.png) |
| **Run 4: Guidance bas <br> (γ=4.0)** | **Run 5: Guidance haut <br> (γ=12.0)** | **Run 6: Scheduler DDIM** |
| ![Guidance 4](../outputs/t2i_run04_guid4.png) | ![Guidance 12](../outputs/t2i_run05_guid12.png) | ![DDIM](../outputs/t2i_run06_ddim.png) |


### Analyse Technique des Résultats

*   **Impact du Nombre de Pas d'Inférence (`steps`)**:
    *   **Run 2 (15 steps)**: Le processus de débruitage est manifestement incomplet. On note la présence d'artefacts et un manque de netteté global.
    *   **Run 3 (50 steps) vs Run 1 (30 steps)**: Le passage de 30 à 50 `steps` apporte un gain marginal en termes de finesse des micro-textures, mais au prix d'une augmentation significative du temps d'inférence. Le ratio coût/bénéfice favorise une valeur autour de 30 `steps`.

*   **Impact de l'Échelle de Guidage (`guidance_scale`)**:
    *   **Run 4 (γ=4.0)**: Une `guidance` faible accorde trop de liberté au modèle, qui s'écarte du concept de "photographie de produit". Le résultat est plus artistique mais ne respecte pas les contraintes commerciales (réalisme).
    *   **Run 5 (γ=12.0)**: À l'inverse, une `guidance` forte engendre une sur-optimisation. Les couleurs deviennent sur-saturées et les contrastes excessifs, dégradant le photoréalisme de l'image.

*   **Impact du `Scheduler`**:
    *   **Run 1 (EulerA) vs Run 6 (DDIM)**: `EulerA` a produit un rendu plus doux, avec des transitions douces. `DDIM` produit des images avec des textures plus "dures" et des détails plus marqués et plus réalistes. 
