import torch
from pipeline_utils import (
    load_text2img,
    make_generator,
    DEFAULT_MODEL_ID,
)

def main():
    """
    Script minimal pour générer une image "baseline" avec des paramètres définis.
    """
    # --- Paramètres ---
    prompt = "A beautiful cat, photorealistic, 4k"
    negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft"
    
    width = 512
    height = 512
    
    num_inference_steps = 30  # Nombre d'étapes de débruitage
    guidance_scale = 7.5  # Force de l'alignement sur le prompt
    
    seed = 42
    scheduler_name = "DDIM"
    
    output_path = "TP2/outputs/baseline.png"

    # --- Pipeline ---
    
    # Charger le pipeline text2img
    pipe = load_text2img(DEFAULT_MODEL_ID, scheduler_name=scheduler_name)
    
    # Créer un générateur pour la reproductibilité
    generator = make_generator(seed, device=pipe.device)

    # --- Génération ---
    
    print("Génération de l'image...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # --- Sauvegarde ---
    
    image.save(output_path)
    print(f"Image sauvegardée dans '{output_path}'")


if __name__ == "__main__":
    main()
