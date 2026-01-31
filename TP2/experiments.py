import torch
from PIL import Image
from pipeline_utils import (
    load_text2img,
    make_generator,
    DEFAULT_MODEL_ID,
    get_device,
    to_img2img,
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

def run_text2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    prompt = "A professional e-commerce photograph of a luxury blue suede shoe, studio lighting, plain background"
    negative = "text, watermark, logo, low quality, blurry, deformed"

    plan = [
        # name, scheduler, steps, guidance
        ("run01_baseline", "EulerA", 30, 7.5),
        ("run02_steps15", "EulerA", 15, 7.5),
        ("run03_steps50", "EulerA", 50, 7.5),
        ("run04_guid4",  "EulerA", 30, 4.0),
        ("run05_guid12", "EulerA", 30, 12.0),
        ("run06_ddim",   "DDIM",   30, 7.5),
    ]

    for name, scheduler_name, steps, guidance in plan:
        pipe = load_text2img(model_id, scheduler_name)
        device = get_device()
        g = make_generator(seed, device)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512,
            generator=g,
        )

        img = out.images[0]
        img.save(f"TP2/outputs/t2i_{name}.png")
        print("T2I", name, {"scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance})

def run_img2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    scheduler_name = "EulerA"
    steps = 30
    guidance = 7.5

    init_path = "TP2/inputs/product.jpg"

    prompt = "A professional studio photograph of a bright red running shoe, on a pure white background"
    negative = "text, watermark, logo, low quality, blurry, deformed"

    strengths = [
        ("run07_strength035", 0.35),
        ("run08_strength060", 0.60),
        ("run09_strength085", 0.85),
    ]

    pipe_t2i = load_text2img(model_id, scheduler_name)
    pipe_i2i = to_img2img(pipe_t2i)

    device = get_device()
    g = make_generator(seed, device)

    init_image = Image.open(init_path).convert("RGB")
    init_image = init_image.resize((512, 512))

    for name, strength in strengths:
        # We need to create a new generator for each run to ensure the seed is reset
        g = make_generator(seed, device)
        out = pipe_i2i(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
        )
        img = out.images[0]
        img.save(f"TP2/outputs/i2i_{name}.png")
        print("I2I", name, {"scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance, "strength": strength})


if __name__ == "__main__":
    # run_text2img_experiments()
    run_img2img_experiments()