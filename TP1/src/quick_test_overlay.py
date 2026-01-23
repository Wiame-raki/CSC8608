# TP1/src/quick_test_overlay.py
import numpy as np
import cv2
from pathlib import Path

from sam_utils import load_sam_predictor, predict_mask_from_box
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay

# Dossier images et sorties
img_dir = Path("TP1/data/images")
out_dir = Path("TP1/outputs/overlays")
out_dir.mkdir(parents=True, exist_ok=True)

# Charger le modèle
ckpt = "TP1/models/sam_vit_h_4b8939.pth"
pred = load_sam_predictor(ckpt, model_type="vit_h")

# Boucle sur toutes les images
for img_path in img_dir.glob("*.jpg"):
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Sélection interactive de la boîte (appuie sur ENTER pour valider)
    print(f"Définis la boîte pour {img_path.name}...")
    r = cv2.selectROI("Select ROI", bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = r
    box = np.array([x, y, x + w, y + h], dtype=np.int32)

    # Prédiction du masque
    mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)

    # Calcul des métriques
    m_area = mask_area(mask)
    m_bbox = mask_bbox(mask)
    m_per = mask_perimeter(mask)

    # Overlay
    overlay = render_overlay(rgb, mask, box, alpha=0.5)

    out_path = out_dir / f"overlay_{img_path.stem}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"{img_path.name} -> score: {score:.3f}, area: {m_area}, bbox: {m_bbox}, perimeter: {m_per:.1f}")
    print("saved:", out_path)
