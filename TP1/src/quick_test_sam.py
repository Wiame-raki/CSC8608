# TP1/src/quick_test_sam.py
import numpy as np
import cv2
from pathlib import Path
from sam_utils import load_sam_predictor, predict_mask_from_box

# prends la première image .jpg
img_path = next(Path("TP1/data/images").glob("*.jpg"))  # adapte si besoin
bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# chemin du checkpoint SAM
ckpt = "TP1/models/sam_vit_h_4b8939.pth"
pred = load_sam_predictor(ckpt, model_type="vit_h")  # vit_h, vit_l ou vit_b

# bbox "à la main"
box = np.array([50, 50, 250, 250], dtype=np.int32)

# prédiction du masque
mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)

# affiche les infos
print("img", rgb.shape, "mask", mask.shape, "score", score, "mask_sum", mask.sum())
