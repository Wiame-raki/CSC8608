import numpy as np
import cv2
from pathlib import Path

# Then your function:
def load_image_rgb(path: Path, max_dim: int = 1024) -> np.ndarray:
    """
    Charge une image, convertit en RGB et redimensionne si nécessaire
    pour que le plus grand côté ne dépasse pas max_dim.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Image illisible: {path}")
    
    # --- Redimensionnement automatique ---
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # INTER_AREA est meilleur pour réduire la taille proprement
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # -------------------------------------

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb