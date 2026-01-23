import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from sam_utils import load_sam_predictor, predict_mask_from_box
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay


DATA_DIR = Path("TP1/data/images")
OUT_DIR = Path("TP1/outputs/overlays")
CKPT_PATH = "TP1/models/sam_vit_h_4b8939.pth"   # checkpoint SAM
MODEL_TYPE = "vit_h"
MAX_DIM = 1024  # Maximum dimension for images


def load_image_rgb(path: Path, max_dim: int = MAX_DIM) -> np.ndarray:
    """
    Load image, convert to RGB and resize if necessary.
    Max dimension won't exceed max_dim to save GPU memory.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Image illisible: {path}")
    
    # Auto-resize if too large
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # INTER_AREA is better for shrinking
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


@st.cache_resource
def get_predictor():
    # Load SAM once: important to avoid reloading at each UI interaction
    return load_sam_predictor(CKPT_PATH, model_type=MODEL_TYPE)


def draw_box_preview(image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """Draw bounding box preview on image"""
    preview = image_rgb.copy()
    bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
    cv2.rectangle(
        bgr,
        (x1, y1),
        (x2, y2),
        color=(0, 255, 0),   # green
        thickness=2
    )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --- Page config ---
st.set_page_config(page_title="TP1 - SAM Segmentation", layout="wide")
st.title("TP1 — Segmentation interactive (SAM)")

# 1) Image list
imgs = sorted([p for p in DATA_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
if len(imgs) == 0:
    st.error("Aucune image trouvée dans TP1/data/images/")
    st.stop()

img_name = st.selectbox("Choisir une image", [p.name for p in imgs])
img_path = DATA_DIR / img_name
img = load_image_rgb(img_path, max_dim=MAX_DIM)  # Load with auto-resize
H, W = img.shape[:2]

# 2) Display image
st.image(img, caption=f"{img_name} ({W}x{H}) [resized if > {MAX_DIM}px]", use_container_width=True)

# 3) Bounding box sliders
st.subheader("Bounding box (pixels)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    x1 = st.slider("x1", 0, W - 1, 0)
with col2:
    y1 = st.slider("y1", 0, H - 1, 0)
with col3:
    x2 = st.slider("x2", 0, W - 1, W - 1)
with col4:
    y2 = st.slider("y2", 0, H - 1, H - 1)

# Normalize bbox (x1<x2, y1<y2)
x_min, x_max = (x1, x2) if x1 < x2 else (x2, x1)
y_min, y_max = (y1, y2) if y1 < y2 else (y2, y1)
box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

# --- Box preview (before segmentation) ---
preview = draw_box_preview(img, box)
st.image(
    preview,
    caption="Prévisualisation : bounding box (avant segmentation)",
    use_container_width=True
)

# --- Warning for small bounding boxes ---
MIN_BOX_SIZE = 30  # pixels

if (x_max - x_min) < MIN_BOX_SIZE or (y_max - y_min) < MIN_BOX_SIZE:
    st.warning("⚠️ BBox très petite : la segmentation risque d'être mauvaise. Essayez une bbox plus large.")

# 4) Segmentation
do_segment = st.button("Segmenter", use_container_width=True)
if do_segment:
    predictor = get_predictor()

    t0 = time.time()
    mask, score = predict_mask_from_box(predictor, img, box, multimask=True)
    dt = (time.time() - t0) * 1000.0

    overlay = render_overlay(img, mask, box, alpha=0.5)

    m_area = mask_area(mask)
    m_bbox = mask_bbox(mask)
    m_per = mask_perimeter(mask)

    st.subheader("Résultat")
    st.image(overlay, caption=f"score={score:.3f} | time={dt:.1f} ms", use_container_width=True)

    st.write({
        "score": float(score),
        "time_ms": float(dt),
        "area_px": int(m_area),
        "mask_bbox": m_bbox,
        "perimeter": float(m_per),
    })

    # 5) Save overlay
    save = st.button("Sauvegarder overlay")
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"overlay_{img_path.stem}.png"
        # Convert RGB back to BGR for cv2.imwrite
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), overlay_bgr)
        st.success(f"✅ Sauvegardé: {out_path}")