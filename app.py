import cv2
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import Optional
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os
from collections import Counter
import pandas as pd
import base64
import io
import json
import requests



# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Room Decoration Analyser",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7F4EF;
    color: #2C2C2C;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #2C2C2C;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    font-size: 0.9rem;
    color: #8B6F47;
    margin-bottom: 2rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.nv-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #EDE8E0;
}
.card-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8B6F47;
    font-weight: 600;
}
.card-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #2C2C2C;
    font-weight: 600;
    margin-top: 0.2rem;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #2C2C2C;
    margin: 1.2rem 0 0.8rem 0;
    border-bottom: 1px solid #EDE8E0;
    padding-bottom: 0.4rem;
}
.algo-box {
    background: #F0EDE8;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    border-left: 4px solid #8B6F47;
}
.algo-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8B6F47;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.algo-desc {
    font-size: 0.78rem;
    color: #5C6875;
    line-height: 1.7;
}
.conf-bar-bg {
    background: #EDE8E0;
    border-radius: 99px;
    height: 6px;
    width: 100%;
}
.conf-bar-fill {
    height: 6px;
    border-radius: 99px;
    background: #8B6F47;
}
.stButton > button {
    background: #2C2C2C !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    width: 100%;
}
.stButton > button:hover {
    background: #8B6F47 !important;
}

/* New feature styles */
.improvement-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F9F6F1 100%);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07);
    border: 1px solid #EDE8E0;
    transition: transform 0.2s ease;
}
.improvement-card:hover {
    transform: translateY(-2px);
}
.priority-high {
    border-left: 5px solid #B85C38;
}
.priority-medium {
    border-left: 5px solid #C9962A;
}
.priority-low {
    border-left: 5px solid #6B8F71;
}
.priority-badge {
    display: inline-block;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 99px;
    margin-bottom: 0.5rem;
}
.badge-high { background: #FFF0EC; color: #B85C38; }
.badge-medium { background: #FFF8EC; color: #C9962A; }
.badge-low { background: #F0F7F0; color: #6B8F71; }
.product-chip {
    display: inline-block;
    background: #F0EDE8;
    border: 1px solid #DDD8D0;
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 0.68rem;
    color: #5C6875;
    margin: 2px;
    font-weight: 500;
}
.before-after-label {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    text-align: center;
    padding: 0.5rem;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}
.label-before {
    background: #EDE8E0;
    color: #5C6875;
}
.label-after {
    background: #2C2C2C;
    color: #F7F4EF;
}
.ai-thinking {
    background: linear-gradient(90deg, #F0EDE8, #E8E3DC, #F0EDE8);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    color: #8B6F47;
    font-size: 0.8rem;
}
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.product-category {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #2C2C2C;
    margin: 1rem 0 0.5rem;
    font-weight: 600;
}
.budget-tag {
    font-size: 0.65rem;
    color: #8B6F47;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  ZONE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

ZONE_DEFINITIONS = {
    "Living / Lounge": {
        "icon": "🛋️",
        "anchors":   {"couch", "sofa", "tv"},
        "supports":  {"chair", "remote", "potted plant", "vase", "clock", "book", "coffee table"},
        "conflicts": {"bed", "oven", "sink", "refrigerator", "microwave", "toilet"},
        "color_bgr": (124, 154, 130),
        "color_hex": "#6B8F71",
        "tip": "Keep a clear 90 cm path around seating. Add a rug to anchor the conversation zone.",
        "decor": "Neutral tones with warm accent colours. Layer textures — linen, wool, wood.",
    },
    "Work / Study": {
        "icon": "💻",
        "anchors":   {"laptop", "keyboard", "mouse"},
        "supports":  {"chair", "book", "cell phone", "clock"},
        "conflicts": {"bed", "couch", "oven", "sink"},
        "color_bgr": (92, 103, 160),
        "color_hex": "#5C6875",
        "tip": "Position screen perpendicular to windows to cut glare.",
        "decor": "Minimalist design reduces distraction. Use a desktop plant for air quality.",
    },
    "Dining": {
        "icon": "🍽️",
        "anchors":   {"dining table"},
        "supports":  {"chair", "bowl", "cup", "wine glass", "fork", "knife", "spoon", "bottle"},
        "conflicts": {"bed", "laptop", "toilet"},
        "color_bgr": (76, 168, 201),
        "color_hex": "#C9962A",
        "tip": "Allow 75 cm clearance on all sides of the table.",
        "decor": "A pendant light centred over the table creates ambience and defines the zone.",
    },
    "Bedroom / Rest": {
        "icon": "🛏️",
        "anchors":   {"bed"},
        "supports":  {"clock", "book", "cell phone", "teddy bear", "hair drier"},
        "conflicts": {"oven", "dining table", "sink", "refrigerator"},
        "color_bgr": (140, 100, 180),
        "color_hex": "#8B5EA8",
        "tip": "Avoid screens within 1 m of the bed for better sleep quality.",
        "decor": "Soft lighting, blackout curtains, and muted tones promote deep rest.",
    },
    "Kitchen / Cooking": {
        "icon": "🍳",
        "anchors":   {"oven", "microwave", "refrigerator", "sink", "toaster"},
        "supports":  {"bowl", "bottle", "cup", "knife", "fork", "spoon"},
        "conflicts": {"bed", "couch", "laptop"},
        "color_bgr": (56, 180, 210),
        "color_hex": "#B85C38",
        "tip": "The work triangle (fridge → sink → oven) should sum to 4-7 m.",
        "decor": "Open shelving with plants and ceramics adds warmth to functional kitchens.",
    },
    "Fitness / Active": {
        "icon": "🏋️",
        "anchors":   {"sports ball", "skateboard", "surfboard", "tennis racket"},
        "supports":  {"person", "frisbee"},
        "conflicts": {"bed", "oven", "sink"},
        "color_bgr": (60, 180, 100),
        "color_hex": "#3CAA64",
        "tip": "Allow 2×2 m of open floor per activity zone.",
        "decor": "Rubber flooring, mirrors, and motivational wall art define the space.",
    },
    "Pet Zone": {
        "icon": "🐾",
        "anchors":   {"dog", "cat", "bird"},
        "supports":  {"teddy bear", "bowl"},
        "conflicts": set(),
        "color_bgr": (180, 140, 80),
        "color_hex": "#A07850",
        "tip": "Place pet zones away from work desks and dining areas.",
        "decor": "Washable rugs and built-in pet nooks keep the space tidy and stylish.",
    },
}

CONFLICT_PAIRS = [
    ("Work / Study",      "Bedroom / Rest",   "Screen light near sleep area disrupts circadian rhythm."),
    ("Work / Study",      "Living / Lounge",  "Work and relaxation zones overlap — consider a visual divider."),
    ("Dining",            "Work / Study",     "Eating at the desk is linked to lower productivity."),
    ("Kitchen / Cooking", "Bedroom / Rest",   "Cooking smells and heat intrude on the rest zone."),
    ("Fitness / Active",  "Living / Lounge",  "Active zone too close to fragile furnishings."),
]

# ════════════════════════════════════════════════════════════════════════════
#  PRODUCT DATABASE  (used in "How to Improve" tab)
# ════════════════════════════════════════════════════════════════════════════

ZONE_PRODUCTS = {
    "Living / Lounge": {
        "Lighting": [
            {"name": "Arc Floor Lamp", "price": "₹3,500–8,000", "why": "Fills corners, eliminates harsh overhead shadows"},
            {"name": "LED Strip (warm 2700K)", "price": "₹800–2,000", "why": "Adds ambient glow behind TV / shelves"},
            {"name": "Table Lamp with Linen Shade", "price": "₹1,500–4,000", "why": "Soft pools of light for conversation areas"},
        ],
        "Furniture": [
            {"name": "Jute / Wool Area Rug (2×3 m)", "price": "₹4,000–12,000", "why": "Anchors the seating group and absorbs echo"},
            {"name": "Nesting Side Tables", "price": "₹2,500–6,000", "why": "Flexible surface area without bulk"},
            {"name": "Open Bookshelf / Etagère", "price": "₹5,000–15,000", "why": "Display + storage without closing the space"},
        ],
        "Decor": [
            {"name": "Large Canvas Art (80×100 cm)", "price": "₹2,000–8,000", "why": "Draws the eye up and sets the colour tone"},
            {"name": "Ceramic Vase Set", "price": "₹600–2,500", "why": "Organic shapes soften hard furniture lines"},
            {"name": "Indoor Plant (Monstera / Fiddle Leaf)", "price": "₹800–3,000", "why": "Brings life, improves air quality"},
        ],
    },
    "Work / Study": {
        "Lighting": [
            {"name": "LED Desk Lamp (4000K, dimmable)", "price": "₹1,200–4,000", "why": "Task light reduces eye strain during long sessions"},
            {"name": "Bias Lighting Strip (behind monitor)", "price": "₹600–1,500", "why": "Reduces contrast fatigue on the eyes"},
        ],
        "Furniture": [
            {"name": "Ergonomic Chair", "price": "₹8,000–25,000", "why": "Lumbar support is non-negotiable for 4+ hour sessions"},
            {"name": "Monitor Arm / Riser", "price": "₹1,500–5,000", "why": "Raises screen to eye level, frees desk space"},
            {"name": "Cable Management Tray", "price": "₹400–1,200", "why": "Clean desk = clear mind"},
        ],
        "Decor": [
            {"name": "Pinboard / Pegboard Wall Panel", "price": "₹800–3,000", "why": "Keeps notes visible and off the desk"},
            {"name": "Small Succulent Planter", "price": "₹300–800", "why": "Low maintenance, boosts mood and focus"},
        ],
    },
    "Bedroom / Rest": {
        "Lighting": [
            {"name": "Bedside Lamp (warm 2200K)", "price": "₹1,000–3,500", "why": "Low colour temperature signals body to wind down"},
            {"name": "Blackout Curtains", "price": "₹2,500–7,000", "why": "Blocks early morning light for deeper sleep"},
            {"name": "Smart Bulb (sunset mode)", "price": "₹800–2,000", "why": "Gradual dimming mimics natural dusk"},
        ],
        "Furniture": [
            {"name": "Upholstered Headboard", "price": "₹5,000–18,000", "why": "Adds warmth and acts as a focal point"},
            {"name": "Under-Bed Storage Drawers", "price": "₹3,000–8,000", "why": "Hides clutter that disrupts mental rest"},
            {"name": "Bedside Tray / Floating Shelf", "price": "₹500–2,000", "why": "Phone and book off the mattress = better boundary"},
        ],
        "Decor": [
            {"name": "Weighted Blanket (6–8 kg)", "price": "₹3,000–9,000", "why": "Deep pressure stimulation improves sleep quality"},
            {"name": "Diffuser + Lavender Oil", "price": "₹800–2,500", "why": "Scent cue trains the brain to associate room with sleep"},
        ],
    },
    "Dining": {
        "Lighting": [
            {"name": "Pendant Light (above table)", "price": "₹3,000–12,000", "why": "Defines the zone and creates intimate atmosphere"},
            {"name": "Dimmer Switch", "price": "₹400–1,200", "why": "Shift from bright lunch to cosy dinner mood"},
        ],
        "Furniture": [
            {"name": "Bench Seating (one side)", "price": "₹4,000–10,000", "why": "Adds capacity and a relaxed, modern look"},
            {"name": "Table Runner + Placemats Set", "price": "₹600–2,000", "why": "Defines individual place settings and protects surface"},
        ],
        "Decor": [
            {"name": "Low Centrepiece Bowl / Candles", "price": "₹500–2,500", "why": "Keeps sightlines open across the table"},
            {"name": "Wall-mounted Spice Rack", "price": "₹800–2,500", "why": "Keeps kitchen items nearby without cluttering table"},
        ],
    },
    "Kitchen / Cooking": {
        "Lighting": [
            {"name": "Under-Cabinet LED Strip", "price": "₹700–2,500", "why": "Illuminates work surfaces, eliminates prep-area shadows"},
        ],
        "Furniture": [
            {"name": "Magnetic Knife Strip", "price": "₹600–2,000", "why": "Frees drawer space, knives visible and safe"},
            {"name": "Rolling Kitchen Cart", "price": "₹4,000–12,000", "why": "Extra prep surface that moves out of the way"},
        ],
        "Decor": [
            {"name": "Ceramic Canister Set", "price": "₹1,200–3,500", "why": "Consistent containers reduce visual clutter on counters"},
            {"name": "Herb Garden (windowsill)", "price": "₹500–1,500", "why": "Fresh herbs + greenery for a lived-in, styled look"},
        ],
    },
    "Fitness / Active": {
        "Lighting": [
            {"name": "Bright Daylight Bulbs (5000K)", "price": "₹500–1,500", "why": "High colour temperature boosts alertness during workouts"},
        ],
        "Furniture": [
            {"name": "Rubber Interlocking Floor Mat", "price": "₹2,000–8,000", "why": "Protects floor, reduces joint impact, defines zone"},
            {"name": "Full-Length Mirror", "price": "₹3,000–9,000", "why": "Check form and makes the space feel larger"},
        ],
        "Decor": [
            {"name": "Motivational Wall Art / Vinyl Decal", "price": "₹500–2,500", "why": "Visual cue that reinforces the zone's purpose"},
        ],
    },
    "Pet Zone": {
        "Furniture": [
            {"name": "Washable Pet Bed (raised)", "price": "₹1,500–5,000", "why": "Defined sleeping spot reduces furniture use"},
            {"name": "Built-in / Corner Feeding Station", "price": "₹2,000–6,000", "why": "Tidier than floor bowls, easier to clean"},
        ],
        "Decor": [
            {"name": "Washable Accent Rug", "price": "₹1,500–5,000", "why": "Machine-washable material handles pet traffic"},
            {"name": "Wall-mounted Toy Storage", "price": "₹600–2,000", "why": "Keeps toys off floor and contained"},
        ],
    },
}

# ════════════════════════════════════════════════════════════════════════════
#  DATA CLASS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneResult:
    name: str
    icon: str
    confidence: float
    detected_labels: list
    color_hex: str
    tip: str
    decor: str
    bbox_union: Optional[tuple] = None


# ════════════════════════════════════════════════════════════════════════════
#  YOLO MODEL
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            st.error("❌ yolov8n.pt not found. Place it in the same folder as app.py.")
            return None
        return YOLO(model_path)
    except ImportError:
        st.error("❌ Run: pip install ultralytics")
        return None


def enhance_image(img_rgb: np.ndarray) -> np.ndarray:
    """Brighten dark images using CLAHE on the L channel."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def run_yolo_detection(img_rgb: np.ndarray, conf_threshold: float = 0.3, enhance: bool = False):
    model = load_yolo_model()
    if model is None:
        return []
    if enhance:
        img_rgb = enhance_image(img_rgb)
    results = model(img_rgb, conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id].lower()
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label":   label,
                "display": f"{label} {conf:.0%}",
                "conf":    conf,
                "bbox":    (x1, y1, x2, y2),
            })
    return detections


# ════════════════════════════════════════════════════════════════════════════
#  ZONE ENGINE
# ════════════════════════════════════════════════════════════════════════════

def detect_room_zones(detections, img_shape):
    H, W = img_shape[:2]
    detected_labels = {d["label"] for d in detections}
    label_to_boxes  = {}
    for d in detections:
        label_to_boxes.setdefault(d["label"], []).append(d["bbox"])

    zone_results = []
    for zone_name, zdef in ZONE_DEFINITIONS.items():
        anchors_hit   = detected_labels & zdef["anchors"]
        supports_hit  = detected_labels & zdef["supports"]
        conflicts_hit = detected_labels & zdef["conflicts"]
        if not anchors_hit and not supports_hit:
            continue
        anchor_score     = len(anchors_hit)   * 40.0
        support_score    = len(supports_hit)  * 12.0
        conflict_penalty = len(conflicts_hit) * 15.0
        raw  = min(anchor_score + support_score - conflict_penalty, 100.0)
        conf = max(raw, 5.0) if anchors_hit else max(raw * 0.6, 5.0)

        all_boxes = []
        for lbl in anchors_hit | supports_hit:
            all_boxes.extend(label_to_boxes.get(lbl, []))
        bbox_union = None
        if all_boxes:
            bbox_union = (min(b[0] for b in all_boxes), min(b[1] for b in all_boxes),
                          max(b[2] for b in all_boxes), max(b[3] for b in all_boxes))

        zone_results.append(ZoneResult(
            name=zone_name, icon=zdef["icon"],
            confidence=round(conf, 1),
            detected_labels=sorted(anchors_hit | supports_hit),
            color_hex=zdef["color_hex"],
            tip=zdef["tip"],
            decor=zdef["decor"],
            bbox_union=bbox_union,
        ))

    zone_results.sort(key=lambda z: z.confidence, reverse=True)

    active_zones = {z.name for z in zone_results if z.confidence >= 30}
    conflict_warnings = []
    for z1, z2, msg in CONFLICT_PAIRS:
        if z1 in active_zones and z2 in active_zones:
            conflict_warnings.append(f"⚠️ {z1} + {z2}: {msg}")

    coverage_mask = np.zeros((H, W), dtype=np.uint8)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        coverage_mask[y1:y2, x1:x2] = 255
    density   = np.sum(coverage_mask > 0) / (H * W)
    clearance = "good" if density < 0.15 else "tight" if density < 0.35 else "crowded"
    heatmap   = _build_heatmap(detections, img_shape, zone_results)

    return {"zones": zone_results, "conflicts": conflict_warnings,
            "density": density, "clearance": clearance, "heatmap": heatmap}


def _build_heatmap(detections, img_shape, zone_results):
    H, W = img_shape[:2]
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 245
    label_to_color = {}
    for zone in zone_results:
        zdef = ZONE_DEFINITIONS[zone.name]
        for lbl in list(zdef["anchors"]) + list(zdef["supports"]):
            if lbl not in label_to_color:
                label_to_color[lbl] = zdef["color_bgr"]
    overlay = canvas.copy()
    for d in detections:
        color = label_to_color.get(d["label"], (180, 180, 180))
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    blended = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    for d in detections:
        color = label_to_color.get(d["label"], (120, 120, 120))
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
        text = d["display"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(blended, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(blended, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def draw_boxes(img_rgb, detections):
    img = img_rgb.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (46, 139, 87), 2)
        text = d["display"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (46, 139, 87), -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_zone_results(detections, img_rgb, prefix=""):
    """Shared renderer for both upload and webcam tabs."""
    if not detections:
        st.warning("No objects detected. Try lowering the confidence threshold.")
        return

    result      = detect_room_zones(detections, img_rgb.shape)
    zones       = result["zones"]
    conflicts   = result["conflicts"]
    density     = result["density"]
    clearance   = result["clearance"]
    heatmap     = result["heatmap"]
    density_pct = round(density * 100, 1)

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem">
            <span class="card-label">Zones Detected</span>
            <div class="card-value">{len(zones)}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem">
            <span class="card-label">Furniture Density</span>
            <div class="card-value">{density_pct}%</div></div>""", unsafe_allow_html=True)
    with m3:
        clr_color = {"good": "#6B8F71", "tight": "#C9962A", "crowded": "#B85C38"}[clearance]
        clr_label = {"good": "✓ Good", "tight": "⚡ Tight", "crowded": "⚠️ Crowded"}[clearance]
        st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem">
            <span class="card-label">Clearance Rating</span>
            <div class="card-value" style="color:{clr_color};font-size:1.4rem">{clr_label}</div>
            </div>""", unsafe_allow_html=True)

    # Heatmap + Zone list
    col_map, col_list = st.columns([1.2, 1], gap="large")
    with col_map:
        st.markdown('<div class="section-title">Zone Map</div>', unsafe_allow_html=True)
        st.image(heatmap, use_column_width=True,
                 caption="Objects coloured by functional zone")
    with col_list:
        st.markdown('<div class="section-title">Detected Zones</div>', unsafe_allow_html=True)
        if not zones:
            st.info("No zones identified.")
        else:
            for z in zones:
                labels_str = ", ".join(z.detected_labels) or "—"
                st.markdown(f"""
                <div class="nv-card" style="padding:1rem;border-left:4px solid {z.color_hex}">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-family:'Playfair Display',serif;font-size:1.05rem">
                            {z.icon} {z.name}</span>
                        <span style="color:{z.color_hex};font-weight:600">{z.confidence:.0f}%</span>
                    </div>
                    <div class="conf-bar-bg" style="margin:6px 0 8px">
                        <div class="conf-bar-fill"
                             style="width:{z.confidence:.0f}%;background:{z.color_hex}"></div>
                    </div>
                    <div style="font-size:0.65rem;color:#8B6F47;margin-bottom:4px;letter-spacing:1px">
                        SIGNALS · {labels_str}</div>
                    <div style="font-size:0.72rem;color:#5C6875;line-height:1.6">
                        💡 {z.tip}</div>
                    <div style="font-size:0.72rem;color:#8B6F47;line-height:1.6;margin-top:4px">
                        🎨 {z.decor}</div>
                </div>""", unsafe_allow_html=True)

    # Conflicts
    if conflicts:
        st.markdown('<div class="section-title">⚠️ Zone Conflicts</div>', unsafe_allow_html=True)
        for msg in conflicts:
            st.markdown(f"""<div class="nv-card" style="border-left:4px solid #B85C38;padding:0.9rem">
                <span style="font-size:0.8rem;color:#B85C38">{msg}</span></div>""",
                unsafe_allow_html=True)
        st.markdown("""<div class="nv-card">
            <span class="card-label">2026 Decoration Solutions</span>
            <div style="font-size:0.78rem;color:#5C6875;line-height:1.9;margin-top:0.5rem">
                • <b>Room dividers</b> — open-frame shelving or tall plant rows create visual separation.<br>
                • <b>Lighting zones</b> — warm 2700K for rest/dining; cool 4000K for work areas.<br>
                • <b>Rug anchoring</b> — one rug per zone defines territory on open-plan floors.<br>
                • <b>Modular furniture</b> — fold-away desks and sofa-beds allow temporal zone changes.
            </div></div>""", unsafe_allow_html=True)
    else:
        if zones:
            st.success("✓ No zone conflicts — the room's functional areas are well separated.")

    # Multi-use score
    if len(zones) >= 2:
        score = min(len(zones) * 22 + density * 30, 100)
        st.markdown(f"""<div class="nv-card" style="margin-top:1rem;text-align:center">
            <span class="card-label">Multi-Purpose Utilisation Score</span>
            <div class="card-value">{score:.0f} / 100</div>
            <div class="conf-bar-bg" style="margin:0.6rem auto;max-width:300px">
                <div class="conf-bar-fill" style="width:{score:.0f}%"></div></div>
            <div style="font-size:0.7rem;color:#8B6F47;margin-top:4px">
                {len(zones)} zones · {density_pct}% area utilised ·
                {"No conflicts" if not conflicts else f"{len(conflicts)} conflict(s)"}
            </div></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  AI IMPROVEMENT ENGINE  (calls Claude API)
# ════════════════════════════════════════════════════════════════════════════

def img_to_base64(img_rgb: np.ndarray) -> str:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def get_ai_improvement_plan(img_rgb, zones, conflicts, density, clearance, budget):
    import base64, io, json
    from groq import Groq
    from PIL import Image

    zone_names     = [z.name for z in zones]
    conflict_msgs  = conflicts if conflicts else ["None detected"]
    detected_items = list({lbl for z in zones for lbl in z.detected_labels})

    # Convert image to base64 for vision model
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    b64_image = base64.b64encode(buf.getvalue()).decode()

    prompt = f"""You are an expert interior decorator. Analyse this room image and return ONLY valid JSON.

Room context:
- Zones: {zone_names}
- Items: {detected_items}
- Clearance: {clearance} (density: {round(density*100,1)}%)
- Conflicts: {conflict_msgs}
- Budget: {budget}

Return ONLY this JSON, no markdown, no extra text:
{{
  "room_summary": "2-sentence assessment",
  "overall_score": 65,
  "improvements": [
    {{
      "title": "action title",
      "priority": "high",
      "category": "Lighting",
      "description": "specific advice",
      "impact": "visual benefit",
      "effort": "quick win"
    }}
  ],
  "colour_palette": {{
    "primary": "#8B6F47",
    "secondary": "#6B8F71",
    "accent": "#C9962A",
    "description": "warm earthy tones"
  }},
  "style_direction": "2-sentence style recommendation",
  "quick_wins": ["win1", "win2", "win3"]
}}

Generate 5-8 improvements. Be specific to what you see."""

    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
            temperature=0.4,
            max_tokens=1500,
        )

        text = response.choices[0].message.content.strip()
        if not text:
            return {"error": "Empty response from AI — please try again."}
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except json.JSONDecodeError:
        return {"error": "We couldn't get a complete response — this usually happens when the network connection is slow or unstable. Please try again."}
    except Exception as e:
        err_msg = str(e).lower()
        if "timeout" in err_msg or "connection" in err_msg:
            return {"error": "The connection seems slow right now. Please check your internet and try again."}
        return {"error": "Something went wrong generating the plan. Please try again in a moment."}

# ════════════════════════════════════════════════════════════════════════════
#  BEFORE / AFTER VISUALISER
# ════════════════════════════════════════════════════════════════════════════

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def build_after_image(img_rgb: np.ndarray, plan: dict, zones: list) -> np.ndarray:
    """
    Create a stylised 'after' visualisation by:
    1. Slightly enhancing brightness / saturation
    2. Overlaying the recommended colour palette as subtle corner swatches
    3. Annotating top improvements as badges
    4. Adding a clean overlay frame
    """
    pil = Image.fromarray(img_rgb).convert("RGBA")
    W, H = pil.size

    # --- Step 1: enhance the image to look 'better' ---
    enhancer = ImageEnhance.Brightness(pil)
    pil = enhancer.enhance(1.08)
    enhancer = ImageEnhance.Color(pil)
    pil = enhancer.enhance(1.15)
    enhancer = ImageEnhance.Contrast(pil)
    pil = enhancer.enhance(1.05)
    enhancer = ImageEnhance.Sharpness(pil)
    pil = enhancer.enhance(1.1)

    draw = ImageDraw.Draw(pil, "RGBA")

    # --- Step 2: colour palette swatches (bottom-left corner) ---
    palette = plan.get("colour_palette", {})
    swatch_colours = [
        palette.get("primary",   "#8B6F47"),
        palette.get("secondary", "#6B8F71"),
        palette.get("accent",    "#C9962A"),
    ]
    swatch_w, swatch_h = 60, 18
    sx, sy = 12, H - swatch_h - 12
    for i, hex_col in enumerate(swatch_colours):
        try:
            rgb = hex_to_rgb(hex_col)
        except Exception:
            rgb = (180, 160, 140)
        draw.rectangle([sx + i*(swatch_w+4), sy,
                         sx + i*(swatch_w+4) + swatch_w, sy + swatch_h],
                        fill=rgb + (220,), outline=(255,255,255,180), width=1)

    # --- Step 3: improvement badges (top-right corner) ---
    improvements = plan.get("improvements", [])
    high_priority = [imp for imp in improvements if imp.get("priority") == "high"][:3]
    badge_colors = {"Lighting": (201, 150, 42), "Layout": (107, 143, 113),
                    "Colour": (139, 94, 168), "Declutter": (184, 92, 56),
                    "Texture": (92, 104, 117), "Plants": (60, 170, 100),
                    "Storage": (80, 120, 180), "Art": (180, 100, 120)}
    bx = W - 12
    by = 12
    for imp in high_priority:
        cat   = imp.get("category", "Layout")
        label = f"✦ {imp.get('title', cat)[:22]}"
        col   = badge_colors.get(cat, (140, 120, 100))
        bw    = min(len(label) * 7 + 16, W // 2)
        bh    = 22
        draw.rectangle([bx - bw, by, bx, by + bh],
                        fill=col + (210,), outline=(255,255,255,150), width=1)
        # Draw text (PIL default font — small but readable)
        draw.text((bx - bw + 6, by + 4), label, fill=(255, 255, 255, 230))
        by += bh + 6

    # --- Step 4: overall score badge (top-left) ---
    score = plan.get("overall_score", 0)
    improved_score = min(score + 28, 99)
    draw.rectangle([10, 10, 90, 44], fill=(44, 44, 44, 210))
    draw.text((16, 15), f"After: {improved_score}/100", fill=(247, 244, 239, 240))

    return np.array(pil.convert("RGB"))


def build_before_image(img_rgb: np.ndarray, plan: dict) -> np.ndarray:
    """Add a 'before' score badge to the original image."""
    pil  = Image.fromarray(img_rgb).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    score = plan.get("overall_score", 0)
    draw.rectangle([10, 10, 88, 44], fill=(184, 92, 56, 210))
    draw.text((16, 15), f"Before: {score}/100", fill=(255, 255, 255, 240))
    return np.array(pil)


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏠 Room Analyser")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)
    enhance_img    = st.toggle("🔆 Auto-brighten dark images", value=True,
                               help="Uses CLAHE enhancement — helps with low-light rooms")
    st.markdown("---")
    st.markdown("### 💰 Budget Preference")
    budget_pref = st.radio(
        "For product suggestions",
        ["Budget (under ₹2,000/item)", "Mid-range (₹2,000–10,000)", "Premium (₹10,000+)"],
        index=1
    )
    st.markdown("---")
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. **Upload** a room photo or use **Webcam**
    2. **YOLOv8** detects all objects
    3. Objects are mapped to **decoration zones**
    4. Get **conflict warnings** and **decor tips**
    5. **AI** generates improvement plan + before/after
    """)
    st.markdown("---")
    st.markdown("**Model:** `yolov8n.pt`")
    st.markdown("**Zones:** 7 functional areas")


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🏠 Room Decoration Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">YOLOv8 · Zone Detection · AI Improvement Plan · Before & After · 2026</div>',
            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TABS — always visible
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 ,tab6 = st.tabs([
    "📤 Upload & Analyse",
    "📹 Webcam",
    "✨ How to Improve",
    "🔄 Before & After",
    "📋 Object List",
    "🎨 Wall Painter"
])


# ── Tab 1: Upload ─────────────────────────────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a room image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="nv-card" style="text-align:center;padding:3rem">
            <div style="font-size:3rem">📷</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.3rem;margin:0.5rem 0">
                Upload a room image to get started</div>
            <div style="font-size:0.8rem;color:#8B6F47">
                Works best with living rooms, bedrooms, kitchens, offices</div>
        </div>""", unsafe_allow_html=True)
    else:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_rgb   = np.array(pil_image)
        st.session_state["last_img_rgb"]   = img_rgb
        st.session_state["last_img_source"] = "upload"

        with st.spinner("Running YOLOv8 detection..."):
            detections = run_yolo_detection(img_rgb, conf_threshold, enhance=enhance_img)
        st.session_state["last_detections"] = detections

        # Summary
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem">
                <span class="card-label">Objects Detected</span>
                <div class="card-value">{len(detections)}</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem">
                <span class="card-label">Unique Classes</span>
                <div class="card-value">{len(set(d['label'] for d in detections))}</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem">
                <span class="card-label">Confidence</span>
                <div class="card-value">{conf_threshold:.0%}</div></div>""",
                unsafe_allow_html=True)

        # Original vs Annotated
        st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(img_rgb, use_column_width=True)
        with col_det:
            st.markdown(f"**Detected ({len(detections)} objects)**")
            if detections:
                st.image(draw_boxes(img_rgb, detections), use_column_width=True)
            else:
                st.image(img_rgb, use_column_width=True)
                st.warning("No objects detected. Lower the confidence threshold.")

        # Zone analysis
        st.markdown('<div class="section-title">🎨 Decoration Zone Analysis</div>',
                    unsafe_allow_html=True)
        render_zone_results(detections, img_rgb, prefix="upload")

        st.info("👆 Switch to the **✨ How to Improve** or **🔄 Before & After** tabs for AI-powered suggestions.")


# ── Tab 2: Webcam ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📹 Live Webcam Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="algo-box">
        <div class="algo-title">How to use webcam</div>
        <div class="algo-desc">
            Click <b>Allow</b> when your browser asks for camera permission.
            Then click the <b>camera button</b> inside the box below to take a snapshot.
            Zone detection runs automatically on the captured photo.
        </div>
    </div>""", unsafe_allow_html=True)

    cam_conf     = st.slider("Webcam Confidence", 0.1, 0.9, 0.20, 0.05, key="cam_conf")
    camera_image = st.camera_input("📸 Click below to capture your room")

    if camera_image is not None:
        cam_pil = Image.open(camera_image).convert("RGB")
        cam_rgb = np.array(cam_pil)
        st.session_state["last_img_rgb"]    = cam_rgb
        st.session_state["last_img_source"] = "webcam"

        with st.spinner("Analysing webcam frame..."):
            cam_detections = run_yolo_detection(cam_rgb, cam_conf, enhance=enhance_img)
        st.session_state["last_detections"] = cam_detections

        st.markdown("---")
        col_raw, col_ann = st.columns(2)
        with col_raw:
            st.markdown("**Captured Frame**")
            st.image(cam_rgb, use_column_width=True)
        with col_ann:
            st.markdown(f"**Detected ({len(cam_detections)} objects)**")
            if cam_detections:
                st.image(draw_boxes(cam_rgb, cam_detections), use_column_width=True)
            else:
                st.image(cam_rgb, use_column_width=True)

        st.markdown('<div class="section-title">🎨 Decoration Zone Analysis</div>',
                    unsafe_allow_html=True)
        render_zone_results(cam_detections, cam_rgb, prefix="cam")
        st.info("👆 Switch to the **✨ How to Improve** or **🔄 Before & After** tabs for AI-powered suggestions.")
    else:
        st.markdown("""
        <div class="nv-card" style="text-align:center;padding:2.5rem">
            <div style="font-size:3rem">📹</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.2rem;margin:0.5rem 0">
                Allow camera access then capture your room</div>
            <div style="font-size:0.8rem;color:#8B6F47">
                Works with any room — living room, bedroom, kitchen, office</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — HOW TO IMPROVE  (NEW)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">✨ How to Make Your Room Look Better</div>',
                unsafe_allow_html=True)

    img_rgb_for_ai   = st.session_state.get("last_img_rgb")
    detections_for_ai = st.session_state.get("last_detections")

    if img_rgb_for_ai is None or detections_for_ai is None:
        st.markdown("""
        <div class="nv-card" style="text-align:center;padding:3rem">
            <div style="font-size:3rem">✨</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.2rem;margin:0.5rem 0">
                Upload or capture a room first</div>
            <div style="font-size:0.8rem;color:#8B6F47">
                Go to the Upload or Webcam tab, then come back here</div>
        </div>""", unsafe_allow_html=True)
    else:
        result_ai  = detect_room_zones(detections_for_ai, img_rgb_for_ai.shape)
        zones_ai   = result_ai["zones"]
        conflicts_ai = result_ai["conflicts"]

        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            run_ai = st.button("🤖 Generate AI Improvement Plan", use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div style="padding:0.6rem 0;font-size:0.78rem;color:#5C6875">
                Budget: <b>{budget_pref}</b> · {len(zones_ai)} zones detected ·
                {len(conflicts_ai)} conflict(s)
            </div>""", unsafe_allow_html=True)

        if run_ai or st.session_state.get("ai_plan"):
            if run_ai or "ai_plan" not in st.session_state:
                with st.spinner("🤖 Claude is analysing your room..."):
                    plan = get_ai_improvement_plan(
                        img_rgb_for_ai, zones_ai, conflicts_ai,
                        result_ai["density"], result_ai["clearance"], budget_pref
                    )
                st.session_state["ai_plan"] = plan
            else:
                plan = st.session_state["ai_plan"]

            if "error" in plan:
                st.error(f"AI analysis failed: {plan['error']}")
            else:
                # ── Room summary + score ──────────────────────────────────
                score = plan.get("overall_score", 0)
                score_color = "#B85C38" if score < 40 else "#C9962A" if score < 70 else "#6B8F71"
                st.markdown(f"""
                <div class="nv-card" style="border-left:5px solid {score_color};padding:1.2rem">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div style="flex:1">
                            <span class="card-label">AI Room Assessment</span>
                            <div style="font-size:0.85rem;color:#2C2C2C;margin-top:0.5rem;line-height:1.8">
                                {plan.get('room_summary','')}</div>
                            <div style="font-size:0.75rem;color:#8B6F47;margin-top:0.6rem;font-style:italic">
                                🎨 {plan.get('style_direction','')}</div>
                        </div>
                        <div style="text-align:center;min-width:80px;margin-left:1rem">
                            <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                                        color:{score_color};font-weight:600">{score}</div>
                            <div style="font-size:0.6rem;text-transform:uppercase;
                                        letter-spacing:1px;color:#8B6F47">Current Score</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Quick wins ────────────────────────────────────────────
                quick_wins = plan.get("quick_wins", [])
                if quick_wins:
                    st.markdown('<div class="section-title">⚡ Quick Wins (Do Today)</div>',
                                unsafe_allow_html=True)
                    cols_qw = st.columns(min(len(quick_wins), 3))
                    for i, win in enumerate(quick_wins[:3]):
                        with cols_qw[i]:
                            st.markdown(f"""
                            <div class="nv-card" style="text-align:center;padding:1rem;
                                         border-top:3px solid #6B8F71">
                                <div style="font-size:1.4rem">✅</div>
                                <div style="font-size:0.78rem;color:#2C2C2C;
                                            line-height:1.6;margin-top:0.4rem">{win}</div>
                            </div>""", unsafe_allow_html=True)

                # ── Colour palette ────────────────────────────────────────
                palette = plan.get("colour_palette", {})
                if palette:
                    st.markdown('<div class="section-title">🎨 Recommended Colour Palette</div>',
                                unsafe_allow_html=True)
                    pal_cols = st.columns(4)
                    pal_entries = [
                        ("Primary",   palette.get("primary",   "#8B6F47")),
                        ("Secondary", palette.get("secondary", "#6B8F71")),
                        ("Accent",    palette.get("accent",    "#C9962A")),
                    ]
                    for i, (label, hex_val) in enumerate(pal_entries):
                        with pal_cols[i]:
                            st.markdown(f"""
                            <div style="background:{hex_val};height:60px;border-radius:10px;
                                        margin-bottom:6px;box-shadow:0 2px 6px rgba(0,0,0,0.1)"></div>
                            <div style="font-size:0.65rem;text-transform:uppercase;
                                        letter-spacing:1px;color:#8B6F47;font-weight:600">{label}</div>
                            <div style="font-size:0.72rem;color:#2C2C2C;font-weight:500">{hex_val}</div>
                            """, unsafe_allow_html=True)
                    with pal_cols[3]:
                        st.markdown(f"""
                        <div style="padding-top:0.3rem">
                            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;
                                        color:#8B6F47;font-weight:600;margin-bottom:0.3rem">Mood</div>
                            <div style="font-size:0.75rem;color:#5C6875;line-height:1.6">
                                {palette.get('description','')}</div>
                        </div>""", unsafe_allow_html=True)

                # ── Improvement cards ─────────────────────────────────────
                st.markdown('<div class="section-title">🛠️ Improvement Plan</div>',
                            unsafe_allow_html=True)
                improvements = plan.get("improvements", [])
                priority_order = {"high": 0, "medium": 1, "low": 2}
                improvements.sort(key=lambda x: priority_order.get(x.get("priority","low"), 2))

                for imp in improvements:
                    pri   = imp.get("priority", "low")
                    cat   = imp.get("category", "Layout")
                    effort = imp.get("effort", "weekend project")
                    effort_icons = {"quick win": "⚡", "weekend project": "🔨", "major upgrade": "🏗️"}
                    st.markdown(f"""
                    <div class="improvement-card priority-{pri}">
                        <span class="priority-badge badge-{pri}">{pri.upper()} PRIORITY</span>
                        <span style="font-size:0.6rem;color:#8B6F47;margin-left:8px;
                                     text-transform:uppercase;letter-spacing:1px">
                            {effort_icons.get(effort,'🔨')} {effort}</span>
                        <div style="font-family:'Playfair Display',serif;font-size:1rem;
                                    color:#2C2C2C;margin:0.3rem 0 0.5rem">
                            [{cat}] {imp.get('title','')}</div>
                        <div style="font-size:0.8rem;color:#5C6875;line-height:1.7;margin-bottom:0.5rem">
                            {imp.get('description','')}</div>
                        <div style="font-size:0.72rem;color:#6B8F71;font-weight:500">
                            ✦ Impact: {imp.get('impact','')}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Products to buy ───────────────────────────────────────
                st.markdown('<div class="section-title">🛍️ Products to Consider</div>',
                            unsafe_allow_html=True)
                st.markdown("""
                <div class="algo-box">
                    <div class="algo-title">Curated for your room's zones</div>
                    <div class="algo-desc">
                        Products are matched to the functional zones detected in your room.
                        Prices are indicative estimates for the Indian market (2026).
                    </div>
                </div>""", unsafe_allow_html=True)

                detected_zone_names = [z.name for z in zones_ai]

                for zone_name in detected_zone_names:
                    zone_products = ZONE_PRODUCTS.get(zone_name, {})
                    if not zone_products:
                        continue
                    zdef = ZONE_DEFINITIONS[zone_name]
                    st.markdown(f"""
                    <div style="margin:1rem 0 0.5rem">
                        <span style="font-family:'Playfair Display',serif;font-size:1.05rem;
                                     color:{zdef['color_hex']}">{zdef['icon']} {zone_name}</span>
                    </div>""", unsafe_allow_html=True)

                    for category, products in zone_products.items():
                        st.markdown(f'<div class="product-category">▸ {category}</div>',
                                    unsafe_allow_html=True)
                        prod_cols = st.columns(min(len(products), 3))
                        for i, prod in enumerate(products):
                            with prod_cols[i % 3]:
                                st.markdown(f"""
                                <div class="nv-card" style="padding:1rem;min-height:130px">
                                    <div style="font-weight:600;font-size:0.82rem;
                                                color:#2C2C2C;margin-bottom:0.3rem">
                                        {prod['name']}</div>
                                    <div class="budget-tag">{prod['price']}</div>
                                    <div style="font-size:0.72rem;color:#5C6875;
                                                line-height:1.6;margin-top:0.4rem">
                                        {prod['why']}</div>
                                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — BEFORE & AFTER  (NEW)
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🔄 Before & After Visualisation</div>',
                unsafe_allow_html=True)

    img_rgb_ba    = st.session_state.get("last_img_rgb")
    dets_ba       = st.session_state.get("last_detections")
    ai_plan_ba    = st.session_state.get("ai_plan")

    if img_rgb_ba is None:
        st.markdown("""
        <div class="nv-card" style="text-align:center;padding:3rem">
            <div style="font-size:3rem">🔄</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.2rem;margin:0.5rem 0">
                Upload or capture a room first</div>
            <div style="font-size:0.8rem;color:#8B6F47">
                Go to Upload or Webcam tab → then visit ✨ How to Improve → then come back here</div>
        </div>""", unsafe_allow_html=True)
    elif ai_plan_ba is None:
        st.info("💡 First run the AI plan in the **✨ How to Improve** tab, then the before/after will appear here.")
        # Still show a basic before/after with what we have
        if dets_ba:
            result_ba = detect_room_zones(dets_ba, img_rgb_ba.shape)
            basic_plan = {
                "overall_score": 55,
                "colour_palette": {
                    "primary": "#8B6F47",
                    "secondary": "#6B8F71",
                    "accent": "#C9962A",
                },
                "improvements": []
            }
            st.markdown('<div class="section-title">Quick Preview (run AI for full analysis)</div>',
                        unsafe_allow_html=True)
            col_b, col_a = st.columns(2)
            with col_b:
                st.markdown('<div class="before-after-label label-before">BEFORE</div>',
                            unsafe_allow_html=True)
                st.image(img_rgb_ba, use_column_width=True)
            with col_a:
                st.markdown('<div class="before-after-label label-after">AFTER (preview)</div>',
                            unsafe_allow_html=True)
                after_img = build_after_image(img_rgb_ba, basic_plan, result_ba["zones"])
                st.image(after_img, use_column_width=True)
    else:
        plan_ba    = ai_plan_ba
        result_ba  = detect_room_zones(dets_ba, img_rgb_ba.shape) if dets_ba else {"zones": []}
        before_img = build_before_image(img_rgb_ba, plan_ba)
        after_img  = build_after_image(img_rgb_ba, plan_ba, result_ba.get("zones", []))

        # ── Score improvement banner ──────────────────────────────────────
        before_score = plan_ba.get("overall_score", 55)
        after_score  = min(before_score + 28, 99)
        delta        = after_score - before_score

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem;
                             border-top:4px solid #B85C38">
                <span class="card-label">Before Score</span>
                <div class="card-value" style="color:#B85C38">{before_score}/100</div>
                </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem;
                             border-top:4px solid #6B8F71">
                <span class="card-label">After Score (projected)</span>
                <div class="card-value" style="color:#6B8F71">{after_score}/100</div>
                </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""<div class="nv-card" style="text-align:center;padding:1rem;
                             border-top:4px solid #C9962A">
                <span class="card-label">Improvement</span>
                <div class="card-value" style="color:#C9962A">+{delta} pts</div>
                </div>""", unsafe_allow_html=True)

        # ── Side-by-side images ───────────────────────────────────────────
        col_b, col_a = st.columns(2, gap="medium")
        with col_b:
            st.markdown('<div class="before-after-label label-before">BEFORE — Current State</div>',
                        unsafe_allow_html=True)
            st.image(before_img, use_column_width=True)
        with col_a:
            st.markdown('<div class="before-after-label label-after">AFTER — With Improvements</div>',
                        unsafe_allow_html=True)
            st.image(after_img, use_column_width=True,
                     caption="Enhanced brightness, saturation & contrast · Colour palette applied · Top improvements highlighted")

        # ── What changed panel ────────────────────────────────────────────
        st.markdown('<div class="section-title">What Changes in the After Image</div>',
                    unsafe_allow_html=True)
        change_c1, change_c2 = st.columns(2)
        with change_c1:
            palette = plan_ba.get("colour_palette", {})
            st.markdown(f"""
            <div class="nv-card">
                <span class="card-label">Visual Enhancements Applied</span>
                <div style="font-size:0.78rem;color:#5C6875;line-height:2;margin-top:0.5rem">
                    • +8% brightness — lifts the room's perceived size<br>
                    • +15% colour saturation — richer, more inviting tones<br>
                    • +5% contrast — crisper furniture edges<br>
                    • +10% sharpness — cleaner detail definition<br>
                    • Palette swatches overlaid: {palette.get('description','')}
                </div>
            </div>""", unsafe_allow_html=True)
        with change_c2:
            high_imps = [i for i in plan_ba.get("improvements",[]) if i.get("priority")=="high"][:3]
            imp_html = "".join(
                f"• <b>{imp.get('title','')}</b> — {imp.get('impact','')}<br>"
                for imp in high_imps
            )
            st.markdown(f"""
            <div class="nv-card">
                <span class="card-label">Top Improvements Highlighted</span>
                <div style="font-size:0.78rem;color:#5C6875;line-height:2;margin-top:0.5rem">
                    {imp_html if imp_html else "Run AI plan for detailed highlights."}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Download button ───────────────────────────────────────────────
        st.markdown('<div class="section-title">💾 Download</div>', unsafe_allow_html=True)
        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            buf_b = io.BytesIO()
            Image.fromarray(before_img).save(buf_b, format="JPEG", quality=92)
            st.download_button(
                "⬇️ Download Before Image",
                data=buf_b.getvalue(),
                file_name="room_before.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
        with dl_c2:
            buf_a = io.BytesIO()
            Image.fromarray(after_img).save(buf_a, format="JPEG", quality=92)
            st.download_button(
                "⬇️ Download After Image",
                data=buf_a.getvalue(),
                file_name="room_after.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )


# ── Tab 5: Object List ────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Detected Objects</div>', unsafe_allow_html=True)
    dets_list = st.session_state.get("last_detections")
    if dets_list:
        label_counts = Counter(d["label"] for d in dets_list)
        df = pd.DataFrame(
            [(lbl, cnt, f"{max(d['conf'] for d in dets_list if d['label']==lbl):.0%}")
             for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])],
            columns=["Object", "Count", "Best Confidence"]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Upload an image or capture a webcam photo first, then come back here to see the object list.")

# ════════════════════════════════════════════════════════════════════════════
#  WALL COLOUR PAINTER — helper functions
# ════════════════════════════════════════════════════════════════════════════

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def recolor_wall(img_rgb: np.ndarray, mask: np.ndarray, target_hex: str,
                  strength: float = 0.85) -> np.ndarray:
    """Repaint masked area with target_hex while preserving shading (LAB lightness)."""
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return img_rgb

    target_rgb = np.array(hex_to_rgb(target_hex), dtype=np.uint8).reshape(1, 1, 3)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    target_L, target_A, target_B = target_lab

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    masked_L = L[mask_bool]
    mean_L = masked_L.mean()
    new_L = target_L + (L - mean_L) * 0.92
    new_L = np.clip(new_L, 0, 255)

    feather = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (15, 15), 0)
    feather = np.clip(feather, 0, 1)

    out_L = L * (1 - feather) + new_L * feather
    out_A = A * (1 - feather * strength) + target_A * (feather * strength)
    out_B = B * (1 - feather * strength) + target_B * (feather * strength)

    out_lab = cv2.merge([out_L, out_A, out_B]).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)


def generate_rainbow_palette(n_hue_steps: int = 12, n_shade_rows: int = 5):
    """Build a gradient spectrum like a paint-swatch chart: hue across, lightness down."""
    import colorsys
    rows = []
    for row in range(n_shade_rows):
        lightness = 0.85 - (row / (n_shade_rows - 1)) * 0.65
        row_colors = []
        for col in range(n_hue_steps):
            hue = col / n_hue_steps
            r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.55)
            hexv = "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))
            row_colors.append(hexv)
        rows.append(row_colors)
    return rows


def polygon_to_mask(points: list, H: int, W: int) -> Optional[np.ndarray]:
    """Fill a polygon defined by clicked points into a binary mask."""
    if len(points) < 3:
        return None
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def draw_point_overlay(img_rgb: np.ndarray, points: list, selected_idx: int = None) -> np.ndarray:
    """Show clicked points connected by lines, with a translucent fill once closed."""
    overlay = img_rgb.copy()
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        fill_layer = overlay.copy()
        cv2.fillPoly(fill_layer, [pts], (255, 80, 80))
        overlay = cv2.addWeighted(overlay, 0.6, fill_layer, 0.4, 0)
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=(len(points) >= 3),
                       color=(255, 60, 60), thickness=2)
    for i, (x, y) in enumerate(points):
        if i == selected_idx:
            cv2.circle(overlay, (int(x), int(y)), 10, (255, 255, 0), -1)
            cv2.circle(overlay, (int(x), int(y)), 10, (0, 0, 0), 2)
        else:
            cv2.circle(overlay, (int(x), int(y)), 6, (255, 255, 255), -1)
            cv2.circle(overlay, (int(x), int(y)), 6, (200, 40, 40), 2)
    return overlay


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — WALL COLOUR PAINTER  (click-to-outline + rainbow palette)
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">🎨 Wall Colour Painter</div>', unsafe_allow_html=True)

    img_rgb_wp = st.session_state.get("last_img_rgb")

    if img_rgb_wp is None:
        st.markdown("""
        <div class="nv-card" style="text-align:center;padding:3rem">
            <div style="font-size:3rem">🎨</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.2rem;margin:0.5rem 0">
                Upload or capture a room first</div>
            <div style="font-size:0.8rem;color:#8B6F47">
                Go to the Upload or Webcam tab, then come back here to paint the walls</div>
        </div>""", unsafe_allow_html=True)
    else:
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
        except ImportError:
            st.error("❌ Run: pip install streamlit-image-coordinates")
            st.stop()

        st.markdown("""
        <div class="algo-box">
            <div class="algo-title">How to use</div>
            <div class="algo-desc">
                Click points around the wall area, one corner at a time, to outline it
                (3+ points needed). Pick a colour from the palette, then hit <b>Apply Paint</b>.
            </div>
        </div>""", unsafe_allow_html=True)

        H_wp, W_wp = img_rgb_wp.shape[:2]
        max_disp_w = 700
        scale = min(1.0, max_disp_w / W_wp)
        disp_w, disp_h = int(W_wp * scale), int(H_wp * scale)
        img_disp = cv2.resize(img_rgb_wp, (disp_w, disp_h))

        if "wp_points" not in st.session_state:
            st.session_state["wp_points"] = []
        if "wp_selected_color" not in st.session_state:
            st.session_state["wp_selected_color"] = "#8B6F47"
        if "wp_last_click" not in st.session_state:
            st.session_state["wp_last_click"] = None

        col_tools, col_img = st.columns([1, 2], gap="large")

        with col_tools:
            st.markdown("**Paint Colour**")
            palette_rows = generate_rainbow_palette(n_hue_steps=12, n_shade_rows=5)
            for row in palette_rows:
                cols = st.columns(len(row))
                for col, hexv in zip(cols, row):
                    is_sel = st.session_state["wp_selected_color"].upper() == hexv
                    border = "2px solid #2C2C2C" if is_sel else "1px solid rgba(0,0,0,0.08)"
                    with col:
                        st.markdown(f"""<div style="background:{hexv};height:26px;
                            border:{border};cursor:pointer"></div>""", unsafe_allow_html=True)
                        if st.button(" ", key=f"sw_{hexv}", use_container_width=True):
                            st.session_state["wp_selected_color"] = hexv
                            st.rerun()

            custom_pick = st.color_picker("Custom colour", value=st.session_state["wp_selected_color"])
            if custom_pick != st.session_state["wp_selected_color"]:
                st.session_state["wp_selected_color"] = custom_pick
                st.rerun()

            custom_color = st.session_state["wp_selected_color"]
            st.markdown(f"""<div style="background:{custom_color};height:44px;border-radius:8px;
                margin-top:8px"></div><div style="font-size:0.7rem;color:#8B6F47;
                text-align:center;margin-top:4px">{custom_color}</div>""", unsafe_allow_html=True)

            strength = st.slider("Recolour strength", 0.3, 1.0, 0.85, 0.05)

            st.markdown("---")
            n_pts = len(st.session_state["wp_points"])
            st.markdown(f"**{n_pts} point(s) placed**" + (" — ready to paint" if n_pts >= 3 else ""))
            apply_paint = st.button("🖌️ Apply Paint", use_container_width=True)
            undo_point  = st.button("↶ Undo Last Point", use_container_width=True)
            clear_points = st.button("🧹 Clear Outline", use_container_width=True)
            clear_paint = st.button("↩️ Reset to Original", use_container_width=True)

        if undo_point and st.session_state["wp_points"]:
            st.session_state["wp_points"].pop()
            st.session_state["wp_dragging_idx"] = None
            st.rerun()
        if clear_points:
            st.session_state["wp_points"] = []
            st.session_state["wp_dragging_idx"] = None
            st.rerun()
        if clear_paint:
            st.session_state.pop("painted_wall_img", None)
            st.rerun()

        with col_img:
            st.markdown("**Click to add points · click an existing point to select, then click to move it**")
            if "wp_dragging_idx" not in st.session_state:
                st.session_state["wp_dragging_idx"] = None

            preview = draw_point_overlay(img_disp, st.session_state["wp_points"],
                                          selected_idx=st.session_state["wp_dragging_idx"])
            click = streamlit_image_coordinates(Image.fromarray(preview), key="wp_click")

            if click is not None:
                pt = (click["x"], click["y"])
                if pt != st.session_state["wp_last_click"]:
                    st.session_state["wp_last_click"] = pt

                    if st.session_state["wp_dragging_idx"] is not None:
                        # Move the selected point to the new click location
                        idx = st.session_state["wp_dragging_idx"]
                        if 0 <= idx < len(st.session_state["wp_points"]):
                            st.session_state["wp_points"][idx] = pt
                        st.session_state["wp_dragging_idx"] = None
                    else:
                        # Check if click is near an existing point (within 15px) → select it
                        click_radius = 15
                        nearest_idx = None
                        for i, (px, py) in enumerate(st.session_state["wp_points"]):
                            if abs(px - pt[0]) <= click_radius and abs(py - pt[1]) <= click_radius:
                                nearest_idx = i
                                break
                        if nearest_idx is not None:
                            st.session_state["wp_dragging_idx"] = nearest_idx
                        else:
                            st.session_state["wp_points"].append(pt)
                    st.rerun()

            if st.session_state["wp_dragging_idx"] is not None:
                st.info(f"📍 Point {st.session_state['wp_dragging_idx']+1} selected — click anywhere to move it there.")

        if apply_paint:
            if len(st.session_state["wp_points"]) < 3:
                st.warning("Click at least 3 points to outline an area first.")
            else:
                scale_x = W_wp / disp_w
                scale_y = H_wp / disp_h
                full_points = [(x * scale_x, y * scale_y) for (x, y) in st.session_state["wp_points"]]
                mask_full = polygon_to_mask(full_points, H_wp, W_wp)
                with st.spinner("Repainting wall..."):
                    painted = recolor_wall(img_rgb_wp, mask_full, custom_color, strength)
                st.session_state["painted_wall_img"] = painted
                st.session_state["painted_wall_color"] = custom_color

        if "painted_wall_img" in st.session_state:
            st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
            col_before, col_after = st.columns(2, gap="medium")
            with col_before:
                st.markdown('<div class="before-after-label label-before">ORIGINAL</div>',
                            unsafe_allow_html=True)
                st.image(img_rgb_wp, use_column_width=True)
            with col_after:
                st.markdown('<div class="before-after-label label-after">REPAINTED</div>',
                            unsafe_allow_html=True)
                st.image(st.session_state["painted_wall_img"], use_column_width=True,
                         caption=f"Wall colour: {st.session_state.get('painted_wall_color','')}")

            buf_wp = io.BytesIO()
            Image.fromarray(st.session_state["painted_wall_img"]).save(buf_wp, format="JPEG", quality=92)
            st.download_button(
                "⬇️ Download Repainted Image",
                data=buf_wp.getvalue(),
                file_name="room_repainted.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )