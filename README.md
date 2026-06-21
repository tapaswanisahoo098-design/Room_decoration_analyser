# 🏠 Room Decoration Analyser

A Streamlit app that uses **YOLOv8** object detection to identify functional zones in a room photo (Living, Work, Dining, Bedroom, Kitchen, Fitness, Pet), flags decoration conflicts, and generates an **AI-powered improvement plan** — complete with a colour palette, product recommendations, a before/after visualisation, and an interactive wall-colour painter.

## ✨ Features

| Tab | What it does |
|---|---|
| 📤 **Upload & Analyse** | Upload a room photo, run YOLOv8 detection, see annotated bounding boxes |
| 📹 **Webcam** | Capture a live photo and run the same analysis in real time |
| 🎨 **Zone Detection** | Maps detected objects to 7 functional zones with confidence scores, a colour-coded heatmap, clearance/density rating, and zone-conflict warnings |
| ✨ **How to Improve** | Calls an AI vision model to generate a room assessment, score, colour palette, prioritised improvements, quick wins, and a curated shopping list |
| 🔄 **Before & After** | Auto-enhances the photo and overlays the recommended palette + top improvements as a visual "after" preview, with downloadable images |
| 📋 **Object List** | Table of every detected object, count, and best confidence |
| 🎨 **Wall Painter** | Click to outline any wall area and repaint it with a custom or palette colour while preserving shading/shadows |

## 🧰 Requirements

- Python 3.9+
- A YOLOv8 nano weights file: **`yolov8n.pt`** (place it in the same folder as the app — it is *not* included and is not a pip package; it downloads automatically the first time `ultralytics` loads it, or you can grab it manually from the [Ultralytics releases page](https://github.com/ultralytics/assets/releases))
- A **Groq API key** (free tier available) for the AI improvement plan — uses the `meta-llama/llama-4-scout-17b-16e-instruct` vision model

## 📦 Installation

```bash
# 1. Clone / copy the project files
cd room-decoration-analyser

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🔑 Configure your API key

Create a `.streamlit/secrets.toml` file in the project root:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

Get a free key at [console.groq.com](https://console.groq.com/keys).

> ⚠️ Never commit `secrets.toml` to version control — add it to `.gitignore`.

## ▶️ Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## 📁 Project structure

```
room-decoration-analyser/
├── app.py                  # Main Streamlit app
├── yolov8n.pt               # YOLOv8 nano weights (download separately)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml          # Your Groq API key (not committed)
└── README.md
```

## 🛠️ Tech stack

- **Streamlit** — UI framework
- **Ultralytics YOLOv8** — object detection
- **OpenCV** — image processing (CLAHE enhancement, heatmaps, LAB-space wall recolouring)
- **Pillow** — image overlays, badges, before/after enhancement
- **Groq (Llama 4 Scout, vision)** — AI improvement plan generation
- **streamlit-image-coordinates** — click-to-outline interaction for the Wall Painter

## 💡 Notes

- Detection confidence and auto-brighten (CLAHE) settings are adjustable from the sidebar.
- Zone scoring is rule-based (anchor / support / conflict objects per zone); the AI plan is a separate, vision-based pass run on demand to control API usage.
- Works best with well-lit photos of living rooms, bedrooms, kitchens, dining areas, and home offices.
- If `opencv-python-headless` causes issues locally (e.g. you need GUI features), swap it for `opencv-python` in `requirements.txt`.

## 📄 License

Add your preferred license here (e.g. MIT).
