# 🪑 AR Furniture Assembly Assistant

> Open-vocabulary object detection + LLM reasoning pipeline that detects IKEA furniture parts in a scene and generates **personalised, context-aware assembly instructions** — structured as Unity-ready AR JSON output.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![GroundingDINO](https://img.shields.io/badge/GroundingDINO-SwinT%2FSwinB-orange)
![Llama](https://img.shields.io/badge/Llama_3.2-3B-purple)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![Unity](https://img.shields.io/badge/Output-Unity_AR_Ready-green)
![Colab](https://img.shields.io/badge/Environment-Google_Colab-F9AB00?logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

Traditional assembly manuals show every step regardless of your current state. This system does the opposite — it **looks at what's in front of you** and tells you exactly what to do next.

The pipeline takes a photo of your workspace, detects which furniture parts and tools are present using **GroundingDINO** (open-vocabulary detection — no custom training required), then feeds the results into **Llama 3.2 3B** running locally via Ollama to generate the next assembly step personalised to the current scene. The output is structured JSON designed to drive an **AR overlay in Unity**.

---

## 🎯 Key Features

- **Zero-shot part detection** — no custom training needed; GroundingDINO detects parts from natural language prompts
- **Dual-threshold detection** — separate confidence thresholds for large parts (frame, legs) vs small parts (screws, Allen key, dowels)
- **Missing parts detection** — automatically warns if required tools or components are not visible
- **LLM-guided next step** — Llama 3.2 3B determines the next logical assembly step based on what's detected
- **Unity AR JSON output** — structured output with normalised bounding boxes, highlight commands, arrows, voice text, and warnings

---

## 🏗️ System Architecture

```
Input Image (workspace photo)
         │
         ▼
┌────────────────────────────────────┐
│        GroundingDINO Detection     │
│                                    │
│  Stage 1: Large parts              │
│  • Backrest, seat frame, legs,     │
│    cushion, instructions           │
│  • box_threshold:  0.35            │
│  • text_threshold: 0.30            │
│                                    │
│  Stage 2: Small parts              │
│  • Screws, dowels, cam lock nuts,  │
│    Allen key, screwdriver, pliers  │
│  • box_threshold:  0.12            │
│  • text_threshold: 0.18            │
│                                    │
│  → NMS on large parts only         │
│  → Merge large + small             │
│  → Smart confidence filtering      │
│  → Clean label mapping             │
└──────────────┬─────────────────────┘
               │
               ▼
    detected_objects list
    [name, confidence, bbox_normalized]
               │
               ▼
┌──────────────────────────────────────┐
│     Missing Parts Check              │
│  Required: screws, cam lock nuts,    │
│  wooden dowels, Allen key,           │
│  screwdriver                         │
│  → Generates warning if any missing  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   Llama 3.2 3B (text-only)           │
│   via Ollama — runs in Colab         │
│                                      │
│  Input:                              │
│  • Full IKEA assembly instructions   │
│  • Detected parts summary (text)     │
│  • Missing parts warning             │
│                                      │
│  Note: LLM receives detected parts   │
│  as text — not the raw image.        │
│  GroundingDINO acts as the "eyes".   │
│                                      │
│  Output:                             │
│  • Next logical assembly step        │
│  • AR highlight/arrow commands       │
│  • Voice narration text              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│        Unity-Ready JSON Output       │
│  {                                   │
│    step_text,                        │
│    voice_text,                       │
│    ar_commands: {                    │
│      highlight, arrow,               │
│      outline_color,                  │
│      show_text_bubble                │
│    },                                │
│    warning,                          │
│    detected_objects: [...]           │
│  }                                   │
└──────────────────────────────────────┘
```

---

## 📋 Sample Output

```json
{
  "step_text": "Insert wooden dowels into the front legs",
  "voice_text": "Great! I can see the seat frame and wooden dowels. Your next step is to insert 2 wooden dowels into each front leg before attaching them to the seat frame.",
  "ar_commands": {
    "highlight": ["wooden dowels", "chair legs"],
    "arrow": {"from": "wooden dowels", "to": "seat frame"},
    "outline_color": "#00ff00",
    "show_text_bubble": true
  },
  "warning": "",
  "detected_objects": [
    {"name": "seat frame", "confidence": 0.821, "bbox_normalized": [0.45, 0.38, 0.52, 0.41]},
    {"name": "wooden dowels", "confidence": 0.634, "bbox_normalized": [0.12, 0.67, 0.08, 0.06]},
    {"name": "chair legs", "confidence": 0.578, "bbox_normalized": [0.78, 0.45, 0.18, 0.32]}
  ]
}
```

See `sample_output/ikea_assembly_instruction.json` for a full example.

---

## 📁 Repository Structure

```
ar-furniture-assembly-assistant/
├── notebook/
│   └── ikea_ar_assistant.ipynb         # Full self-contained pipeline (run this)
├── sample_output/
│   └── ikea_assembly_instruction.json  # Example JSON output
├── requirements.txt                    # Dependency reference (installed by notebook)
└── README.md
```

---

## 🚀 Getting Started

**This project runs entirely on Google Colab. No local setup is required.**

### Step 1 — Open the notebook

Open `notebook/ikea_ar_assistant.ipynb` in [Google Colab](https://colab.research.google.com) with a **GPU runtime** (Runtime → Change runtime type → T4 GPU).

### Step 2 — Run Cell 1 (Installation)

Cell 1 installs everything automatically:
- PyTorch (CUDA 11.8)
- GroundingDINO (Roboflow fork)
- Ollama + Llama 3.2 3B
- Downloads GroundingDINO Swin-T weights

> ⏱ This takes 3–5 minutes on first run.

### Step 3 — Run Cell 2 (Load Model)

Loads GroundingDINO. Uses Swin-B if available, falls back to Swin-T automatically.

### Step 4 — Run Cell 3 (Upload Image)

Upload a photo of your IKEA assembly workspace when prompted.

### Step 5 — Run Cell 4 (Detection)

Runs GroundingDINO on your image using dual-threshold detection for large and small parts. Displays annotated image with bounding boxes and prints all detected objects with confidence scores.

### Step 6 — Run Cell 5 (Generate Instructions)

Builds the prompt from detected objects, calls Llama 3.2 3B via Ollama, and generates the Unity-ready JSON. The file `ikea_assembly_instruction.json` is automatically downloaded.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | GroundingDINO (Swin-T / Swin-B backbone) |
| Detection library | `groundingdino-py` (Roboflow fork) |
| LLM | Llama 3.2 3B (local, via Ollama) |
| Post-processing | NMS via `torchvision.ops`, custom confidence filtering |
| Annotation | `supervision` library |
| Target platform | Unity AR (JSON output format) |
| Environment | Google Colab (T4 GPU) |

---

## 💡 Design Decisions

**Why GroundingDINO instead of a trained YOLO model?**
Open-vocabulary detection means the system works on any furniture brand or product without retraining — just change the text prompt.

**Why separate confidence thresholds for large vs small parts?**
Small parts like screws and cam lock nuts have much lower detection confidence by nature. A unified threshold either misses small parts or creates false positives on large ones. Dual thresholds solves this cleanly.

**Why a local LLM (Ollama) instead of GPT-4?**
Keeps the pipeline fully free to run and offline-capable, which matters for a real-world AR deployment on a device.

**Why Unity-structured JSON?**
The `ar_commands` block (highlight, arrow, outline_color) maps directly to Unity AR Foundation components — designed to be consumed by a Unity C# script with minimal parsing.

---

## 🔭 Future Work

- [ ] **VLM upgrade** — replace text-only Llama 3.2 3B with `llama3.2-vision` so the LLM receives the actual image for richer scene understanding
- [ ] Generalise to any furniture brand (prompt engineering)
- [ ] Replace static instruction set with PDF/manual parser
- [ ] Add step tracking — remember which steps are already completed
- [ ] Unity AR app implementation consuming the JSON output
- [ ] Real-time video mode (frame-by-frame detection)

---

## 👤 Author

**Binu Shefield Shifani**
Software Engineer (5 years, Cognizant Technology Solutions)
MS AI & Automation · University West, Trollhättan, Sweden

[![GitHub](https://img.shields.io/badge/GitHub-BinuShefieldShifani-black?logo=github)](https://github.com/BinuShefieldShifani)

---

## 📄 License

MIT License — free to use for research and personal projects.
