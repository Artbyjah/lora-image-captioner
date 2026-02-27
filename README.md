# LoRA Toolkit

A local web application for building LoRA training datasets — AI-powered captioning, video frame extraction, character sheet generation, and dataset export with multi-provider support.

## Features

### Captioning
- **Multi-Provider AI Captioning** — Generate captions using Claude (Anthropic), Gemini (Google), GPT-4o (OpenAI), or Ollama (local). The app auto-detects which providers are available.
- **5 LoRA Types** — Character, Style, Object, Concept, and Motion — each with type-specific captioning instructions that tell the AI what to describe and what to leave for the trigger word.
- **Concise / Detailed Mode** — Toggle between full descriptive captions or terse ~40 token tag-only output.
- **Customizable Templates** — Configure trigger words, toggle caption elements (subject, expression, pose, action, shot type, angle, lighting, background), set style tags, and list special items.
- **Template Presets** — Save and load your own presets, or use built-ins (Anime Character, Realistic Portrait, Art Style).

### Dataset Tools
- **Caption Quality Scoring** — AI-scores each image+caption pair (1-10) for accuracy, conciseness, tag coverage, and training suitability. Badges appear on cards color-coded green/yellow/red.
- **Reference Image Matching** — Upload a reference image and sort your gallery by visual similarity (1-10). Useful for curating consistent datasets.
- **Caption Diversity Check** — Flags captions that are too similar to each other using Jaccard similarity.
- **Duplicate Image Detection** — Perceptual hashing to flag near-identical images.
- **Find & Replace** — Search and replace text across all captions at once.
- **Prefix / Suffix** — Batch-add text before or after all captions.
- **+Trigger** — Prepend your trigger word to any captions missing it.
- **Resolution Crop Overlay** — Visualize the training crop area on each image at your target resolution.

### Video Frame Extraction
- **Drag & drop video files** (MP4, MOV, AVI, WebM, MKV) to extract frames.
- **Two extraction modes**: fixed interval (every N seconds) or total count (N evenly spaced frames).
- **AI Frame Selection** — Claude Vision scores each frame for LoRA training suitability and recommends the best subset.
- **Manual curation** — Click to select/deselect individual frames before adding to your gallery.

### Character Sheet Generation
- **Gemini-powered** — Upload a reference image and generate turnaround sheets, expression sheets, or action/pose sheets.
- **Grouped or Singular mode** — Generate all views in one sheet or one image per view.
- **Multiple styles** — Anime, Realistic, Cartoon, Pixel Art, 3D Render, Concept Art.
- **Custom prompts** — Write your own generation prompt for full control.

### Export
- **ai-toolkit format** — Export your dataset as a folder or downloadable ZIP with images, caption `.txt` files, and a ready-to-use `config.yaml`.
- **Pre-export validation** — Checks for missing captions, empty captions, missing trigger words, token count range, and duplicates. Errors block export; warnings are informational.
- **Configurable training params** — Base model (Flux Dev/Schnell, SDXL, SD 1.5), LoRA rank, learning rate, steps, batch size, resolution, caption dropout, gradient accumulation, and sample prompts.

### UI / UX
- **Dark / Light theme** with full theming across all components.
- **Grid / List view toggle** for the image gallery.
- **Drag-and-drop reordering** of images in the gallery.
- **Per-card actions** — Copy, Regen, and Undo on every image card.
- **Caption counters** — Word and token counts on every card with warnings when over limits.
- **Filter chips** — Filter gallery by status: All, Pending, Captioned, Edited, Error.
- **Stats bar** — Total images, captioned %, trigger word %, average/min/max token counts.
- **Auto-save** — Session state saves to localStorage every 30 seconds with restore-on-reload.
- **Keyboard shortcuts** — Ctrl+Enter to generate, Ctrl+S to save, Tab to navigate captions, ? for help.
- **Button tooltips** — Hover any action bar button for a plain-English description of what it does.

## Setup

1. **Get API Keys**
   - **Required**: [Anthropic API key](https://console.anthropic.com/) for Claude captioning
   - **Optional**: [Google API key](https://aistudio.google.com/apikey) for Gemini character sheets + captioning
   - **Optional**: [OpenAI API key](https://platform.openai.com/api-keys) for GPT-4o captioning
   - **Optional**: [Ollama](https://ollama.com/) running locally for offline captioning

2. **Configure API Keys**
   - Copy `.env.example` to `.env`
   - Add your keys:
     ```
     ANTHROPIC_API_KEY=sk-ant-xxxxx
     GOOGLE_API_KEY=xxxxx
     OPENAI_API_KEY=sk-xxxxx
     ```

3. **Run the Application**
   - Double-click `run.bat`
   - Or manually:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
     python app.py
     ```

4. **Open in Browser**
   - Go to http://localhost:5000

## Caption Format

Captions are generated as comma-separated tags:

```
trigger_word, subject, expression, pose, action, shot_type, camera_angle, lighting, [special_items], background, style_tag
```

Example:
```
Kamina, grinning, dynamic fighting stance, riding a mech, full body shot, front view, soft even lighting, cape, white neutral background, anime style
```

## Requirements

- Python 3.8+
- At least one API key (Anthropic recommended)
- Internet connection for cloud providers (Ollama works offline)

## Cost Estimation

Each image caption uses approximately:
- **Claude**: ~1500 input tokens + ~50 output tokens per image
- **Gemini**: Similar token usage, pricing varies
- **GPT-4o-mini**: Similar, generally lower cost
- **Ollama**: Free (runs locally)

For 50 images with Claude, expect roughly $0.50-1.00 in API costs. Scoring and reference matching use additional API calls.
