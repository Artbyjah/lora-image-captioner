# LoRA Image Captioner

A local web application for batch generating image captions for LoRA/AI training datasets using Claude Vision API.

## Features

- **Drag & Drop Upload**: Easily upload batches of images
- **Claude Vision Powered**: Uses Claude's vision capabilities for accurate image analysis
- **Customizable Templates**: Configure trigger words, included elements, style tags
- **Live Preview**: See and edit captions before saving
- **Batch Processing**: Process all images with one click
- **Edit Before Save**: Review and modify any caption before saving

## Setup

1. **Get an Anthropic API Key**
   - Go to https://console.anthropic.com/
   - Create an account and generate an API key

2. **Configure the API Key**
   - Copy `.env.example` to `.env`
   - Edit `.env` and add your API key:
     ```
     ANTHROPIC_API_KEY=sk-ant-xxxxx
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

## Usage

1. **Set Template Settings** (left sidebar)
   - Enter your trigger word (e.g., "Kamina")
   - Toggle which elements to include (expression, pose, lighting, etc.)
   - Add any special items to look for
   - Set your style tag

2. **Upload Images**
   - Drag & drop images onto the upload area
   - Or click to browse and select files

3. **Generate Captions**
   - Click "Generate Captions" button
   - Wait for processing (uses Claude API)

4. **Review & Edit**
   - Check each generated caption
   - Make manual edits if needed

5. **Save**
   - Click "Save All Captions"
   - Caption .txt files are saved next to the original images

## Caption Format

Captions are generated in comma-separated format:

```
trigger_word, expression, pose, shot_type, camera_angle, lighting, [special_items], background, style_tag
```

Example:
```
Kamina, grinning, dynamic fighting stance, full body shot, front view, soft even lighting, cape, white neutral background, anime style
```

## Requirements

- Python 3.8+
- Anthropic API key
- Internet connection (for API calls)

## Cost Estimation

Claude API pricing applies. Each image analysis uses approximately:
- ~1500 input tokens (image + prompt)
- ~50 output tokens (caption)

For 50 images, expect roughly $0.50-1.00 in API costs.
