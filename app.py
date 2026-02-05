import os
import base64
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image
import io

# Load .env from the same directory as this script
APP_DIR = Path(__file__).parent.resolve()
env_path = APP_DIR / '.env'
load_dotenv(env_path)

# Debug: print if key is loaded
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    # Try reading .env manually as fallback
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('ANTHROPIC_API_KEY='):
                    api_key = line.strip().split('=', 1)[1]
                    break

print(f"API Key loaded: {'Yes' if api_key else 'No'}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

UPLOAD_FOLDER = APP_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

client = Anthropic(api_key=api_key)

DEFAULT_TEMPLATE = {
    "trigger_word": "character_name",
    "include_expression": True,
    "include_pose": True,
    "include_shot_type": True,
    "include_camera_angle": True,
    "include_lighting": True,
    "include_background": True,
    "style_tag": "anime style",
    "special_items": ["red glasses", "cape"]
}


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path):
    """Detect actual media type from file content using PIL."""
    try:
        with Image.open(image_path) as img:
            format_to_media = {
                'JPEG': 'image/jpeg',
                'PNG': 'image/png',
                'GIF': 'image/gif',
                'WEBP': 'image/webp'
            }
            return format_to_media.get(img.format, 'image/jpeg')
    except Exception:
        # Fallback to extension-based detection
        ext = Path(image_path).suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(ext, 'image/jpeg')


def generate_caption(image_path, template):
    """Generate caption for an image using Claude Vision."""
    base64_image = encode_image(image_path)
    media_type = get_image_media_type(image_path)

    # Build the prompt based on template settings
    prompt_parts = [f"Analyze this image and generate a caption for LoRA training."]
    prompt_parts.append(f"\nThe caption must be comma-separated tags in this exact order:")
    prompt_parts.append(f"\n1. Trigger word: \"{template['trigger_word']}\"")

    order = 2
    if template.get('include_expression'):
        prompt_parts.append(f"\n{order}. Expression (e.g., grinning, serious, confident, determined, shouting, relaxed, pensive)")
        order += 1
    if template.get('include_pose'):
        prompt_parts.append(f"\n{order}. Pose description (e.g., dynamic fighting stance, casual standing pose, arms crossed, hands on hips)")
        order += 1
    if template.get('include_shot_type'):
        prompt_parts.append(f"\n{order}. Shot type (full body shot, medium shot, close-up, portrait shot)")
        order += 1
    if template.get('include_camera_angle'):
        prompt_parts.append(f"\n{order}. Camera angle (front view, side view, side profile, three-quarter view, low angle, dynamic angle)")
        order += 1
    if template.get('include_lighting'):
        prompt_parts.append(f"\n{order}. Lighting (soft even lighting, dramatic lighting, backlighting, rim lighting, dark silhouette)")
        order += 1

    if template.get('special_items'):
        items_str = ", ".join(template['special_items'])
        prompt_parts.append(f"\n{order}. Special items ONLY if clearly visible in the image: {items_str}")
        order += 1

    if template.get('include_background'):
        prompt_parts.append(f"\n{order}. Background description (e.g., white neutral background, outdoor scene, motion blur background)")
        order += 1

    if template.get('style_tag'):
        prompt_parts.append(f"\n{order}. Style tag: \"{template['style_tag']}\"")

    prompt_parts.append("\n\nIMPORTANT: Return ONLY the caption as a single line of comma-separated tags. No explanations, no quotes around the full caption.")
    prompt_parts.append("\nExample output: character_name, grinning, dynamic fighting stance, full body shot, front view, soft even lighting, white neutral background, anime style")

    prompt = "".join(prompt_parts)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return message.content[0].text.strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle image uploads."""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    uploaded = []

    for file in files:
        if file.filename:
            # Save the file
            filename = file.filename
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)

            # Create thumbnail for preview
            try:
                with Image.open(filepath) as img:
                    img.thumbnail((200, 200))
                    thumb_buffer = io.BytesIO()
                    img.save(thumb_buffer, format='JPEG', quality=85)
                    thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            except Exception:
                thumb_base64 = None

            uploaded.append({
                'filename': filename,
                'path': str(filepath),
                'thumbnail': f"data:image/jpeg;base64,{thumb_base64}" if thumb_base64 else None
            })

    return jsonify({'uploaded': uploaded})


@app.route('/generate', methods=['POST'])
def generate_captions():
    """Generate captions for uploaded images."""
    data = request.json
    images = data.get('images', [])
    template = data.get('template', DEFAULT_TEMPLATE)

    results = []
    for img in images:
        filepath = Path(img['path'])
        if filepath.exists():
            try:
                caption = generate_caption(str(filepath), template)
                results.append({
                    'filename': img['filename'],
                    'path': img['path'],
                    'caption': caption,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'filename': img['filename'],
                    'path': img['path'],
                    'caption': '',
                    'status': 'error',
                    'error': str(e)
                })

    return jsonify({'results': results})


@app.route('/save', methods=['POST'])
def save_captions():
    """Save captions to text files."""
    data = request.json
    captions = data.get('captions', [])
    output_dir = data.get('output_dir', '')

    saved = []
    for item in captions:
        # Use original filename if provided, otherwise extract from path
        original_filename = item.get('filename', Path(item['path']).name)
        # Remove extension and add .txt
        base_name = Path(original_filename).stem

        # Determine output path
        if output_dir:
            out_path = Path(output_dir) / (base_name + '.txt')
        else:
            img_path = Path(item['path'])
            out_path = img_path.with_suffix('.txt')

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(item['caption'])
            saved.append({
                'filename': original_filename,
                'caption_file': str(out_path),
                'status': 'saved'
            })
        except Exception as e:
            saved.append({
                'filename': original_filename,
                'status': 'error',
                'error': str(e)
            })

    return jsonify({'saved': saved})


@app.route('/clear', methods=['POST'])
def clear_uploads():
    """Clear all uploaded files."""
    for file in UPLOAD_FOLDER.iterdir():
        if file.is_file():
            file.unlink()
    return jsonify({'status': 'cleared'})


@app.route('/browse-folder', methods=['POST'])
def browse_folder():
    """Open native folder picker dialog."""
    try:
        # Create a hidden Tk window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Bring dialog to front

        # Open folder picker
        folder_path = filedialog.askdirectory(
            title="Select Output Folder for Captions"
        )

        root.destroy()

        if folder_path:
            return jsonify({'folder': folder_path})
        else:
            return jsonify({'folder': ''})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("LoRA Image Captioner")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5000)
