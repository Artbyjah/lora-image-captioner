import os
import base64
import json
import shutil
import subprocess
from pathlib import Path
import zipfile
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image
import io
import cv2
from google import genai
from google.genai import types

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
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max (videos can be large, deleted after extraction)

UPLOAD_FOLDER = APP_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}

client = Anthropic(api_key=api_key)

# Google Gemini client for character sheet generation
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key and env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('GOOGLE_API_KEY='):
                google_api_key = line.strip().split('=', 1)[1]
                break

gemini_client = None
if google_api_key:
    try:
        gemini_client = genai.Client(api_key=google_api_key)
        print(f"Google API Key loaded: Yes")
    except Exception as e:
        print(f"Google API Key error: {e}")
else:
    print("Google API Key loaded: No (character sheet generation disabled)")


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum upload size is 500MB.'}), 413

DEFAULT_TEMPLATE = {
    "trigger_word": "character_name",
    "include_subject": True,
    "include_expression": True,
    "include_pose": True,
    "include_action": True,
    "include_shot_type": True,
    "include_camera_angle": True,
    "include_lighting": True,
    "include_background": True,
    "style_tag": "anime style",
    "special_items": ["red glasses", "cape"]
}


MAX_IMAGE_BYTES = 4_500_000  # Stay under Claude's 5MB limit


def encode_image(image_path):
    """Encode image to base64, compressing if over the API size limit.
    Returns (base64_string, media_type)."""
    media_type = get_image_media_type(image_path)

    with open(image_path, "rb") as f:
        raw = f.read()

    # If already under the limit, return as-is
    if len(raw) <= MAX_IMAGE_BYTES:
        return base64.standard_b64encode(raw).decode("utf-8"), media_type

    # Compress: convert to JPEG and reduce quality / resolution until it fits
    print(f"  Compressing {Path(image_path).name} ({len(raw)/1_000_000:.1f}MB)...")
    img = Image.open(image_path)
    img = img.convert("RGB")  # JPEG doesn't support alpha
    compressed_type = "image/jpeg"

    for quality in (90, 80, 70, 60, 50):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), compressed_type

    # Still too large — scale down and retry
    for scale in (0.75, 0.5, 0.35):
        resized = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
        for quality in (85, 70, 55):
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=quality)
            if buf.tell() <= MAX_IMAGE_BYTES:
                return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), compressed_type

    # Last resort: aggressive resize
    resized = img.resize((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=50)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), compressed_type


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
    base64_image, media_type = encode_image(image_path)

    lora_type = template.get('lora_type', 'character')

    # Type-specific captioning instructions
    type_instructions = {
        'character': (
            "You are captioning images for character/person LoRA training. "
            "The trigger word represents the CHARACTER's identity. "
            "Describe everything EXCEPT the character's defining physical features (face, body type, identity). "
            "Describe what varies across images: pose, clothing, setting, lighting, expression. "
            "The trigger word alone captures who the character is."
        ),
        'style': (
            "You are captioning images for style LoRA training. "
            "The trigger word represents the ARTISTIC STYLE. "
            "Describe the subject matter in thorough detail — what is depicted, objects, composition, spatial relationships, actions. "
            "Do NOT describe stylistic qualities (brushstrokes, color palette, rendering style, artistic mood, medium, texture, linework). "
            "Do NOT use quality tags like 'masterpiece', 'detailed', 'high quality'. "
            "Leave all aesthetic and rendering qualities for the trigger word to capture."
        ),
        'object': (
            "You are captioning images for object LoRA training. "
            "The trigger word represents a SPECIFIC OBJECT. "
            "Pair the trigger word with a generic class noun (e.g., 'ohwx shoe', 'ohwx mug'). "
            "Describe the environment, context, lighting, camera angle, surfaces, and interactions — "
            "everything EXCEPT the object's defining visual features (shape, color, material, design details, branding, patterns). "
            "The object's appearance is captured entirely by the trigger word."
        ),
        'concept': (
            "You are captioning images for concept LoRA training. "
            "The trigger word represents an ABSTRACT PATTERN (a pose, composition, lighting effect, visual effect, scene archetype, etc.). "
            "Describe every concrete, tangible element in the image — subjects, objects, settings, clothing, colors, spatial details. "
            "Do NOT describe the concept itself (the pattern, arrangement, effect, technique, or abstract quality). "
            "The concept is captured entirely by the trigger word."
        )
    }

    # Type-specific example captions
    type_examples = {
        'character': "character_name, a young man in a red beanie and white t-shirt, relaxed, crouching on a skateboard, riding a skateboard down the street, full body shot, side view, soft even lighting, white neutral background, illustration style",
        'style': "ohwx, a woman in a white dress sitting in a garden, holding a parasol, flowers surrounding her, a stone path leading to a cottage in the background, full body shot, front view",
        'object': "a ohwx mug on a wooden kitchen counter, morning light from a window, coffee inside, steam rising, close-up, slightly angled, soft even lighting, cozy kitchen background, product photography",
        'concept': "ohwx_concept, a woman in a blue sundress, on a beach, waves in background, full body shot, low angle, golden hour, sandy shore with footprints"
    }

    # Build the prompt based on template settings
    prompt_parts = ["Analyze this image and generate a caption for LoRA training."]
    prompt_parts.append(f"\n\n{type_instructions.get(lora_type, type_instructions['character'])}")
    prompt_parts.append(f"\n\nThe caption must be comma-separated tags in this exact order:")
    prompt_parts.append(f"\n1. Trigger word: \"{template['trigger_word']}\"")

    order = 2
    if template.get('include_subject'):
        if lora_type == 'object':
            prompt_parts.append(f"\n{order}. Subject/context description - describe what surrounds the object, where it is placed, and how it's being used. Do NOT describe the object's own appearance.")
        elif lora_type == 'style':
            prompt_parts.append(f"\n{order}. Subject description - describe in thorough detail what is depicted in the image (people, objects, scenes, composition, spatial relationships). Be very specific about the content.")
        elif lora_type == 'concept':
            prompt_parts.append(f"\n{order}. Subject description - describe all concrete, tangible elements visible in the image. Be specific about subjects, objects, and their attributes.")
        else:
            prompt_parts.append(f"\n{order}. Subject description - describe what the main subject of the image is (e.g., a coiled green snake with black spots, a pink candy wrapper with text, a wooden telephone pole with street signs, a storefront with a yellow sign). Be specific and descriptive about the object, creature, or scene depicted.")
        order += 1
    if template.get('include_expression'):
        prompt_parts.append(f"\n{order}. Expression (e.g., grinning, serious, confident, determined, shouting, relaxed, pensive)")
        order += 1
    if template.get('include_pose'):
        prompt_parts.append(f"\n{order}. Pose description (e.g., dynamic fighting stance, casual standing pose, arms crossed, hands on hips)")
        order += 1
    if template.get('include_action'):
        prompt_parts.append(f"\n{order}. Action - describe what the subject is doing (e.g., riding a skateboard, sitting in a car, climbing stairs, playing guitar, walking down the street, eating tacos). Focus on the activity or scene taking place.")
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
    prompt_parts.append(f"\nExample output: {type_examples.get(lora_type, type_examples['character'])}")

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
    """Open native folder picker dialog using PowerShell in STA mode."""
    try:
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.Form; "
            "$f.TopMost = $true; $f.Visible = $false; "
            "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dlg.Description = 'Select Output Folder for Captions'; "
            "$dlg.ShowNewFolderButton = $true; "
            "if ($dlg.ShowDialog($f) -eq 'OK') { Write-Output $dlg.SelectedPath }; "
            "$f.Dispose()"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", ps_script],
            capture_output=True, text=True, timeout=120
        )
        folder_path = result.stdout.strip()

        if folder_path:
            return jsonify({'folder': folder_path})
        else:
            return jsonify({'folder': ''})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload-video', methods=['POST'])
def upload_video():
    """Handle video upload and frame extraction."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    video_file = request.files['video']
    if not video_file.filename:
        return jsonify({'error': 'No video selected'}), 400

    ext = Path(video_file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    mode = request.form.get('mode', 'total')       # 'fps' or 'total'
    value = float(request.form.get('value', '10'))  # interval in seconds or total count

    # Save video temporarily
    video_stem = Path(video_file.filename).stem
    video_path = UPLOAD_FOLDER / video_file.filename
    video_file.save(video_path)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Determine which frame indices to extract
        if mode == 'fps':
            # value = interval in seconds between frames
            interval_frames = int(fps * value)
            if interval_frames < 1:
                interval_frames = 1
            frame_indices = list(range(0, total_frames, interval_frames))
        else:
            # 'total' mode: extract exactly N frames evenly spaced
            count = max(1, int(value))
            if count >= total_frames:
                frame_indices = list(range(total_frames))
            elif count == 1:
                frame_indices = [0]
            else:
                frame_indices = [int(i * (total_frames - 1) / (count - 1)) for i in range(count)]

        extracted = []
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_filename = f"{video_stem}_frame_{idx + 1:04d}.jpg"
            frame_path = UPLOAD_FOLDER / frame_filename
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Create thumbnail (same pattern as /upload)
            try:
                with Image.open(frame_path) as img:
                    img.thumbnail((200, 200))
                    thumb_buffer = io.BytesIO()
                    img.save(thumb_buffer, format='JPEG', quality=85)
                    thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            except Exception:
                thumb_base64 = None

            extracted.append({
                'filename': frame_filename,
                'path': str(frame_path),
                'thumbnail': f"data:image/jpeg;base64,{thumb_base64}" if thumb_base64 else None,
                'frame_index': frame_idx,
                'timestamp': round(frame_idx / fps, 2) if fps > 0 else 0
            })

        cap.release()
    finally:
        # Delete the source video file
        try:
            video_path.unlink()
        except Exception:
            pass

    return jsonify({
        'video_name': video_file.filename,
        'duration': round(duration, 2),
        'total_video_frames': total_frames,
        'video_fps': round(fps, 2),
        'extracted': extracted
    })


def encode_image_for_analysis(image_path):
    """Create a lightweight 512px thumbnail for AI analysis (much smaller than full image)."""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        # Resize to max 512px on longest side — enough for Claude to judge quality
        img.thumbnail((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
    except Exception:
        return None, None


@app.route('/analyze-frames', methods=['POST'])
def analyze_frames():
    """Use Claude Vision to score extracted frames for a specific LoRA type."""
    data = request.json
    frames = data.get('frames', [])
    lora_type = data.get('lora_type', 'character')
    max_select = data.get('max_select', 10)

    if not frames:
        return jsonify({'error': 'No frames provided'}), 400

    # LoRA-type-specific evaluation criteria
    analysis_criteria = {
        'character': (
            "You are selecting the best frames for CHARACTER LoRA training. "
            "A good character dataset needs:\n"
            "- VARIETY of poses (standing, sitting, walking, different angles)\n"
            "- VARIETY of expressions (smiling, serious, talking, etc.)\n"
            "- Clear visibility of the character (not too far, not too blurry)\n"
            "- Different camera angles (front, side, three-quarter, full body, close-up)\n"
            "- Different lighting conditions and backgrounds\n"
            "- AVOID: duplicate/near-identical frames, motion blur, frames where character is obscured or off-screen\n"
            "Prioritize frames that show the character clearly from diverse angles and in diverse situations."
        ),
        'style': (
            "You are selecting the best frames for STYLE LoRA training. "
            "A good style dataset needs:\n"
            "- VARIETY of subject matter (different scenes, objects, compositions)\n"
            "- Frames that strongly showcase the visual style (color palette, rendering, texture, artistic quality)\n"
            "- Diverse compositions (wide shots, close-ups, different framing)\n"
            "- Consistent stylistic quality across selected frames\n"
            "- AVOID: visually bland/generic frames, transitional frames, frames with artifacts\n"
            "Prioritize frames with the richest stylistic content and the most diverse subjects."
        ),
        'object': (
            "You are selecting the best frames for OBJECT LoRA training. "
            "A good object dataset needs:\n"
            "- Clear, sharp views of the target object\n"
            "- VARIETY of angles (front, side, top, angled views)\n"
            "- Different scales (close-up details, medium shots showing context)\n"
            "- Different lighting conditions\n"
            "- Different backgrounds/environments showing the object in context\n"
            "- AVOID: frames where the object is occluded, blurry, or too small to see clearly\n"
            "Prioritize frames where the object is clearly visible from diverse angles and environments."
        ),
        'concept': (
            "You are selecting the best frames for CONCEPT LoRA training. "
            "A good concept dataset needs:\n"
            "- Frames that strongly exemplify the visual concept/pattern/effect\n"
            "- VARIETY of subjects/scenes that all demonstrate the concept\n"
            "- Different applications of the concept (varied contexts, scales, subjects)\n"
            "- Clear, high-quality frames where the concept is prominent\n"
            "- AVOID: frames where the concept is weak/unclear, transitional frames, ambiguous content\n"
            "Prioritize frames where the concept is most clearly demonstrated across diverse subjects."
        )
    }

    criteria = analysis_criteria.get(lora_type, analysis_criteria['character'])

    # Use lightweight 512px thumbnails for analysis, batch of 8 at a time
    BATCH_SIZE = 8
    all_scores = []
    total_batches = (len(frames) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Analyzing {len(frames)} frames in {total_batches} batches for {lora_type} LoRA...")

    for batch_start in range(0, len(frames), BATCH_SIZE):
        batch = frames[batch_start:batch_start + BATCH_SIZE]
        content_blocks = []
        valid_in_batch = 0

        for i, frame in enumerate(batch):
            frame_path = Path(frame['path'])
            if not frame_path.exists():
                all_scores.append({'index': batch_start + i, 'score': 0, 'reason': 'File not found'})
                continue

            b64_thumb, media_type = encode_image_for_analysis(str(frame_path))
            if not b64_thumb:
                all_scores.append({'index': batch_start + i, 'score': 0, 'reason': 'Encoding error'})
                continue

            content_blocks.append({
                "type": "text",
                "text": f"[Frame {batch_start + i + 1}] (timestamp: {frame.get('timestamp', '?')}s)"
            })
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_thumb
                }
            })
            valid_in_batch += 1

        if not content_blocks:
            continue

        batch_num = (batch_start // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{total_batches} ({valid_in_batch} frames)...")

        prompt = (
            f"{criteria}\n\n"
            f"I've shown you {valid_in_batch} frames extracted from a video. "
            f"Score each frame from 1-10 for suitability in a {lora_type} LoRA training dataset. "
            f"Consider both individual quality AND how well the frame contributes to dataset diversity.\n\n"
            f"Respond ONLY with a JSON array, one object per frame, in order:\n"
            f'[{{"frame": 1, "score": 8, "reason": "clear front view, good lighting"}}, ...]\n\n'
            f"Include ALL {valid_in_batch} frames in your response. No extra text, just the JSON array."
        )
        content_blocks.append({"type": "text", "text": prompt})

        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": content_blocks}]
            )

            response_text = message.content[0].text.strip()
            # Extract JSON from response (handle potential markdown code blocks)
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1]
                response_text = response_text.rsplit('```', 1)[0]
            response_text = response_text.strip()

            batch_scores = json.loads(response_text)
            for item in batch_scores:
                idx = batch_start + item['frame'] - 1
                all_scores.append({
                    'index': idx,
                    'score': item.get('score', 5),
                    'reason': item.get('reason', '')
                })
        except Exception as e:
            print(f"  Analysis batch error: {e}")
            # Fallback: give all frames in this batch a neutral score
            for i in range(len(batch)):
                if not any(s['index'] == batch_start + i for s in all_scores):
                    all_scores.append({
                        'index': batch_start + i,
                        'score': 5,
                        'reason': 'Analysis failed, neutral score'
                    })

    # Sort by score descending and pick top N
    all_scores.sort(key=lambda x: x['score'], reverse=True)
    recommended_indices = [s['index'] for s in all_scores[:max_select]]

    print(f"Analysis complete. Recommended {len(recommended_indices)} frames.")

    return jsonify({
        'scores': sorted(all_scores, key=lambda x: x['index']),
        'recommended': recommended_indices,
        'lora_type': lora_type
    })


@app.route('/check-gemini', methods=['GET'])
def check_gemini():
    """Check if Gemini API is configured."""
    return jsonify({'available': gemini_client is not None})


# --- Character sheet prompt helpers & item descriptions ---
CHARSHEET_ITEMS = {
    'views': {
        'front': 'front view',
        'side': 'side profile view (left or right)',
        'back': 'back view',
        'three-quarter': 'three-quarter view',
        'close-up': 'face close-up',
    },
    'expressions': {
        'happy': 'happy / joyful expression',
        'angry': 'angry / furious expression',
        'sad': 'sad / sorrowful expression',
        'surprised': 'surprised / shocked expression',
        'disgusted': 'disgusted expression',
        'fearful': 'fearful / scared expression',
        'neutral': 'neutral / calm expression',
        'smirking': 'smirking / sly expression',
    },
    'actions': {
        'standing': 'standing in a neutral pose',
        'running': 'running / sprinting',
        'jumping': 'jumping / leaping',
        'fighting': 'fighting / combat stance',
        'sitting': 'sitting down',
        'crouching': 'crouching / kneeling',
        'waving': 'waving hello',
        'casting': 'casting a spell / using power',
    },
}


def _gemini_generate(ref_image, prompt_text, base_filename):
    """Call Gemini image generation and return list of {filename, path, thumbnail} dicts."""
    import time
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt_text, ref_image],
        config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image']
        )
    )

    generated_images = []
    response_text = ''

    for part in response.parts:
        if hasattr(part, 'text') and part.text is not None:
            response_text = part.text
        elif hasattr(part, 'inline_data') and part.inline_data is not None:
            try:
                img_data = part.inline_data.data
                ts = int(time.time() * 1000)
                sheet_filename = f"charsheet_{base_filename}_{ts}_{len(generated_images) + 1}.png"
                sheet_path = UPLOAD_FOLDER / sheet_filename
                pil_img = Image.open(io.BytesIO(img_data))
                pil_img.save(sheet_path, format='PNG')

                thumb_img = pil_img.copy()
                thumb_img.thumbnail((400, 400), Image.LANCZOS)
                thumb_buffer = io.BytesIO()
                thumb_img.save(thumb_buffer, format='PNG')
                thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()

                generated_images.append({
                    'filename': sheet_filename,
                    'path': str(sheet_path),
                    'thumbnail': f"data:image/png;base64,{thumb_base64}"
                })
            except Exception as img_err:
                print(f"Error processing generated image: {img_err}")

    return generated_images, response_text


def _build_charsheet_prompt(tab, items_text, style, mode):
    """Build the Gemini prompt based on tab type and mode."""
    consistency = (
        "CRITICAL: Maintain perfect consistency — same outfit, colors, proportions, "
        "hairstyle, and features as the reference image. "
    )

    if mode == 'grouped':
        layout = (
            "Generate a single image showing this exact character in ALL of the following arranged in a row. "
            "Clean white background. Each pose/view should be clearly separated and labeled at the bottom. "
            "Professional character sheet layout suitable for animation or game production reference. "
        )
    else:
        layout = (
            "Generate a single, clean image of ONLY this one pose/view. "
            "Clean white background. High quality, detailed, full body visible. "
        )

    if tab == 'views':
        return (
            f"Create a professional character reference sheet / turnaround sheet for this exact character. "
            f"{layout}"
            f"Views to show: {items_text}. "
            f"Style: {style}. "
            f"{consistency}"
        )
    elif tab == 'expressions':
        return (
            f"Create a professional character expression sheet for this exact character. "
            f"{layout}"
            f"Expressions to show: {items_text}. "
            f"Show the character's face and upper body clearly for each expression. "
            f"Style: {style}. "
            f"{consistency}"
        )
    elif tab == 'actions':
        return (
            f"Create a professional character action/pose sheet for this exact character. "
            f"{layout}"
            f"Actions/poses to show: {items_text}. "
            f"Show the full body in dynamic poses. "
            f"Style: {style}. "
            f"{consistency}"
        )
    else:
        # Custom prompt — user provides their own text
        return (
            f"{items_text} "
            f"Style: {style}. "
            f"{consistency}"
            f"Clean white background. Professional character reference quality."
        )


@app.route('/generate-character-sheet', methods=['POST'])
def generate_character_sheet():
    """Generate a character sheet (grouped mode) from a reference image using Gemini."""
    if not gemini_client:
        return jsonify({'error': 'Google API key not configured. Add GOOGLE_API_KEY to your .env file.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if not image_file.filename:
        return jsonify({'error': 'No image selected'}), 400

    style = request.form.get('style', 'anime')
    tab = request.form.get('tab', 'views')
    items = request.form.get('items', 'front,side,back')
    custom_prompt = request.form.get('custom_prompt', '')

    # Save reference image temporarily
    ref_filename = f"charsheet_ref_{image_file.filename}"
    ref_path = UPLOAD_FOLDER / ref_filename
    image_file.save(ref_path)

    try:
        ref_image = Image.open(ref_path)

        # Build item descriptions
        if tab == 'custom':
            items_text = custom_prompt
        else:
            item_list = [v.strip() for v in items.split(',')]
            item_descs = CHARSHEET_ITEMS.get(tab, CHARSHEET_ITEMS['views'])
            items_text = ', '.join(item_descs.get(v, v) for v in item_list)

        prompt = _build_charsheet_prompt(tab, items_text, style, mode='grouped')
        print(f"Generating character sheet (grouped): tab={tab}, items={items}, style={style}")

        generated_images, response_text = _gemini_generate(
            ref_image, prompt, Path(image_file.filename).stem
        )

        if not generated_images:
            return jsonify({
                'error': f'No images generated. Model response: {response_text or "empty"}',
            }), 500

        print(f"Character sheet generated: {len(generated_images)} image(s)")

        return jsonify({
            'images': generated_images,
            'text': response_text,
            'reference': ref_filename
        })

    except Exception as e:
        print(f"Character sheet generation error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            ref_path.unlink()
        except Exception:
            pass


@app.route('/generate-character-sheet-single', methods=['POST'])
def generate_character_sheet_single():
    """Generate a SINGLE character sheet item (singular mode) — called once per item."""
    if not gemini_client:
        return jsonify({'error': 'Google API key not configured.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if not image_file.filename:
        return jsonify({'error': 'No image selected'}), 400

    style = request.form.get('style', 'anime')
    tab = request.form.get('tab', 'views')
    item = request.form.get('item', 'front')
    custom_prompt = request.form.get('custom_prompt', '')

    # Save reference image temporarily
    ref_filename = f"charsheet_ref_{image_file.filename}"
    ref_path = UPLOAD_FOLDER / ref_filename
    image_file.save(ref_path)

    try:
        ref_image = Image.open(ref_path)

        # Build single-item description
        if tab == 'custom':
            items_text = custom_prompt
        else:
            item_descs = CHARSHEET_ITEMS.get(tab, CHARSHEET_ITEMS['views'])
            items_text = item_descs.get(item, item)

        prompt = _build_charsheet_prompt(tab, items_text, style, mode='singular')
        print(f"Generating character sheet (singular): tab={tab}, item={item}, style={style}")

        generated_images, response_text = _gemini_generate(
            ref_image, prompt, f"{Path(image_file.filename).stem}_{item}"
        )

        if not generated_images:
            return jsonify({
                'error': f'No image generated for "{item}". Model response: {response_text or "empty"}',
            }), 500

        return jsonify({
            'images': generated_images,
            'text': response_text,
            'item': item,
            'reference': ref_filename
        })

    except Exception as e:
        print(f"Character sheet single generation error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            ref_path.unlink()
        except Exception:
            pass


@app.route('/save-charsheet', methods=['POST'])
def save_charsheet():
    """Save a generated character sheet to a user-chosen folder."""
    data = request.json
    filename = data.get('filename', '')
    folder = data.get('folder', '')

    if not filename or not folder:
        return jsonify({'error': 'Missing filename or folder'}), 400

    # Safety: only serve files from UPLOAD_FOLDER
    src = UPLOAD_FOLDER / Path(filename).name
    if not src.exists() or src.parent.resolve() != UPLOAD_FOLDER.resolve():
        return jsonify({'error': 'File not found'}), 404

    dest_folder = Path(folder)
    if not dest_folder.is_dir():
        return jsonify({'error': 'Destination folder does not exist'}), 400

    try:
        dest = dest_folder / src.name
        shutil.copy2(str(src), str(dest))
        return jsonify({'saved': str(dest)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cleanup-frames', methods=['POST'])
def cleanup_frames():
    """Delete extracted frame files that the user deselected."""
    data = request.json
    filenames = data.get('filenames', [])
    deleted = 0
    for filename in filenames:
        # Safety: only delete from UPLOAD_FOLDER, prevent path traversal
        filepath = UPLOAD_FOLDER / Path(filename).name
        if filepath.exists() and filepath.parent.resolve() == UPLOAD_FOLDER.resolve():
            filepath.unlink()
            deleted += 1
    return jsonify({'deleted': deleted})


def generate_toolkit_yaml(params, dataset_path):
    """Generate ai-toolkit YAML config string."""
    MODEL_PRESETS = {
        'flux-dev': {
            'name_or_path': 'black-forest-labs/FLUX.1-dev',
            'is_flux': True,
            'quantize': True,
            'noise_scheduler': 'flowmatch',
            'sampler': 'flowmatch',
            'resolution': 1024,
            'train_text_encoder': False,
        },
        'flux-schnell': {
            'name_or_path': 'black-forest-labs/FLUX.1-schnell',
            'is_flux': True,
            'quantize': True,
            'noise_scheduler': 'flowmatch',
            'sampler': 'flowmatch',
            'resolution': 1024,
            'train_text_encoder': False,
        },
        'sdxl': {
            'name_or_path': 'stabilityai/stable-diffusion-xl-base-1.0',
            'is_flux': False,
            'quantize': False,
            'noise_scheduler': 'ddpm',
            'sampler': 'ddpm',
            'resolution': 1024,
            'train_text_encoder': False,
        },
        'sd15': {
            'name_or_path': 'runwayml/stable-diffusion-v1-5',
            'is_flux': False,
            'quantize': False,
            'noise_scheduler': 'ddpm',
            'sampler': 'ddpm',
            'resolution': 512,
            'train_text_encoder': True,
        },
    }

    model_key = params.get('base_model', 'flux-dev')
    preset = MODEL_PRESETS.get(model_key, MODEL_PRESETS['flux-dev'])

    name = params.get('name', 'my_lora_v1')
    trigger_word = params.get('trigger_word', '')
    lora_rank = params.get('lora_rank', 16)
    learning_rate = params.get('learning_rate', '4e-4')
    steps = params.get('steps', 2500)
    resolution = params.get('resolution', preset['resolution'])
    batch_size = params.get('batch_size', 1)
    save_every = params.get('save_every', 250)
    sample_every = params.get('sample_every', 250)
    sample_prompts = params.get('sample_prompts', [])
    caption_dropout = params.get('caption_dropout', 0.05)
    gradient_accumulation = params.get('gradient_accumulation', 1)

    # Normalize Windows backslash paths for ai-toolkit compatibility
    dataset_path = dataset_path.replace('\\', '/')

    lines = [
        '---',
        'job: extension',
        'config:',
        f'  name: "{name}"',
        '  process:',
        '    - type: sd_trainer',
        '      training_folder: "output"',
        '      device: cuda:0',
    ]

    if trigger_word:
        lines.append(f'      trigger_word: "{trigger_word}"')

    lines += [
        '      network:',
        '        type: "lora"',
        f'        linear: {lora_rank}',
        f'        linear_alpha: {lora_rank}',
        '      save:',
        '        dtype: float16',
        f'        save_every: {save_every}',
        '      datasets:',
        f'        - folder_path: "{dataset_path}"',
        '          caption_ext: "txt"',
        f'          caption_dropout_rate: {caption_dropout}',
        f'          resolution: [{resolution}, {resolution}]',
        '          cache_latents_to_disk: true',
        '      train:',
        f'        batch_size: {batch_size}',
        f'        steps: {steps}',
        f'        gradient_accumulation_steps: {gradient_accumulation}',
        '        train_unet: true',
        f'        train_text_encoder: {str(preset["train_text_encoder"]).lower()}',
        '        gradient_checkpointing: true',
        f'        noise_scheduler: "{preset["noise_scheduler"]}"',
        '        optimizer: "adamw8bit"',
        f'        lr: {learning_rate}',
        '      model:',
        f'        name_or_path: "{preset["name_or_path"]}"',
    ]

    if preset['is_flux']:
        lines.append('        is_flux: true')
    if preset['quantize']:
        lines.append('        quantize: true')

    lines += [
        '      sample:',
        f'        sampler: "{preset["sampler"]}"',
        f'        sample_every: {sample_every}',
        f'        width: {resolution}',
        f'        height: {resolution}',
        '        prompts:',
    ]

    if sample_prompts:
        for prompt in sample_prompts:
            lines.append(f'          - "{prompt}"')
    else:
        lines.append('          []')

    return '\n'.join(lines) + '\n'


@app.route('/export-toolkit', methods=['POST'])
def export_toolkit():
    """Export captioned dataset to a folder in ai-toolkit format."""
    data = request.json
    images = data.get('images', [])
    captions_map = data.get('captions', {})
    output_dir = Path(data.get('output_dir', ''))
    config_params = data.get('config', {})

    if not output_dir or not images:
        return jsonify({'error': 'Missing output directory or images'}), 400

    dataset_dir = output_dir / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img in images:
        filename = img['filename']
        src = UPLOAD_FOLDER / Path(filename).name
        if not src.exists():
            continue
        # Copy image
        shutil.copy2(str(src), str(dataset_dir / filename))
        # Write caption
        caption_text = captions_map.get(filename, '')
        if caption_text:
            txt_path = dataset_dir / (Path(filename).stem + '.txt')
            txt_path.write_text(caption_text, encoding='utf-8')
        copied += 1

    # Generate YAML config
    yaml_content = generate_toolkit_yaml(config_params, str(dataset_dir))
    config_path = output_dir / 'config.yaml'
    config_path.write_text(yaml_content, encoding='utf-8')

    return jsonify({
        'status': 'success',
        'path': str(output_dir),
        'image_count': copied,
        'config_path': str(config_path)
    })


@app.route('/export-toolkit-zip', methods=['POST'])
def export_toolkit_zip():
    """Export captioned dataset as a downloadable zip in ai-toolkit format."""
    data = request.json
    images = data.get('images', [])
    captions_map = data.get('captions', {})
    config_params = data.get('config', {})

    if not images:
        return jsonify({'error': 'No images to export'}), 400

    output_name = config_params.get('name', 'lora_dataset')

    # Build zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            filename = img['filename']
            src = UPLOAD_FOLDER / Path(filename).name
            if not src.exists():
                continue
            # Add image under dataset/ folder
            zf.write(str(src), f'dataset/{filename}')
            # Add caption .txt
            caption_text = captions_map.get(filename, '')
            if caption_text:
                txt_name = Path(filename).stem + '.txt'
                zf.writestr(f'dataset/{txt_name}', caption_text)

        # Generate and add YAML config with relative dataset path
        yaml_content = generate_toolkit_yaml(config_params, './dataset')
        zf.writestr('config.yaml', yaml_content)

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{output_name}.zip'
    )


if __name__ == '__main__':
    print("=" * 50)
    print("LoRA Image Captioner")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5000)
